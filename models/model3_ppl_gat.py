"""
AEGIS Model 3 — Perplexity-Aware GAT with Learnable Edge Weights.

Node features:  RoBERTa 768d contextual embeddings.
Edge features:  4 components per edge — cosine similarity, attention score,
                token entropy, token perplexity — combined via *learnable*
                softmax-weighted sum.
Architecture:   3× GATConv (8 heads) + residual + LayerNorm,
                global mean+add pooling, auxiliary heads.
"""

import os
import math
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, ModuleList, Parameter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.utils
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModelForMaskedLM, AutoTokenizer
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========== CONFIGURATION ==========
BATCH_SIZE = 16
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 30
MAX_TOKENS = 256
K_NEIGHBORS = 8
LABEL_SMOOTHING = 0.1
CONTEXTUAL_MODEL = "roberta-base"

UNIVERSAL_POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def standardize_text(text):
    if not isinstance(text, str):
        return ""
    paragraphs = text.strip().split("\n")
    return "\n".join(p.strip() for p in paragraphs if p.strip())


def normalize_score(score, min_val=0.0, max_val=1.0):
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))


# ========== DATA PROCESSING ==========

class EnhancedTextGraphDataset(Dataset):
    """Builds graphs with 4-component edge attributes (cosine sim, attention, entropy, perplexity)."""

    def __init__(self, dataframe, tokenizer, language_model, nlp, vocab_size, max_tokens=MAX_TOKENS):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.nlp = nlp
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.pos_tag_map = {tag: i for i, tag in enumerate(UNIVERSAL_POS_TAGS)}
        self.log_vocab_size = math.log(vocab_size) if vocab_size > 1 else 1.0

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row["text"]).lower()
        doc = self.nlp(text[:self.nlp.max_length - 100])

        inputs = self.tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_tokens, return_attention_mask=True,
        )
        input_ids = inputs["input_ids"][0].to(DEVICE)
        attention_mask = inputs["attention_mask"][0].to(DEVICE)
        n_tokens = min(int(attention_mask.sum().item()), self.max_tokens)

        with torch.no_grad():
            out = self.language_model(
                input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0),
                output_attentions=True, output_hidden_states=True,
            )
            token_embs = out.hidden_states[-1][0, :n_tokens].cpu().numpy()

            attn_matrix = None
            if out.attentions:
                attn_matrix = out.attentions[-1][0].mean(dim=0).cpu().numpy()[:n_tokens, :n_tokens]

            logits = out.logits[0, :n_tokens]
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()

            entropy_norm = np.clip(entropy / self.log_vocab_size, 0.0, 1.0) if self.log_vocab_size > 0 else np.zeros_like(entropy)
            ppl_norm = np.clip((np.exp(entropy) - 1.0) / (self.vocab_size - 1.0), 0.0, 1.0) if self.vocab_size > 1 else np.zeros_like(entropy)

        if n_tokens == 0:
            token_embs = np.zeros((0, self.language_model.config.hidden_size))

        x, edge_index, edge_attr = self._build_knn_graph(
            token_embs, n_tokens, attn_matrix, entropy_norm, ppl_norm,
        )

        y = torch.tensor([row["generated"]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.complexity_score = torch.tensor([row.get("complexity", 0.5)], dtype=torch.float)
        data.readability_label = torch.tensor([row.get("readability", 1)], dtype=torch.long)
        return data

    def _build_knn_graph(self, node_features, n_tokens, attn_matrix, entropy_norm, ppl_norm):
        if n_tokens <= 1:
            x = torch.tensor(node_features, dtype=torch.float)
            return x, torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float)

        valid_feats = node_features[:n_tokens]
        knn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS + 1, n_tokens), metric="cosine")
        knn.fit(valid_feats)
        _, indices = knn.kneighbors(valid_feats)

        rows, cols, components = [], [], []
        valid_t = torch.tensor(valid_feats, dtype=torch.float)

        for i in range(n_tokens):
            fi = valid_t[i].unsqueeze(0)
            for j in indices[i][1:]:
                fj = valid_t[j].unsqueeze(0)
                cos_sim = normalize_score(F.cosine_similarity(fi, fj).item(), -1, 1)
                attn = normalize_score(attn_matrix[i, j], 0, 1) if attn_matrix is not None and i < attn_matrix.shape[0] and j < attn_matrix.shape[1] else 0.0
                ent = (entropy_norm[i] + entropy_norm[j]) / 2.0 if i < len(entropy_norm) and j < len(entropy_norm) else 0.0
                ppl = (ppl_norm[i] + ppl_norm[j]) / 2.0 if i < len(ppl_norm) and j < len(ppl_norm) else 0.0

                comp = [cos_sim, attn, ent, ppl]
                rows.extend([i, j])
                cols.extend([j, i])
                components.extend([comp, comp])

        if not rows:
            x = torch.tensor(node_features, dtype=torch.float)
            return x, torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float)

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(components, dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)

        if edge_index.numel() > 0:
            edge_index, edge_attr = torch_geometric.utils.coalesce(
                edge_index, edge_attr, num_nodes=x.size(0), reduce="mean",
            )
        return x, edge_index, edge_attr


# ========== MODEL ==========

class EnhancedTextGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers=3, dropout_rate=0.3):
        super().__init__()
        self.num_layers = num_layers

        self.edge_component_weights = Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        self.input_proj = Linear(num_node_features, hidden_channels)

        self.gat_layers = ModuleList()
        self.layer_norms = ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_channels, hidden_channels // 8, heads=8,
                        concat=True, dropout=dropout_rate,
                        add_self_loops=True, edge_dim=1)
            )
            self.layer_norms.append(LayerNorm(hidden_channels))

        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.dropout = Dropout(dropout_rate)
        self.lin2 = Linear(hidden_channels, 2)
        self.complexity_regressor = Linear(hidden_channels, 1)
        self.readability_classifier = Linear(hidden_channels, 3)

    def forward(self, x, edge_index, edge_attr_components, batch):
        x = self.input_proj(x)

        w = F.softmax(self.edge_component_weights, dim=0)
        scalar_ea = (edge_attr_components * w.unsqueeze(0)).sum(dim=1, keepdim=True)

        for i in range(self.num_layers):
            gat_out = self.gat_layers[i](x, edge_index, edge_attr=scalar_ea)
            x = gat_out + x if i > 0 else gat_out
            x = F.elu(self.layer_norms[i](x))
            if i < self.num_layers - 1:
                x = self.dropout(x)

        x_pool = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        h = F.relu(self.lin1(x_pool))
        h = self.dropout(h)
        return self.lin2(h), self.complexity_regressor(h), self.readability_classifier(h)

    def predict(self, x, edge_index, edge_attr_components, batch):
        main_out, _, _ = self.forward(x, edge_index, edge_attr_components, batch)
        return main_out


# ========== AUXILIARY LABELS ==========

def compute_text_complexity(text):
    words = str(text).lower().split()
    if not words:
        return 0.0
    unique = len(set(words)) / len(words)
    avg_len = sum(len(w) for w in words) / len(words)
    long_ratio = sum(1 for w in words if len(w) > 6) / len(words)
    return min(max(0.4 * unique + 0.3 * min(avg_len / 10, 1.0) + 0.3 * long_ratio, 0.0), 1.0)


def compute_readability_class(text):
    text = str(text).strip()
    if len(text) < 50:
        return 0
    sentences = text.split(". ")
    n_sent = len(sentences) or 1
    total_words = sum(len(s.split()) for s in sentences)
    total_chars = sum(sum(len(w) for w in s.split()) for s in sentences)
    if total_words == 0:
        return 0
    fk = 0.39 * (total_words / n_sent) + 11.8 * (total_chars / total_words) - 15.59
    return 0 if fk < 8 else (1 if fk < 12 else 2)


# ========== TRAINING / EVALUATION ==========

def train_epoch(model, loader, optimizer, scheduler, criterions, device, scaler=None):
    criterion_main, crit_complex, crit_read = criterions
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()

        comp_targets = data.complexity_score
        if comp_targets.ndim == 1:
            comp_targets = comp_targets.unsqueeze(1)

        main_out, comp_out, read_out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion_main(main_out, data.y.view(-1)) + \
               0.2 * crit_complex(comp_out, comp_targets) + \
               0.3 * crit_read(read_out, data.readability_label.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        correct += (main_out.argmax(1) == data.y.view(-1)).sum().item()
        total += data.y.size(0)

    scheduler.step()
    w = F.softmax(model.edge_component_weights.detach(), dim=0).cpu().numpy()
    print(f"  Learned edge weights (softmax): {w}")
    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for data in tqdm(loader, desc="Evaluating"):
        data = data.to(device)
        main_out, _, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(main_out, data.y.view(-1))
        total_loss += loss.item() * data.num_graphs
        pred = main_out.argmax(1)
        correct += (pred == data.y.view(-1)).sum().item()
        total += data.y.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(data.y.view(-1).cpu().tolist())

    return total_loss / len(loader.dataset), correct / total, all_preds, all_labels


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description="AEGIS Model 3 — Perplexity-Aware GAT")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    df = pd.read_csv(args.data)
    df["text"] = df["text"].apply(standardize_text)
    df = df[["text", "generated"]].astype({"generated": int})

    df["complexity"] = df["text"].apply(compute_text_complexity)
    df["readability"] = df["text"].apply(compute_readability_class)
    print(f"Loaded {len(df)} samples")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["generated"])

    print(f"Loading {CONTEXTUAL_MODEL} (masked LM) ...")
    tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
    language_model = AutoModelForMaskedLM.from_pretrained(CONTEXTUAL_MODEL).to(DEVICE)
    language_model.eval()
    vocab_size = tokenizer.vocab_size

    print("Loading spaCy ...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    train_dataset = EnhancedTextGraphDataset(train_df, tokenizer, language_model, nlp, vocab_size)
    test_dataset = EnhancedTextGraphDataset(test_df, tokenizer, language_model, nlp, vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sample = train_dataset[0]
    num_features = sample.x.size(1)
    print(f"Node feature dimension: {num_features}")

    model = EnhancedTextGNN(num_features, HIDDEN_CHANNELS, NUM_LAYERS).to(DEVICE)

    y_train = train_df["generated"].values
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    criterion_main = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    crit_complex = torch.nn.MSELoss()
    crit_read = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            (criterion_main, crit_complex, crit_read), DEVICE,
        )
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion_main, DEVICE)
        print(f"  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), "checkpoints/model3_best.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(classification_report(labels, preds, digits=4, zero_division=0))

    torch.save(model.state_dict(), "checkpoints/model3_final.pt")
    w = F.softmax(model.edge_component_weights.detach(), dim=0).cpu().numpy()
    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Final learned edge weights (alpha, beta, gamma, delta): {w}")


if __name__ == "__main__":
    main()
