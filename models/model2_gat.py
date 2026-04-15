"""
AEGIS Model 2 — Enhanced GAT with Multi-Task Learning.

Node features: RoBERTa 768d contextual embeddings + POS tags (19) + position (1)
               + stopword/punctuation indicators (2) = 790 total.
Graph:         KNN graph (k=8) from feature space.
Architecture:  3× GATConv (8 heads) with residual connections + LayerNorm,
               global mean+add pooling, auxiliary complexity/readability heads.
"""

import os
import argparse
import subprocess
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, ModuleList
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModel, AutoTokenizer
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
NUM_EPOCHS = 20
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
    paragraphs = text.strip().split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return "\n".join(paragraphs)


# ========== DATA PROCESSING ==========

class EnhancedTextGraphDataset(Dataset):
    def __init__(self, dataframe, tokenizer, language_model, nlp, max_tokens=MAX_TOKENS):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.nlp = nlp
        self.max_tokens = max_tokens
        self.pos_tag_map = {tag: i for i, tag in enumerate(UNIVERSAL_POS_TAGS)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row["text"]).lower()

        doc = self.nlp(text[:10000])

        tokens, pos_tags, is_stop, is_punct = [], [], [], []
        for token in doc:
            tokens.append(token.text)
            pos_tags.append(token.pos_)
            is_stop.append(int(token.is_stop))
            is_punct.append(int(token.is_punct))
            if len(tokens) >= self.max_tokens:
                break

        if not tokens:
            tokens, pos_tags, is_stop, is_punct = ["<pad>"], ["X"], [0], [0]

        text_input = " ".join(tokens)
        inputs = self.tokenizer(
            text_input, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_tokens,
        )

        with torch.no_grad():
            outputs = self.language_model(**{k: v.to(DEVICE) for k, v in inputs.items()})
            embeddings = outputs.last_hidden_state[0, :min(len(tokens) + 1, self.max_tokens)].cpu().numpy()
            if len(embeddings) > len(tokens):
                embeddings = embeddings[1:len(tokens) + 1]

        if len(embeddings) < len(tokens):
            padding = np.zeros((len(tokens) - len(embeddings), embeddings.shape[1]))
            embeddings = np.vstack([embeddings, padding])
        elif len(embeddings) > len(tokens):
            embeddings = embeddings[:len(tokens)]

        num_tokens = len(tokens)

        pos_encoding = np.zeros((num_tokens, len(UNIVERSAL_POS_TAGS)))
        for i, pos in enumerate(pos_tags):
            pos_encoding[i, self.pos_tag_map.get(pos, self.pos_tag_map["X"])] = 1

        position = np.arange(num_tokens).reshape(-1, 1) / num_tokens
        stop_punct = np.column_stack((np.array(is_stop), np.array(is_punct)))

        node_features = np.hstack([embeddings, pos_encoding, position, stop_punct])
        x, edge_index, edge_attr = self._build_knn_graph(node_features)

        if edge_index.size(1) > 0:
            edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float)
        else:
            dep_edges = []
            for tok in doc:
                if tok.i < len(tokens) and tok.head.i < len(tokens) and tok.i != tok.head.i:
                    dep_edges.extend([(tok.i, tok.head.i), (tok.head.i, tok.i)])
            if dep_edges:
                edge_index = torch.tensor(dep_edges, dtype=torch.long).t()
                edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float)

        y = torch.tensor([row["generated"]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        sentence_boundaries = [0]
        for sent in doc.sents:
            if sent.end < self.max_tokens and sent.end < len(tokens):
                sentence_boundaries.append(sent.end)
        if len(sentence_boundaries) <= 1:
            sentence_boundaries.append(min(len(tokens), self.max_tokens))
        data.sentence_boundaries = torch.tensor(sentence_boundaries, dtype=torch.long)

        return data

    def _build_knn_graph(self, node_features, k=K_NEIGHBORS):
        num_nodes = len(node_features)
        if num_nodes <= 1:
            x = torch.tensor(node_features, dtype=torch.float)
            return x, torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

        knn = NearestNeighbors(n_neighbors=min(k + 1, num_nodes))
        knn.fit(node_features)
        distances, indices = knn.kneighbors(node_features)

        rows, cols, weights = [], [], []
        for i, (neighbors, dist) in enumerate(zip(indices, distances)):
            for j, d in zip(neighbors[1:], dist[1:]):
                sim = 1.0 / (1.0 + d)
                rows.extend([i, j])
                cols.extend([j, i])
                weights.extend([sim, sim])

        if not rows:
            x = torch.tensor(node_features, dtype=torch.float)
            return x, torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float).view(-1, 1)
        return torch.tensor(node_features, dtype=torch.float), edge_index, edge_attr


# ========== MODEL ==========

class EnhancedTextGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers=3, dropout_rate=0.3):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

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

        self.pool_attention = Linear(hidden_channels, 1)
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.dropout = Dropout(dropout_rate)
        self.lin2 = Linear(hidden_channels, 2)
        self.complexity_regressor = Linear(hidden_channels, 1)
        self.readability_classifier = Linear(hidden_channels, 3)

    def forward(self, x, edge_index, edge_attr, batch, sentence_boundaries=None):
        x = self.input_proj(x)
        for i in range(self.num_layers):
            gat_out = self.gat_layers[i](x, edge_index, edge_attr=edge_attr)
            x = gat_out + x if i > 0 else gat_out
            x = F.elu(self.layer_norms[i](x))
            if i < self.num_layers - 1:
                x = self.dropout(x)

        x_pool = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        h = F.relu(self.lin1(x_pool))
        h = self.dropout(h)
        return self.lin2(h), self.complexity_regressor(h), self.readability_classifier(h)

    def predict(self, x, edge_index, edge_attr, batch, sentence_boundaries=None):
        main_output, _, _ = self.forward(x, edge_index, edge_attr, batch, sentence_boundaries)
        return main_output


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
    criterion, crit_complex, crit_read = criterions
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()

        texts = [loader.dataset.dataframe.iloc[i]["text"] for i in range(len(data.y))]
        read_labels = torch.tensor([compute_readability_class(t) for t in texts], dtype=torch.long).to(device)
        comp_scores = torch.tensor([[compute_text_complexity(t)] for t in texts], dtype=torch.float).to(device)

        main_out, comp_out, read_out = model(
            data.x, data.edge_index, data.edge_attr, data.batch,
            sentence_boundaries=getattr(data, "sentence_boundaries", None),
        )
        loss = criterion(main_out, data.y.view(-1)) + \
               0.2 * crit_complex(comp_out, comp_scores) + \
               0.3 * crit_read(read_out, read_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        correct += (main_out.argmax(1) == data.y.view(-1)).sum().item()
        total += data.y.size(0)

    scheduler.step()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for data in tqdm(loader, desc="Evaluating"):
        data = data.to(device)
        main_out, _, _ = model(
            data.x, data.edge_index, data.edge_attr, data.batch,
            sentence_boundaries=getattr(data, "sentence_boundaries", None),
        )
        loss = criterion(main_out, data.y.view(-1))
        total_loss += loss.item() * data.num_graphs
        pred = main_out.argmax(1)
        correct += (pred == data.y.view(-1)).sum().item()
        total += data.y.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(data.y.view(-1).cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description="AEGIS Model 2 — Enhanced GAT")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    df = pd.read_csv(args.data)
    df["text"] = df["text"].apply(standardize_text)
    df = df[["text", "generated"]].astype({"generated": int})
    print(f"Loaded {len(df)} samples")

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["generated"])

    print(f"Loading {CONTEXTUAL_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
    language_model = AutoModel.from_pretrained(CONTEXTUAL_MODEL).to(DEVICE)

    print("Loading spaCy ...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    train_dataset = EnhancedTextGraphDataset(train_df, tokenizer, language_model, nlp)
    test_dataset = EnhancedTextGraphDataset(test_df, tokenizer, language_model, nlp)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sample = next(iter(train_loader))
    num_features = sample.x.size(1)
    print(f"Node feature dimension: {num_features}")

    model = EnhancedTextGNN(num_features, HIDDEN_CHANNELS, NUM_LAYERS).to(DEVICE)

    y_train = train_df["generated"].values
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    crit_complex = torch.nn.MSELoss()
    crit_read = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            (criterion, crit_complex, crit_read), DEVICE,
        )
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
        print(f"Epoch {epoch:02d}  Train: {train_acc:.4f}  Val: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), "checkpoints/model2_best.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(classification_report(labels, preds, digits=4))

    torch.save(model.state_dict(), "checkpoints/model2_final.pt")
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
