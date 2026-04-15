"""
GNN inference utilities for Model 2 (Enhanced GAT).

Provides:
  - EnhancedTextGNN model class
  - Text-to-graph preprocessing (process_text)
  - Model loading (load_model)
  - Single-text prediction (predict_text)

Usage:
    python gnn_utils.py --model checkpoints/model2_best.pt --input "Some text to classify"
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, ModuleList
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer
import spacy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
MAX_TOKENS = 256
K_NEIGHBORS = 8
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
        main_out, _, _ = self.forward(x, edge_index, edge_attr, batch, sentence_boundaries)
        return main_out


# ========== TEXT → GRAPH ==========

def build_knn_graph(node_features, k=K_NEIGHBORS):
    num_nodes = len(node_features)
    if num_nodes <= 1:
        x = torch.tensor(node_features, dtype=torch.float)
        return x, torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

    knn = NearestNeighbors(n_neighbors=min(k + 1, num_nodes))
    knn.fit(node_features)
    distances, indices = knn.kneighbors(node_features)

    rows, cols, weights = [], [], []
    for i, (nbrs, dist) in enumerate(zip(indices, distances)):
        for j, d in zip(nbrs[1:], dist[1:]):
            sim = 1.0 / (1.0 + d)
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([sim, sim])

    if not rows:
        x = torch.tensor(node_features, dtype=torch.float)
        return x, torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

    return (
        torch.tensor(node_features, dtype=torch.float),
        torch.tensor([rows, cols], dtype=torch.long),
        torch.ones(len(rows), 1, dtype=torch.float),
    )


def process_text(text, tokenizer, language_model, nlp, max_tokens=MAX_TOKENS):
    """Convert a text string into a PyG Data object (single graph)."""
    pos_tag_map = {tag: i for i, tag in enumerate(UNIVERSAL_POS_TAGS)}
    text = text.lower()
    doc = nlp(text[:10000])

    tokens, pos_tags, is_stop, is_punct = [], [], [], []
    for tok in doc:
        tokens.append(tok.text)
        pos_tags.append(tok.pos_)
        is_stop.append(int(tok.is_stop))
        is_punct.append(int(tok.is_punct))
        if len(tokens) >= max_tokens:
            break

    if not tokens:
        tokens, pos_tags, is_stop, is_punct = ["<pad>"], ["X"], [0], [0]

    inputs = tokenizer(" ".join(tokens), return_tensors="pt", padding=True,
                        truncation=True, max_length=max_tokens)

    with torch.no_grad():
        outputs = language_model(**{k: v.to(DEVICE) for k, v in inputs.items()})
        embs = outputs.last_hidden_state[0, :min(len(tokens) + 1, max_tokens)].cpu().numpy()
        if len(embs) > len(tokens):
            embs = embs[1:len(tokens) + 1]

    if len(embs) < len(tokens):
        embs = np.vstack([embs, np.zeros((len(tokens) - len(embs), embs.shape[1]))])
    elif len(embs) > len(tokens):
        embs = embs[:len(tokens)]

    n = len(tokens)
    pos_enc = np.zeros((n, len(UNIVERSAL_POS_TAGS)))
    for i, pos in enumerate(pos_tags):
        pos_enc[i, pos_tag_map.get(pos, pos_tag_map["X"])] = 1

    position = np.arange(n).reshape(-1, 1) / n
    stop_punct = np.column_stack((np.array(is_stop), np.array(is_punct)))
    node_features = np.hstack([embs, pos_enc, position, stop_punct])

    x, edge_index, edge_attr = build_knn_graph(node_features, k=K_NEIGHBORS)

    sent_bounds = [0]
    for sent in doc.sents:
        if sent.end < max_tokens and sent.end < n:
            sent_bounds.append(sent.end)
    if len(sent_bounds) == 1:
        sent_bounds.append(min(n, max_tokens))

    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        sentence_boundaries=torch.tensor(sent_bounds, dtype=torch.long),
    )
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    return data, tokens


# ========== INFERENCE ==========

def load_model(model_path, input_dim):
    model = EnhancedTextGNN(input_dim, HIDDEN_CHANNELS, NUM_LAYERS)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def predict_text(text, model, tokenizer, language_model, nlp):
    data, tokens = process_text(text, tokenizer, language_model, nlp)
    data = data.to(DEVICE)

    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, data.batch,
                       sentence_boundaries=data.sentence_boundaries)[0]
        probs = F.softmax(logits, dim=1)

    return {
        "prediction": "AI-generated" if logits.argmax(1).item() == 1 else "Human-written",
        "human_prob": probs[0, 0].item(),
        "ai_prob": probs[0, 1].item(),
    }


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(description="GNN inference utility")
    parser.add_argument("--model", default="checkpoints/model2_best.pt", help="Model weights")
    parser.add_argument("--input", help="Text string to classify")
    parser.add_argument("--file", help="Text file to classify")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
    language_model = AutoModel.from_pretrained(CONTEXTUAL_MODEL).to(DEVICE)
    nlp = spacy.load("en_core_web_sm")

    sample_data, _ = process_text("Sample text.", tokenizer, language_model, nlp)
    model = load_model(args.model, sample_data.x.size(1))

    if args.input:
        text = args.input
    elif args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        print("Enter text (type 'quit' to exit):")
        while True:
            text = input("> ")
            if text.lower() == "quit":
                return
            if text.strip():
                r = predict_text(text, model, tokenizer, language_model, nlp)
                print(f"  {r['prediction']}  (human: {r['human_prob']:.4f}, ai: {r['ai_prob']:.4f})\n")
        return

    r = predict_text(text, model, tokenizer, language_model, nlp)
    print(f"Prediction: {r['prediction']}")
    print(f"  Human: {r['human_prob']:.4f}  AI: {r['ai_prob']:.4f}")


if __name__ == "__main__":
    main()
