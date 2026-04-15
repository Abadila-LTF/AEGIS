"""
AEGIS Model 1 — Simple GCN Baseline.

Proof-of-concept demonstrating that graph-based text representation
contains discriminative signals for AI-generated text detection.

Architecture: GloVe-100d embeddings → cosine-similarity graph → 2× GCNConv → global mean pool → linear
"""

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def standardize_text(text):
    paragraphs = text.strip().split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return "\n".join(paragraphs)


# ========== DATASET ==========

class TextGraphDataset(Dataset):
    def __init__(self, dataframe, word_vectors):
        self.dataframe = dataframe.reset_index(drop=True)
        self.word_vectors = word_vectors
        self.embedding_dim = word_vectors.vector_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row['text']).lower()
        tokens = word_tokenize(text)
        embeddings = self.get_sentence_embeddings(tokens)
        x, edge_index, edge_attr = self.build_graph(embeddings)
        y = torch.tensor([row['generated']], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

    def get_token_embedding(self, token):
        try:
            return self.word_vectors[token]
        except KeyError:
            return np.random.normal(scale=0.6, size=(self.embedding_dim,))

    def get_sentence_embeddings(self, tokens):
        embeddings = [self.get_token_embedding(token) for token in tokens]
        return embeddings

    def build_graph(self, embeddings):
        num_nodes = len(embeddings)
        x = torch.tensor(embeddings, dtype=torch.float)
        if num_nodes == 1:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float)
        else:
            x_normalized = F.normalize(x, p=2, dim=1)
            sim_matrix = torch.mm(x_normalized, x_normalized.t())
            sim_matrix.fill_diagonal_(0)
            indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
            sim_scores = sim_matrix[indices[0], indices[1]]
            threshold = 0.5
            mask = sim_scores > threshold
            edge_index = torch.stack([indices[0][mask], indices[1][mask]], dim=0)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr = sim_scores[mask]
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        return x, edge_index, edge_attr


# ========== MODEL ==========

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = Dropout(p=0.5)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ========== TRAINING ==========

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data.x, data.edge_index, data.batch), data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        correct += int((pred == data.y).sum())
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(data.y.cpu().tolist())
    return correct / len(loader.dataset), all_preds, all_labels


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description="AEGIS Model 1 — Simple GCN")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print("Loading GloVe embeddings (glove-wiki-gigaword-100) ...")
    word_vectors = api.load("glove-wiki-gigaword-100")

    df = pd.read_csv(args.data)
    df["text"] = df["text"].apply(standardize_text)
    df = df[["text", "generated"]].astype({"generated": int})
    print(f"Loaded {len(df)} samples")

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["generated"]
    )

    train_loader = DataLoader(
        TextGraphDataset(train_df, word_vectors),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        TextGraphDataset(test_df, word_vectors),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = GCN(num_node_features=word_vectors.vector_size, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        train_acc, _, _ = evaluate(model, train_loader, device)
        test_acc, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d}  Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f}")

    _, all_preds, all_labels = evaluate(model, test_loader, device)
    print("\n" + classification_report(all_labels, all_preds, digits=4))


if __name__ == "__main__":
    main()

