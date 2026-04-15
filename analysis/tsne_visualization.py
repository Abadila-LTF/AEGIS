"""
t-SNE visualization of GNN embedding space.

Extracts document-level embeddings from Model 2 and projects them
to 2D with t-SNE, coloured by class (human vs AI).

Usage:
    python tsne_visualization.py --model checkpoints/model2_best.pt \
                                 --data ../data/dataset.csv --samples 100
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import global_mean_pool, global_add_pool
from transformers import AutoModel, AutoTokenizer
import spacy
from tqdm import tqdm

from gnn_utils import process_text, load_model, DEVICE, CONTEXTUAL_MODEL


class EmbeddingExtractor:
    def __init__(self, model_path, csv_path, sample_size=100):
        self.sample_size = sample_size

        print("Loading tokenizer and language model ...")
        self.tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
        self.language_model = AutoModel.from_pretrained(CONTEXTUAL_MODEL).to(DEVICE)
        self.nlp = spacy.load("en_core_web_sm")

        self.df = pd.read_csv(csv_path)

        sample_data, _ = process_text("Sample text.", self.tokenizer, self.language_model, self.nlp)
        self.model = load_model(model_path, sample_data.x.size(1))

    def extract_embeddings(self):
        human = self.df[self.df["generated"] == 0]["text"].values
        ai = self.df[self.df["generated"] == 1]["text"].values

        if len(human) > self.sample_size:
            human = np.random.choice(human, self.sample_size, replace=False)
        if len(ai) > self.sample_size:
            ai = np.random.choice(ai, self.sample_size, replace=False)

        embeddings, labels = [], []
        for text in tqdm(human, desc="Human texts"):
            emb = self._embed(text)
            if emb is not None:
                embeddings.append(emb)
                labels.append(0)
        for text in tqdm(ai, desc="AI texts"):
            emb = self._embed(text)
            if emb is not None:
                embeddings.append(emb)
                labels.append(1)

        return np.array(embeddings), np.array(labels)

    def _embed(self, text):
        try:
            data, _ = process_text(text, self.tokenizer, self.language_model, self.nlp)
            data = data.to(DEVICE)
            with torch.no_grad():
                x = self.model.input_proj(data.x)
                for i in range(self.model.num_layers):
                    x = self.model.gat_layers[i](x, data.edge_index, edge_attr=data.edge_attr)
                    x = F.elu(self.model.layer_norms[i](x))
                x_pool = torch.cat([global_mean_pool(x, data.batch),
                                    global_add_pool(x, data.batch)], dim=1)
                return F.relu(self.model.lin1(x_pool)).cpu().numpy()[0]
        except Exception as e:
            print(f"  Skipped: {e}")
            return None

    def create_tsne_plot(self, output="tsne_visualization.png"):
        embeddings, labels = self.extract_embeddings()
        n = len(embeddings)
        if n < 5:
            print("Too few samples for t-SNE.")
            return

        perplexity = min(5, n - 1) if n <= 15 else min(15, n // 3)
        print(f"Running t-SNE on {n} embeddings (perplexity={perplexity}) ...")

        coords = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42).fit_transform(embeddings)

        plt.figure(figsize=(12, 10))
        human_pts = coords[labels == 0]
        ai_pts = coords[labels == 1]
        plt.scatter(human_pts[:, 0], human_pts[:, 1], c="blue", marker="o",
                    label="Human-written", alpha=0.7, s=100)
        plt.scatter(ai_pts[:, 0], ai_pts[:, 1], c="red", marker="x",
                    label="AI-generated", alpha=0.7, s=100)
        plt.title("t-SNE Visualization of GNN Embeddings", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output, dpi=300)
        print(f"Saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="t-SNE embedding visualization")
    parser.add_argument("--model", default="checkpoints/model2_best.pt")
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    extractor = EmbeddingExtractor(args.model, args.data, args.samples)
    extractor.create_tsne_plot()


if __name__ == "__main__":
    main()
