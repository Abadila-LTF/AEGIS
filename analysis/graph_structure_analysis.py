"""
Graph structural properties analysis.

Compares graph metrics (degree, clustering, modularity, etc.) between
human-written and AI-generated texts. Outputs comparison table, LaTeX,
and box-plots.

Usage:
    python graph_structure_analysis.py --model checkpoints/model2_best.pt \
                                       --data ../data/dataset.csv --samples 50
"""

import argparse
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ttest_ind
from community import community_louvain
from torch_geometric.utils import to_networkx
from transformers import AutoModel, AutoTokenizer
import spacy
from tqdm import tqdm

from gnn_utils import process_text, load_model, DEVICE, CONTEXTUAL_MODEL


class GraphStructureAnalyzer:
    def __init__(self, model_path, csv_path, sample_size=50):
        self.sample_size = sample_size

        print("Loading tokenizer and language model ...")
        self.tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
        self.language_model = AutoModel.from_pretrained(CONTEXTUAL_MODEL).to(DEVICE)
        self.nlp = spacy.load("en_core_web_sm")

        self.df = pd.read_csv(csv_path)

        sample_data, _ = process_text("Sample text.", self.tokenizer, self.language_model, self.nlp)
        self.model = load_model(model_path, sample_data.x.size(1))

    # ---- metrics ----

    def compute_graph_metrics(self):
        human = self.df[self.df["generated"] == 0]["text"].values
        ai = self.df[self.df["generated"] == 1]["text"].values

        if len(human) > self.sample_size:
            human = np.random.choice(human, self.sample_size, replace=False)
        if len(ai) > self.sample_size:
            ai = np.random.choice(ai, self.sample_size, replace=False)

        metrics = {"human": defaultdict(list), "ai": defaultdict(list)}

        for text in tqdm(human, desc="Human texts"):
            m = self._metrics_for_text(text)
            if m:
                for k, v in m.items():
                    metrics["human"][k].append(v)

        for text in tqdm(ai, desc="AI texts"):
            m = self._metrics_for_text(text)
            if m:
                for k, v in m.items():
                    metrics["ai"][k].append(v)

        results = {}
        for source in ("human", "ai"):
            results[source] = {}
            for key, vals in metrics[source].items():
                results[source][key] = {"mean": np.mean(vals), "std": np.std(vals), "values": vals}
        return results

    def _metrics_for_text(self, text):
        try:
            data, tokens = process_text(text, self.tokenizer, self.language_model, self.nlp)
            if len(tokens) < 5:
                return None

            G = to_networkx(data, to_undirected=True, remove_self_loops=True)
            if not nx.is_connected(G) and len(G.nodes) > 0:
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            if len(G.nodes) < 3:
                return None

            avg_degree = np.mean([d for _, d in G.degree()])
            ewv = np.var([G.edges[e].get("weight", 1.0) for e in G.edges])
            avg_cc = nx.average_clustering(G, weight="weight")

            try:
                communities = community_louvain.best_partition(G)
                sizes = defaultdict(int)
                for _, cid in communities.items():
                    sizes[cid] += 1
                modularity = community_louvain.modularity(communities, G)
                csv_var = np.var(list(sizes.values()))
                n_comm = len(sizes)
            except Exception:
                modularity, csv_var, n_comm = 0, 0, 0

            return {
                "avg_degree": avg_degree, "edge_weight_variance": ewv,
                "avg_clustering": avg_cc, "num_communities": n_comm,
                "modularity": modularity, "community_size_variance": csv_var,
                "node_count": len(G.nodes), "edge_count": len(G.edges),
            }
        except Exception as e:
            print(f"  Skipped: {e}")
            return None

    # ---- output ----

    def generate_comparison_table(self):
        results = self.compute_graph_metrics()

        display_metrics = [
            ("avg_degree", "Average Node Degree"),
            ("edge_weight_variance", "Edge Weight Variance"),
            ("avg_clustering", "Clustering Coefficient"),
            ("modularity", "Community Modularity"),
            ("community_size_variance", "Community Size Variance"),
            ("node_count", "Average Node Count"),
            ("edge_count", "Average Edge Count"),
        ]

        pvalues = {}
        for key, _ in display_metrics:
            if key in results["human"] and key in results["ai"]:
                hv = results["human"][key]["values"]
                av = results["ai"][key]["values"]
                if hv and av:
                    _, pv = ttest_ind(hv, av, equal_var=False)
                    pvalues[key] = pv

        rows = []
        for key, name in display_metrics:
            if key not in results["human"] or key not in results["ai"]:
                continue
            hm, hs = results["human"][key]["mean"], results["human"][key]["std"]
            am, as_ = results["ai"][key]["mean"], results["ai"][key]["std"]
            sig = ""
            if key in pvalues:
                p = pvalues[key]
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            rows.append({
                "Metric": name,
                "Human Mean": f"{hm:.4f}", "Human Std": f"{hs:.4f}",
                "AI Mean": f"{am:.4f}", "AI Std": f"{as_:.4f}",
                "Difference": f"{hm - am:.4f}{sig}",
            })

        table = pd.DataFrame(rows)
        print("\n" + table.to_string(index=False))

        table.to_csv("graph_structure_comparison.csv", index=False)
        self._save_latex(table)
        self._create_plots(results)
        return table

    def _save_latex(self, df):
        lines = [
            r"\begin{table}[ht]", r"\centering",
            r"\caption{Graph structural properties: Human-written vs.\ AI-generated texts}",
            r"\label{tab:graph_stats}",
            r"\begin{tabular}{lcccc}", r"\hline",
            r"\textbf{Property} & \textbf{Human} & \textbf{AI} & \textbf{Diff.} \\",
            r"\hline",
        ]
        for _, row in df.iterrows():
            h = f"{row['Human Mean']} $\\pm$ {row['Human Std']}"
            a = f"{row['AI Mean']} $\\pm$ {row['AI Std']}"
            lines.append(f"{row['Metric']} & {h} & {a} & {row['Difference']} \\\\")
        lines += [
            r"\hline",
            r"\multicolumn{4}{l}{\footnotesize * $p<0.05$, ** $p<0.01$, *** $p<0.001$} \\",
            r"\end{tabular}", r"\end{table}",
        ]
        with open("graph_structure_comparison.tex", "w") as f:
            f.write("\n".join(lines))
        print("LaTeX saved to graph_structure_comparison.tex")

    def _create_plots(self, results):
        keys = [
            ("avg_degree", "Average Node Degree"),
            ("edge_weight_variance", "Edge Weight Variance"),
            ("avg_clustering", "Clustering Coefficient"),
            ("modularity", "Community Modularity"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        for ax, (key, title) in zip(axes.flatten(), keys):
            if key in results["human"] and key in results["ai"]:
                hv = results["human"][key]["values"]
                av = results["ai"][key]["values"]
                bp = ax.boxplot([hv, av], labels=["Human", "AI"], patch_artist=True)
                bp["boxes"][0].set(facecolor="lightblue")
                bp["boxes"][1].set(facecolor="lightcoral")
                ax.set_title(title, fontsize=14)
                ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("graph_metrics_comparison.png", dpi=300)
        print("Plots saved to graph_metrics_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Graph structure analysis")
    parser.add_argument("--model", default="checkpoints/model2_best.pt")
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    analyzer = GraphStructureAnalyzer(args.model, args.data, args.samples)
    analyzer.generate_comparison_table()


if __name__ == "__main__":
    main()
