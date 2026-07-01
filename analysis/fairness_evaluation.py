"""
Fairness Evaluation — Option A: Evaluate pre-trained AEGIS M2 on L2 English data.

Tests whether AEGIS misclassifies human-written L2 English text as AI-generated
at different rates depending on the writer's L1 background.

Input: CSV with columns:
  - text: the essay/writing sample (human-written)
  - l1: the writer's native language (e.g., "Arabic", "Chinese", "German", ...)
  - proficiency: (optional) proficiency level (e.g., "low", "medium", "high")

All texts are assumed to be human-written (generated=0). The script measures
how often the model incorrectly predicts them as AI-generated (false positive rate).

Usage:
    python fairness_evaluation.py --model ../checkpoints/final_model.pt \
                                  --data l2_essays.csv \
                                  --output fairness_results

To also evaluate on the original arXiv data as a control:
    python fairness_evaluation.py --model ../checkpoints/final_model.pt \
                                  --data l2_essays.csv \
                                  --arxiv ../data/rewritten_texts_file_2K_1.csv \
                                  --output fairness_results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import chi2_contingency, fisher_exact, kruskal
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))
from gnn_utils import process_text, load_model, predict_text, DEVICE, CONTEXTUAL_MODEL

from transformers import AutoModel, AutoTokenizer
import spacy


def setup_models():
    print(f"Device: {DEVICE}")
    print(f"Loading {CONTEXTUAL_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
    language_model = AutoModel.from_pretrained(CONTEXTUAL_MODEL).to(DEVICE)
    language_model.eval()

    print("Loading spaCy ...")
    nlp = spacy.load("en_core_web_sm")

    return tokenizer, language_model, nlp


def evaluate_texts(texts, model, tokenizer, language_model, nlp):
    """Run AEGIS M2 inference on a list of texts. Returns predictions and confidence scores."""
    results = []
    for text in tqdm(texts, desc="Running inference"):
        text = str(text).strip()
        if len(text) < 20:
            results.append({"prediction": -1, "human_prob": 0.5, "ai_prob": 0.5, "error": True})
            continue
        try:
            r = predict_text(text, model, tokenizer, language_model, nlp)
            results.append({
                "prediction": 1 if r["prediction"] == "AI-generated" else 0,
                "human_prob": r["human_prob"],
                "ai_prob": r["ai_prob"],
                "error": False,
            })
        except Exception as e:
            results.append({"prediction": -1, "human_prob": 0.5, "ai_prob": 0.5, "error": True})
    return results


def compute_group_metrics(df):
    """Compute FPR and confidence stats per L1 group."""
    groups = df["l1"].unique()
    rows = []

    for g in sorted(groups):
        gdf = df[df["l1"] == g]
        valid = gdf[gdf["prediction"] >= 0]
        n_total = len(gdf)
        n_valid = len(valid)
        n_errors = n_total - n_valid

        if n_valid == 0:
            continue

        n_fp = (valid["prediction"] == 1).sum()
        fpr = n_fp / n_valid
        mean_ai_prob = valid["ai_prob"].mean()
        std_ai_prob = valid["ai_prob"].std()
        median_ai_prob = valid["ai_prob"].median()

        rows.append({
            "L1 Language": g,
            "N Samples": n_valid,
            "N Errors": n_errors,
            "False Positives": int(n_fp),
            "FPR": fpr,
            "Mean AI Prob": mean_ai_prob,
            "Std AI Prob": std_ai_prob,
            "Median AI Prob": median_ai_prob,
        })

    return pd.DataFrame(rows)


def statistical_tests(df):
    """Run statistical tests for fairness across L1 groups."""
    results = {}

    groups = df["l1"].unique()
    valid = df[df["prediction"] >= 0]

    contingency = []
    ai_probs_by_group = []
    for g in sorted(groups):
        gdf = valid[valid["l1"] == g]
        if len(gdf) < 5:
            continue
        n_fp = (gdf["prediction"] == 1).sum()
        n_tn = (gdf["prediction"] == 0).sum()
        contingency.append([int(n_fp), int(n_tn)])
        ai_probs_by_group.append(gdf["ai_prob"].values)

    # Chi-squared test on FPR across groups
    if len(contingency) >= 2:
        ct = np.array(contingency)
        if ct.sum() > 0 and all(ct.sum(axis=0) > 0):
            try:
                chi2, p_chi2, dof, _ = chi2_contingency(ct)
                results["chi2_test"] = {
                    "statistic": chi2, "p_value": p_chi2, "dof": dof,
                    "interpretation": "SIGNIFICANT bias detected" if p_chi2 < 0.05 else "No significant bias detected",
                }
            except Exception:
                results["chi2_test"] = {"error": "Could not compute chi-squared"}

    # Kruskal-Wallis test on AI probability distributions
    if len(ai_probs_by_group) >= 2:
        try:
            h_stat, p_kw = kruskal(*ai_probs_by_group)
            results["kruskal_wallis"] = {
                "statistic": h_stat, "p_value": p_kw,
                "interpretation": "SIGNIFICANT difference in AI confidence across groups" if p_kw < 0.05
                    else "No significant difference in AI confidence across groups",
            }
        except Exception:
            results["kruskal_wallis"] = {"error": "Could not compute Kruskal-Wallis"}

    # Overall FPR
    total_valid = len(valid)
    total_fp = (valid["prediction"] == 1).sum()
    results["overall"] = {
        "total_samples": total_valid,
        "total_false_positives": int(total_fp),
        "overall_fpr": total_fp / total_valid if total_valid > 0 else 0,
    }

    return results


def evaluate_arxiv_control(model, tokenizer, language_model, nlp, arxiv_path):
    """Evaluate on original arXiv data as a control — checks model accuracy on its training domain."""
    print("\n" + "=" * 60)
    print("  CONTROL: Evaluating on arXiv data (training domain)")
    print("=" * 60)

    df = pd.read_csv(arxiv_path)
    df["text"] = df["text"].astype(str)

    n_sample = min(200, len(df))
    human = df[df["generated"] == 0].sample(n=min(n_sample // 2, len(df[df["generated"] == 0])), random_state=42)
    ai = df[df["generated"] == 1].sample(n=min(n_sample // 2, len(df[df["generated"] == 1])), random_state=42)
    sample = pd.concat([human, ai]).reset_index(drop=True)

    preds = evaluate_texts(sample["text"].tolist(), model, tokenizer, language_model, nlp)
    sample["prediction"] = [p["prediction"] for p in preds]
    sample["ai_prob"] = [p["ai_prob"] for p in preds]

    valid = sample[sample["prediction"] >= 0]
    correct = ((valid["prediction"] == valid["generated"])).sum()
    acc = correct / len(valid) if len(valid) > 0 else 0

    human_valid = valid[valid["generated"] == 0]
    ai_valid = valid[valid["generated"] == 1]
    fpr_arxiv = (human_valid["prediction"] == 1).sum() / len(human_valid) if len(human_valid) > 0 else 0
    fnr_arxiv = (ai_valid["prediction"] == 0).sum() / len(ai_valid) if len(ai_valid) > 0 else 0

    print(f"  Accuracy:           {acc:.4f} ({correct}/{len(valid)})")
    print(f"  FPR (human→AI):     {fpr_arxiv:.4f}")
    print(f"  FNR (AI→human):     {fnr_arxiv:.4f}")
    print(f"  Mean AI prob (human): {human_valid['ai_prob'].mean():.4f}")
    print(f"  Mean AI prob (AI):    {ai_valid['ai_prob'].mean():.4f}")

    return {"accuracy": acc, "fpr": fpr_arxiv, "fnr": fnr_arxiv}


def create_plots(group_df, output_dir):
    """Generate visualization plots."""

    # 1. FPR by L1 group (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    groups = group_df.sort_values("FPR", ascending=False)
    colors = ["#dc2626" if fpr > 0.2 else "#f59e0b" if fpr > 0.1 else "#047857" for fpr in groups["FPR"]]
    bars = ax.bar(range(len(groups)), groups["FPR"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups["L1 Language"], rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("False Positive Rate", fontsize=13)
    ax.set_title("AEGIS M2 — False Positive Rate by L1 Language Group", fontsize=15)
    ax.axhline(y=groups["FPR"].mean(), color="#3b82f6", linestyle="--", linewidth=1.5,
               label=f"Overall FPR: {groups['FPR'].mean():.3f}")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    for bar, n in zip(bars, groups["N Samples"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"n={n}", ha="center", va="bottom", fontsize=9, color="#64748b")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fpr_by_l1_group.png"), dpi=300)
    plt.close()
    print(f"  Plot saved: fpr_by_l1_group.png")

    # 2. AI confidence distribution by L1 group (box plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(groups)), groups["Mean AI Prob"], color="#3b82f6", alpha=0.7, label="Mean AI Prob")
    ax.errorbar(range(len(groups)), groups["Mean AI Prob"], yerr=groups["Std AI Prob"],
                fmt="none", color="#1e3a5f", capsize=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups["L1 Language"], rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("AI Probability (model confidence)", fontsize=13)
    ax.set_title("AEGIS M2 — AI Confidence by L1 Language Group", fontsize=15)
    ax.axhline(y=0.5, color="#dc2626", linestyle=":", linewidth=1, label="Decision boundary (0.5)")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ai_confidence_by_l1.png"), dpi=300)
    plt.close()
    print(f"  Plot saved: ai_confidence_by_l1.png")


def print_results(group_df, stats):
    print("\n" + "=" * 80)
    print("  FAIRNESS EVALUATION RESULTS — AEGIS M2 on L2 English Writing")
    print("=" * 80)

    print(f"\n  Overall: {stats['overall']['total_false_positives']} false positives "
          f"out of {stats['overall']['total_samples']} samples "
          f"(FPR = {stats['overall']['overall_fpr']:.4f})")

    print(f"\n  {'L1 Language':<20} {'N':>6} {'FP':>6} {'FPR':>8} {'Mean P(AI)':>12} {'Median P(AI)':>14}")
    print("  " + "-" * 70)
    for _, row in group_df.sort_values("FPR", ascending=False).iterrows():
        flag = " ←" if row["FPR"] > 2 * stats["overall"]["overall_fpr"] else ""
        print(f"  {row['L1 Language']:<20} {row['N Samples']:>6} {row['False Positives']:>6} "
              f"{row['FPR']:>8.4f} {row['Mean AI Prob']:>12.4f} {row['Median AI Prob']:>14.4f}{flag}")

    if "chi2_test" in stats and "error" not in stats["chi2_test"]:
        ct = stats["chi2_test"]
        print(f"\n  Chi-squared test:  χ² = {ct['statistic']:.4f}, p = {ct['p_value']:.6f}, dof = {ct['dof']}")
        print(f"    → {ct['interpretation']}")

    if "kruskal_wallis" in stats and "error" not in stats["kruskal_wallis"]:
        kw = stats["kruskal_wallis"]
        print(f"\n  Kruskal-Wallis:    H = {kw['statistic']:.4f}, p = {kw['p_value']:.6f}")
        print(f"    → {kw['interpretation']}")

    # Disparity ratio
    fprs = group_df[group_df["N Samples"] >= 10]["FPR"]
    if len(fprs) >= 2 and fprs.min() > 0:
        ratio = fprs.max() / fprs.min()
        print(f"\n  Max/Min FPR ratio: {ratio:.2f}x (groups with n≥10)")
        if ratio > 2.0:
            print("    → SIGNIFICANT DISPARITY: highest-FPR group is >2x the lowest")
        else:
            print("    → Acceptable disparity range")
    elif len(fprs) >= 2:
        print(f"\n  FPR range: {fprs.min():.4f} — {fprs.max():.4f}")

    print("\n" + "=" * 80)


def save_latex(group_df, stats, output_dir):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{AEGIS M2 false positive rates by L1 language group}",
        r"\label{tab:fairness}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{L1 Language} & \textbf{N} & \textbf{FP} & \textbf{FPR} & \textbf{Mean $P(\text{AI})$} & \textbf{Median $P(\text{AI})$} \\",
        r"\hline",
    ]
    for _, row in group_df.sort_values("FPR", ascending=False).iterrows():
        lines.append(
            f"{row['L1 Language']} & {row['N Samples']} & {row['False Positives']} & "
            f"{row['FPR']:.4f} & {row['Mean AI Prob']:.4f} & {row['Median AI Prob']:.4f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(
        f"\\textbf{{Overall}} & {stats['overall']['total_samples']} & "
        f"{stats['overall']['total_false_positives']} & "
        f"{stats['overall']['overall_fpr']:.4f} & — & — \\\\"
    )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(output_dir, "fairness_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="AEGIS M2 Fairness Evaluation")
    parser.add_argument("--model", default="../checkpoints/final_model.pt", help="Path to M2 checkpoint")
    parser.add_argument("--data", required=True, help="CSV with 'text' and 'l1' columns (human-written L2 essays)")
    parser.add_argument("--proficiency", action="store_true", help="Also break down by proficiency level (requires 'proficiency' column)")
    parser.add_argument("--arxiv", default=None, help="Path to arXiv dataset CSV for control evaluation")
    parser.add_argument("--output", default="fairness_results", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per L1 group (for faster testing)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    tokenizer, language_model, nlp = setup_models()

    sample_data, _ = process_text("Sample text.", tokenizer, language_model, nlp)
    model = load_model(args.model, sample_data.x.size(1))
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load L2 data
    df = pd.read_csv(args.data)
    assert "text" in df.columns and "l1" in df.columns, "CSV must have 'text' and 'l1' columns"
    df["text"] = df["text"].astype(str)
    df["l1"] = df["l1"].astype(str).str.strip()

    if args.max_samples:
        df = df.groupby("l1").apply(
            lambda g: g.sample(n=min(args.max_samples, len(g)), random_state=42)
        ).reset_index(drop=True)

    print(f"\nL2 dataset: {len(df)} samples across {df['l1'].nunique()} L1 groups")
    print(f"  Groups: {dict(df['l1'].value_counts())}")

    # Run inference
    print("\n" + "=" * 60)
    print("  EVALUATING AEGIS M2 ON L2 HUMAN-WRITTEN TEXTS")
    print("=" * 60)
    preds = evaluate_texts(df["text"].tolist(), model, tokenizer, language_model, nlp)
    df["prediction"] = [p["prediction"] for p in preds]
    df["human_prob"] = [p["human_prob"] for p in preds]
    df["ai_prob"] = [p["ai_prob"] for p in preds]
    df["error"] = [p["error"] for p in preds]

    # Compute metrics
    group_df = compute_group_metrics(df)
    stats = statistical_tests(df)

    # Print and save
    print_results(group_df, stats)
    create_plots(group_df, args.output)
    save_latex(group_df, stats, args.output)

    # Save detailed results
    df.to_csv(os.path.join(args.output, "detailed_predictions.csv"), index=False)
    group_df.to_csv(os.path.join(args.output, "group_metrics.csv"), index=False)
    print(f"  Detailed results saved to {args.output}/")

    # Proficiency breakdown if requested
    if args.proficiency and "proficiency" in df.columns:
        print("\n" + "=" * 60)
        print("  BREAKDOWN BY PROFICIENCY LEVEL")
        print("=" * 60)
        for prof in sorted(df["proficiency"].unique()):
            pdf = df[df["proficiency"] == prof]
            valid = pdf[pdf["prediction"] >= 0]
            fpr = (valid["prediction"] == 1).sum() / len(valid) if len(valid) > 0 else 0
            print(f"  {prof:<15} n={len(valid):>5}  FPR={fpr:.4f}  Mean P(AI)={valid['ai_prob'].mean():.4f}")

    # Control evaluation on arXiv
    if args.arxiv:
        control = evaluate_arxiv_control(model, tokenizer, language_model, nlp, args.arxiv)

        fpr_l2 = stats["overall"]["overall_fpr"]
        fpr_arxiv = control["fpr"]
        print(f"\n  COMPARISON:")
        print(f"    FPR on arXiv (L1 academic):   {fpr_arxiv:.4f}")
        print(f"    FPR on L2 essays:             {fpr_l2:.4f}")
        if fpr_l2 > 0 and fpr_arxiv > 0:
            print(f"    Ratio (L2/arXiv):             {fpr_l2/fpr_arxiv:.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
