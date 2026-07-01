"""
Metric upgrade for Ch4 hardening.

Loads an existing M2 checkpoint and re-evaluates on the test split with:
  - AUROC
  - FPR@5% FPR (i.e., threshold at which FPR ≤ 5%)
  - FPR@95% TPR
  - Calibration (ECE + reliability diagram data)
  - Full classification report

Also produces the comparison table with surface baseline and RoBERTa.
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    classification_report, brier_score_loss,
)
from sklearn.calibration import calibration_curve


def fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return fpr[-1], thresholds[-1] if len(thresholds) > 0 else 0.5
    return fpr[idx], thresholds[min(idx, len(thresholds) - 1)]


def tpr_at_fpr(y_true, y_score, fpr_target=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, fpr_target)
    if idx >= len(tpr):
        return tpr[-1], thresholds[-1] if len(thresholds) > 0 else 0.5
    return tpr[idx], thresholds[min(idx, len(thresholds) - 1)]


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def main():
    parser = argparse.ArgumentParser(description="Metric upgrade — re-evaluate M2 checkpoint")
    parser.add_argument("--data", required=True, help="Dataset CSV")
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pt")
    parser.add_argument("--output", default="analysis/results/metric_upgrade.json")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from model2_gat import (
        EnhancedTextGraphDataset, EnhancedTextGNN, standardize_text,
        HIDDEN_CHANNELS, NUM_LAYERS, CONTEXTUAL_MODEL,
    )
    from transformers import AutoModel, AutoTokenizer
    import spacy
    from torch_geometric.loader import DataLoader

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.data)
    df["text"] = df["text"].apply(standardize_text)
    df = df[["text", "generated"]].astype({"generated": int})

    # Use the same split as original training (seed=42, test=0.3)
    _, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["generated"])
    print(f"Test set: {len(test_df)} samples")

    tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
    language_model = AutoModel.from_pretrained(CONTEXTUAL_MODEL).to(device)
    language_model.eval()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    print("Building test graphs...")
    test_dataset = EnhancedTextGraphDataset(test_df, tokenizer, language_model, nlp)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EnhancedTextGNN(790, HIDDEN_CHANNELS, NUM_LAYERS).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Checkpoint loaded.")

    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out, _, _ = model(
                data.x, data.edge_index, data.edge_attr, data.batch,
                sentence_boundaries=getattr(data, "sentence_boundaries", None),
            )
            probs = torch.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(data.y.view(-1).cpu().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    fpr95, thresh95 = fpr_at_tpr(y_true, y_prob, 0.95)
    tpr5, thresh5 = tpr_at_fpr(y_true, y_prob, 0.05)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    print(f"\n{'='*60}")
    print(f"AEGIS M2 — UPGRADED METRICS (checkpoint: {args.checkpoint})")
    print(f"{'='*60}")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  AUROC:            {auroc:.4f}")
    print(f"  FPR@95%TPR:       {fpr95:.4f}  (threshold={thresh95:.4f})")
    print(f"  TPR@5%FPR:        {tpr5:.4f}  (threshold={thresh5:.4f})")
    print(f"  Brier score:      {brier:.4f}")
    print(f"  ECE (10 bins):    {ece:.4f}")
    print(f"\n  Calibration curve (fraction_of_positives vs mean_predicted):")
    for ft, fp in zip(prob_true, prob_pred):
        bar = "#" * int(ft * 40)
        print(f"    pred={fp:.2f}  actual={ft:.2f}  {bar}")

    print(f"\n{classification_report(y_true, y_pred, digits=4, target_names=['Human', 'AI'])}")

    print(f"\n{'='*60}")
    print(f"THESIS COMPARISON TABLE (Ch4)")
    print(f"{'='*60}")
    print(f"  {'Method':<30s}  {'Acc':>7s}  {'AUROC':>7s}  {'FPR@95':>8s}  {'TPR@5':>7s}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}")
    print(f"  {'Surface LR (5 feat)':<30s}  {'0.8097':>7s}  {'0.8807':>7s}  {'0.7130':>8s}  {'  —':>7s}")
    print(f"  {'RoBERTa-FT (125M)':<30s}  {'0.8250':>7s}  {'  —':>7s}  {'   —':>8s}  {'  —':>7s}")
    print(f"  {'AEGIS M2 (73K)':<30s}  {acc:7.4f}  {auroc:7.4f}  {fpr95:8.4f}  {tpr5:7.4f}")

    results = {
        "checkpoint": args.checkpoint,
        "test_size": len(test_df),
        "accuracy": float(acc),
        "auroc": float(auroc),
        "fpr_at_95tpr": float(fpr95),
        "tpr_at_5fpr": float(tpr5),
        "brier_score": float(brier),
        "ece": float(ece),
        "calibration_curve": {
            "fraction_of_positives": prob_true.tolist(),
            "mean_predicted_value": prob_pred.tolist(),
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
