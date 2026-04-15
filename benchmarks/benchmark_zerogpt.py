"""
Benchmark ZeroGPT commercial API on a labelled dataset.

Requires a valid ZeroGPT API key (pass via --api_key or ZEROGPT_API_KEY env var).

Usage:
    python benchmark_zerogpt.py --dataset data/dataset.csv --api_key YOUR_KEY
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
)

# ========== CONFIGURATION ==========
ZEROGPT_API_URL = "https://api.zerogpt.com/api/detect/detectText"
RATE_LIMIT_DELAY = 0.1
AI_THRESHOLD = 50.0


# ========== API ==========

def detect_text_zerogpt(text, api_key):
    headers = {"ApiKey": api_key, "Content-Type": "application/json"}
    payload = {"input_text": text}

    try:
        response = requests.post(ZEROGPT_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                data = result.get("data", {})
                return {
                    "success": True,
                    "fake_percentage": data.get("fakePercentage", 0),
                    "ai_words": data.get("aiWords", 0),
                    "text_words": data.get("textWords", 0),
                    "feedback": data.get("feedback", ""),
                    "error": None,
                }
        return {
            "success": False, "fake_percentage": 0, "ai_words": 0,
            "text_words": 0, "feedback": "",
            "error": f"HTTP {response.status_code}: {response.text[:200]}",
        }
    except Exception as e:
        return {
            "success": False, "fake_percentage": 0, "ai_words": 0,
            "text_words": 0, "feedback": "", "error": str(e),
        }


# ========== PLOTTING ==========

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix — ZeroGPT")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(y_true, y_probs, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — ZeroGPT")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall(y_true, y_probs, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — ZeroGPT")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ========== BENCHMARK ==========

def run_benchmark(csv_path, output_dir, api_key, max_samples=None,
                  threshold=AI_THRESHOLD, delay=RATE_LIMIT_DELAY):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("CSV must have 'text' and 'generated' columns")
    df = df[["text", "generated"]].dropna()

    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"Dataset: {len(df)} samples | Threshold: {threshold}%")

    results = []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text, true_label = str(row["text"]), int(row["generated"])
        res = detect_text_zerogpt(text, api_key)

        if res["success"]:
            predicted = 1 if res["fake_percentage"] >= threshold else 0
            results.append({
                "true_label": true_label, "predicted_label": predicted,
                "fake_percentage": res["fake_percentage"],
            })
        else:
            errors += 1
            results.append({"true_label": true_label, "predicted_label": -1, "fake_percentage": 0})
        time.sleep(delay)

    results_df = pd.DataFrame(results)
    valid = results_df[results_df["predicted_label"] >= 0]

    if valid.empty:
        print("All API calls failed.")
        return None

    y_true = valid["true_label"].values
    y_pred = valid["predicted_label"].values
    y_probs = valid["fake_percentage"].values / 100.0

    report = classification_report(y_true, y_pred, target_names=["Human", "AI"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy: {report['accuracy']:.4f}")
    print(f"F1:       {report['weighted avg']['f1-score']:.4f}")
    print(f"Errors:   {errors}/{len(df)}")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": report["accuracy"],
                    "precision": report["weighted avg"]["precision"],
                    "recall": report["weighted avg"]["recall"],
                    "f1": report["weighted avg"]["f1-score"],
                    "threshold": threshold,
                    "samples": len(df), "errors": errors,
                    "runtime_s": time.time() - start_time}, f, indent=2)

    plot_confusion_matrix(cm, ["Human", "AI"], os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc_curve(y_true, y_probs, os.path.join(output_dir, "roc_curve.png"))
    plot_precision_recall(y_true, y_probs, os.path.join(output_dir, "precision_recall.png"))
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)

    print(f"Results saved to {output_dir}/")
    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark ZeroGPT API")
    parser.add_argument("--dataset", required=True, help="Path to labelled CSV")
    parser.add_argument("--output", default="benchmark_zerogpt_output", help="Output directory")
    parser.add_argument("--api_key", default=os.environ.get("ZEROGPT_API_KEY"),
                        help="ZeroGPT API key (or set ZEROGPT_API_KEY env var)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=AI_THRESHOLD)
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_DELAY)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Provide --api_key or set ZEROGPT_API_KEY environment variable")

    run_benchmark(args.dataset, args.output, args.api_key,
                  args.max_samples, args.threshold, args.delay)


if __name__ == "__main__":
    main()
