"""
Seed variance audit for Ch4 hardening.

Retrains M2 and M3 each with N seeds, reports mean±std for:
  - Accuracy
  - AUROC
  - FPR@95%TPR

Determines whether M2>M3 ranking is robust or seed noise.

Usage:
    python analysis/seed_variance.py --data data/rewritten_texts_file_2K_1.csv --model m2 --seeds 3
    python analysis/seed_variance.py --data data/rewritten_texts_file_2K_1.csv --model m3 --seeds 3
    python analysis/seed_variance.py --data data/rewritten_texts_file_2K_1.csv --model both --seeds 3
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from datetime import datetime


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return fpr[-1]
    return fpr[idx]


def train_and_evaluate_m2(df, seed, epochs=20, batch_size=16, device=None):
    set_all_seeds(seed)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from model2_gat import (
        EnhancedTextGraphDataset, EnhancedTextGNN, train_epoch, evaluate,
        standardize_text, HIDDEN_CHANNELS, NUM_LAYERS, LABEL_SMOOTHING,
        LEARNING_RATE, WEIGHT_DECAY, CONTEXTUAL_MODEL,
    )
    from transformers import AutoModel, AutoTokenizer
    import spacy
    from torch_geometric.loader import DataLoader
    from sklearn.utils.class_weight import compute_class_weight
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df = df.copy()
    df["text"] = df["text"].apply(standardize_text)

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df["generated"]
    )

    tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
    language_model = AutoModel.from_pretrained(CONTEXTUAL_MODEL).to(device)
    language_model.eval()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    print(f"  [seed={seed}] Building graphs for train ({len(train_df)}) and test ({len(test_df)})...")
    train_dataset = EnhancedTextGraphDataset(train_df, tokenizer, language_model, nlp)
    test_dataset = EnhancedTextGraphDataset(test_df, tokenizer, language_model, nlp)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample = next(iter(train_loader))
    num_features = sample.x.size(1)

    model = EnhancedTextGNN(num_features, HIDDEN_CHANNELS, NUM_LAYERS).to(device)

    y_train = train_df["generated"].values
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTHING)
    crit_complex = torch.nn.MSELoss()
    crit_read = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            (criterion, crit_complex, crit_read), device,
        )
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion, device)
        print(f"    Epoch {epoch:02d}  Train={train_acc:.4f}  Val={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final eval with best checkpoint
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

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

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    auroc = roc_auc_score(all_labels, all_probs)
    fpr5 = fpr_at_tpr(all_labels, all_probs)

    del model, language_model, train_dataset, test_dataset
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    return {"seed": seed, "accuracy": acc, "auroc": auroc, "fpr_at_95": fpr5}


def train_and_evaluate_m3(df, seed, epochs=30, batch_size=16, device=None):
    set_all_seeds(seed)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from model3_ppl_gat import (
        EnhancedTextGraphDataset, EnhancedTextGNN, train_epoch, evaluate,
        standardize_text, compute_text_complexity, compute_readability_class,
        HIDDEN_CHANNELS, NUM_LAYERS, LABEL_SMOOTHING,
        LEARNING_RATE, WEIGHT_DECAY, CONTEXTUAL_MODEL,
    )
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import spacy
    from torch_geometric.loader import DataLoader
    from sklearn.utils.class_weight import compute_class_weight
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df = df.copy()
    df["text"] = df["text"].apply(standardize_text)
    df["complexity"] = df["text"].apply(compute_text_complexity)
    df["readability"] = df["text"].apply(compute_readability_class)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["generated"]
    )

    tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
    language_model = AutoModelForMaskedLM.from_pretrained(CONTEXTUAL_MODEL).to(device)
    language_model.eval()
    vocab_size = tokenizer.vocab_size

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    print(f"  [seed={seed}] Building graphs for train ({len(train_df)}) and test ({len(test_df)})...")
    train_dataset = EnhancedTextGraphDataset(train_df, tokenizer, language_model, nlp, vocab_size)
    test_dataset = EnhancedTextGraphDataset(test_df, tokenizer, language_model, nlp, vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample = train_dataset[0]
    num_features = sample.x.size(1)

    model = EnhancedTextGNN(num_features, HIDDEN_CHANNELS, NUM_LAYERS).to(device)

    y_train = train_df["generated"].values
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion_main = torch.nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTHING)
    crit_complex = torch.nn.MSELoss()
    crit_read = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            (criterion_main, crit_complex, crit_read), device,
        )
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion_main, device)
        print(f"    Epoch {epoch:02d}  Train={train_acc:.4f}  Val={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out, _, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(data.y.view(-1).cpu().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds_binary = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, preds_binary)
    auroc = roc_auc_score(all_labels, all_probs)
    fpr5 = fpr_at_tpr(all_labels, all_probs)

    del model, language_model, train_dataset, test_dataset
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    return {"seed": seed, "accuracy": acc, "auroc": auroc, "fpr_at_95": fpr5}


def main():
    parser = argparse.ArgumentParser(description="Seed variance audit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["m2", "m3", "both"], default="both")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs_m2", type=int, default=20)
    parser.add_argument("--epochs_m3", type=int, default=30)
    parser.add_argument("--output", default="analysis/results/seed_variance.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[["text", "generated"]].astype({"generated": int})
    print(f"Loaded {len(df)} samples")

    seed_list = [42, 123, 2024][:args.seeds]
    if args.seeds > 3:
        seed_list = list(range(42, 42 + args.seeds))

    results = {}

    if args.model in ("m2", "both"):
        print(f"\n{'='*60}")
        print(f"MODEL 2 (GAT + RoBERTa) — {args.seeds} seeds")
        print(f"{'='*60}")
        m2_results = []
        for seed in seed_list:
            print(f"\n--- Seed {seed} ---")
            r = train_and_evaluate_m2(df, seed, epochs=args.epochs_m2)
            m2_results.append(r)
            print(f"  Result: Acc={r['accuracy']:.4f}  AUROC={r['auroc']:.4f}  FPR@95%={r['fpr_at_95']:.4f}")

        accs = [r["accuracy"] for r in m2_results]
        aurocs = [r["auroc"] for r in m2_results]
        fprs = [r["fpr_at_95"] for r in m2_results]
        print(f"\nM2 Summary ({args.seeds} seeds):")
        print(f"  Accuracy:   {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  AUROC:      {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        print(f"  FPR@95%TPR: {np.mean(fprs):.4f} ± {np.std(fprs):.4f}")
        results["m2"] = {"runs": m2_results, "mean_acc": float(np.mean(accs)),
                         "std_acc": float(np.std(accs)), "mean_auroc": float(np.mean(aurocs)),
                         "std_auroc": float(np.std(aurocs))}

    if args.model in ("m3", "both"):
        print(f"\n{'='*60}")
        print(f"MODEL 3 (PPL-GAT) — {args.seeds} seeds")
        print(f"{'='*60}")
        m3_results = []
        for seed in seed_list:
            print(f"\n--- Seed {seed} ---")
            r = train_and_evaluate_m3(df, seed, epochs=args.epochs_m3)
            m3_results.append(r)
            print(f"  Result: Acc={r['accuracy']:.4f}  AUROC={r['auroc']:.4f}  FPR@95%={r['fpr_at_95']:.4f}")

        accs = [r["accuracy"] for r in m3_results]
        aurocs = [r["auroc"] for r in m3_results]
        fprs = [r["fpr_at_95"] for r in m3_results]
        print(f"\nM3 Summary ({args.seeds} seeds):")
        print(f"  Accuracy:   {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  AUROC:      {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        print(f"  FPR@95%TPR: {np.mean(fprs):.4f} ± {np.std(fprs):.4f}")
        results["m3"] = {"runs": m3_results, "mean_acc": float(np.mean(accs)),
                         "std_acc": float(np.std(accs)), "mean_auroc": float(np.mean(aurocs)),
                         "std_auroc": float(np.std(aurocs))}

    if "m2" in results and "m3" in results:
        print(f"\n{'='*60}")
        print(f"M2 vs M3 RANKING")
        print(f"{'='*60}")
        m2_acc = results["m2"]["mean_acc"]
        m3_acc = results["m3"]["mean_acc"]
        gap = m2_acc - m3_acc
        print(f"  M2: {m2_acc:.4f} ± {results['m2']['std_acc']:.4f}")
        print(f"  M3: {m3_acc:.4f} ± {results['m3']['std_acc']:.4f}")
        print(f"  Gap: {gap:+.4f}")
        overlap = (results["m2"]["mean_acc"] - results["m2"]["std_acc"]) < \
                  (results["m3"]["mean_acc"] + results["m3"]["std_acc"])
        print(f"  Confidence intervals overlap: {overlap}")
        if overlap:
            print("  → M2>M3 ranking may be seed noise — report both with error bars")
        else:
            print("  → M2>M3 ranking is robust across seeds")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
