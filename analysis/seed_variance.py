"""
Seed variance audit for Ch4 hardening.

Two-phase design:
  Phase 1 — precompute graphs (CPU+GPU heavy, runs once)
    Builds all 4000 PyG Data objects and saves to disk.
  Phase 2 — train from cache (GPU only, fast, runs per seed)
    Loads cached graphs, splits by seed, trains, evaluates.

Usage:
    # Phase 1: build graph cache (~30-60 min, only needed once)
    python analysis/seed_variance.py --data data/rewritten_texts_file_2K_1.csv --phase cache --model m2
    python analysis/seed_variance.py --data data/rewritten_texts_file_2K_1.csv --phase cache --model m3

    # Phase 2: run seed variance from cache (~minutes per seed)
    python analysis/seed_variance.py --data data/rewritten_texts_file_2K_1.csv --phase train --model both --seeds 3

    # Or do both in one go:
    python analysis/seed_variance.py --data data/rewritten_texts_file_2K_1.csv --phase full --model both --seeds 3
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return fpr[-1]
    return fpr[idx]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==================== PHASE 1: GRAPH CACHING ====================

def cache_m2_graphs(df, cache_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from model2_gat import (
        EnhancedTextGraphDataset, standardize_text, CONTEXTUAL_MODEL,
    )
    from transformers import AutoModel, AutoTokenizer
    import spacy

    device = get_device()
    df = df.copy()
    df["text"] = df["text"].apply(standardize_text)

    tokenizer = AutoTokenizer.from_pretrained(CONTEXTUAL_MODEL)
    language_model = AutoModel.from_pretrained(CONTEXTUAL_MODEL).to(device)
    language_model.eval()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    print(f"Building M2 graphs for {len(df)} samples...")
    dataset = EnhancedTextGraphDataset(df, tokenizer, language_model, nlp)

    graphs = []
    for i in tqdm(range(len(dataset)), desc="Caching M2 graphs"):
        data = dataset[i]
        data.x = data.x.cpu()
        data.edge_index = data.edge_index.cpu()
        data.edge_attr = data.edge_attr.cpu()
        data.y = data.y.cpu()
        if hasattr(data, "sentence_boundaries"):
            data.sentence_boundaries = data.sentence_boundaries.cpu()
        graphs.append(data)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(graphs, cache_path)
    print(f"Saved {len(graphs)} M2 graphs to {cache_path}")

    del language_model, dataset
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cache_m3_graphs(df, cache_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from model3_ppl_gat import (
        EnhancedTextGraphDataset, standardize_text,
        compute_text_complexity, compute_readability_class, CONTEXTUAL_MODEL,
    )
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import spacy

    device = get_device()
    df = df.copy()
    df["text"] = df["text"].apply(standardize_text)
    df["complexity"] = df["text"].apply(compute_text_complexity)
    df["readability"] = df["text"].apply(compute_readability_class)

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

    print(f"Building M3 graphs for {len(df)} samples...")
    dataset = EnhancedTextGraphDataset(df, tokenizer, language_model, nlp, vocab_size)

    graphs = []
    for i in tqdm(range(len(dataset)), desc="Caching M3 graphs"):
        data = dataset[i]
        data.x = data.x.cpu()
        data.edge_index = data.edge_index.cpu()
        data.edge_attr = data.edge_attr.cpu()
        data.y = data.y.cpu()
        if hasattr(data, "complexity_score"):
            data.complexity_score = data.complexity_score.cpu()
        if hasattr(data, "readability_label"):
            data.readability_label = data.readability_label.cpu()
        graphs.append(data)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(graphs, cache_path)
    print(f"Saved {len(graphs)} M3 graphs to {cache_path}")

    del language_model, dataset
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ==================== PHASE 2: TRAINING FROM CACHE ====================

def train_from_cache_m2(graphs, labels, seed, epochs=20, batch_size=16):
    from torch_geometric.loader import DataLoader

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from model2_gat import EnhancedTextGNN, HIDDEN_CHANNELS, NUM_LAYERS, LABEL_SMOOTHING, LEARNING_RATE, WEIGHT_DECAY
    from model2_gat import compute_text_complexity, compute_readability_class

    set_all_seeds(seed)
    device = get_device()

    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=seed, stratify=labels,
    )

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=0)

    num_features = graphs[0].x.size(1)
    model = EnhancedTextGNN(num_features, HIDDEN_CHANNELS, NUM_LAYERS).to(device)

    y_train = np.array([labels[i] for i in train_idx])
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, _, _ = model(
                data.x, data.edge_index, data.edge_attr, data.batch,
                sentence_boundaries=getattr(data, "sentence_boundaries", None),
            )
            loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == data.y.view(-1)).sum().item()
            total += data.y.size(0)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out, _, _ = model(
                    data.x, data.edge_index, data.edge_attr, data.batch,
                    sentence_boundaries=getattr(data, "sentence_boundaries", None),
                )
                val_correct += (out.argmax(1) == data.y.view(-1)).sum().item()
                val_total += data.y.size(0)

        val_acc = val_correct / val_total
        print(f"    Epoch {epoch:02d}  Train={correct/total:.4f}  Val={val_acc:.4f}")

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
            out, _, _ = model(
                data.x, data.edge_index, data.edge_attr, data.batch,
                sentence_boundaries=getattr(data, "sentence_boundaries", None),
            )
            probs = torch.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(data.y.view(-1).cpu().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
    auroc = roc_auc_score(all_labels, all_probs)
    fpr5 = fpr_at_tpr(all_labels, all_probs)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {"seed": seed, "accuracy": acc, "auroc": auroc, "fpr_at_95": fpr5}


def train_from_cache_m3(graphs, labels, seed, epochs=30, batch_size=16):
    from torch_geometric.loader import DataLoader

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from model3_ppl_gat import EnhancedTextGNN, HIDDEN_CHANNELS, NUM_LAYERS, LABEL_SMOOTHING, LEARNING_RATE, WEIGHT_DECAY

    set_all_seeds(seed)
    device = get_device()

    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=seed, stratify=labels,
    )

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=0)

    num_features = graphs[0].x.size(1)
    model = EnhancedTextGNN(num_features, HIDDEN_CHANNELS, NUM_LAYERS).to(device)

    y_train = np.array([labels[i] for i in train_idx])
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            w = F.softmax(model.edge_component_weights, dim=0)
            scalar_ea = (data.edge_attr * w.unsqueeze(0)).sum(dim=1, keepdim=True)

            out, _, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == data.y.view(-1)).sum().item()
            total += data.y.size(0)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out, _, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
                val_correct += (out.argmax(1) == data.y.view(-1)).sum().item()
                val_total += data.y.size(0)

        val_acc = val_correct / val_total
        print(f"    Epoch {epoch:02d}  Train={correct/total:.4f}  Val={val_acc:.4f}")

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

    acc = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
    auroc = roc_auc_score(all_labels, all_probs)
    fpr5 = fpr_at_tpr(all_labels, all_probs)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {"seed": seed, "accuracy": acc, "auroc": auroc, "fpr_at_95": fpr5}


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Seed variance audit (cached)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["m2", "m3", "both"], default="both")
    parser.add_argument("--phase", choices=["cache", "train", "full"], default="full")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed_list", type=int, nargs="+", help="Specific seeds to run (overrides --seeds)")
    parser.add_argument("--epochs_m2", type=int, default=20)
    parser.add_argument("--epochs_m3", type=int, default=30)
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--output", default="analysis/results/seed_variance.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[["text", "generated"]].astype({"generated": int})
    labels = df["generated"].values
    print(f"Loaded {len(df)} samples")

    m2_cache = os.path.join(args.cache_dir, "m2_graphs.pt")
    m3_cache = os.path.join(args.cache_dir, "m3_graphs.pt")

    # Phase 1: cache graphs
    if args.phase in ("cache", "full"):
        if args.model in ("m2", "both"):
            if os.path.exists(m2_cache):
                print(f"M2 cache exists at {m2_cache} — skipping (delete to rebuild)")
            else:
                cache_m2_graphs(df, m2_cache)

        if args.model in ("m3", "both"):
            if os.path.exists(m3_cache):
                print(f"M3 cache exists at {m3_cache} — skipping (delete to rebuild)")
            else:
                cache_m3_graphs(df, m3_cache)

    if args.phase == "cache":
        print("\nGraph caching complete. Run with --phase train to start seed variance.")
        return

    # Phase 2: train from cache
    if args.seed_list:
        seed_list = args.seed_list
    else:
        seed_list = [42, 123, 2024][:args.seeds]
        if args.seeds > 3:
            seed_list = list(range(42, 42 + args.seeds))

    results = {}

    if args.model in ("m2", "both"):
        print(f"\n{'='*60}")
        print(f"MODEL 2 (GAT + RoBERTa) — {args.seeds} seeds from cache")
        print(f"{'='*60}")

        print(f"Loading M2 graphs from {m2_cache}...")
        m2_graphs = torch.load(m2_cache, map_location="cpu", weights_only=False)
        print(f"Loaded {len(m2_graphs)} cached graphs")

        m2_results = []
        for seed in seed_list:
            print(f"\n--- Seed {seed} ---")
            r = train_from_cache_m2(m2_graphs, labels, seed, epochs=args.epochs_m2)
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

        del m2_graphs

    if args.model in ("m3", "both"):
        print(f"\n{'='*60}")
        print(f"MODEL 3 (PPL-GAT) — {args.seeds} seeds from cache")
        print(f"{'='*60}")

        print(f"Loading M3 graphs from {m3_cache}...")
        m3_graphs = torch.load(m3_cache, map_location="cpu", weights_only=False)
        print(f"Loaded {len(m3_graphs)} cached graphs")

        m3_results = []
        for seed in seed_list:
            print(f"\n--- Seed {seed} ---")
            r = train_from_cache_m3(m3_graphs, labels, seed, epochs=args.epochs_m3)
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

        del m3_graphs

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
