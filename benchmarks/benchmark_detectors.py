"""
AI-Text Detection Benchmark
============================
Runs 4 established detectors on a labelled dataset and reports
Accuracy, Precision, Recall, F1.

Detectors:
  1. GLTR           — Gehrmann et al. 2019, statistical token-rank analysis
  2. DetectGPT      — Mitchell et al. 2023, perturbation-based zero-shot
  3. Fast-DetectGPT — Bao et al. ICLR 2024, improved zero-shot
  4. RoBERTa        — Fine-tuned supervised baseline

Usage:
    python benchmark_detectors.py --data ../data/dataset.csv
"""

import argparse
import os
import sys
import time
import warnings
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    RobertaTokenizer, RobertaForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


def get_device():
    if torch.backends.mps.is_available():
        print("[Device] Apple MPS (Metal GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"[Device] CUDA - {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("[Device] CPU")
        return torch.device("cpu")

DEVICE = get_device()

CONFIG = {
    "seed":              42,
    "test_size":         0.30,
    "max_length":        256,
    "batch_size":        16,
    "n_perturbations":   20,
    "span_length":       2,
    "pct_masked":        0.15,
    "n_samples":         20,
    "epochs":            10,
    "lr":                2e-5,
    "weight_decay":      1e-2,
    "warmup_ratio":      0.1,
    "early_stopping":    3,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(args):
    if args.data:
        df = pd.read_csv(args.data)
        label_col = "label" if "label" in df.columns else "generated"
        texts = df["text"].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
    elif args.human_file and args.ai_file:
        with open(args.human_file) as f:
            human = [l.strip() for l in f if l.strip()]
        with open(args.ai_file) as f:
            ai = [l.strip() for l in f if l.strip()]
        texts = human + ai
        labels = [0] * len(human) + [1] * len(ai)
    else:
        raise ValueError("Provide --data CSV or --human_file + --ai_file")

    print(f"[Data] Total: {len(texts)} | Human: {labels.count(0)} | AI: {labels.count(1)}")
    return texts, labels


def compute_metrics(labels, preds):
    return {
        "accuracy":  round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
        "support":   len(labels),
    }


# ── GPT-2 (shared by GLTR, DetectGPT, Fast-DetectGPT) ──

def load_gpt2():
    print("\n[GPT-2] Loading gpt2 (~548MB, downloads once) ...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    print("[GPT-2] Loaded")
    return tok, model


def get_log_likelihood(text, tok, model):
    inputs = tok(
        text, return_tensors="pt",
        truncation=True, max_length=CONFIG["max_length"],
    ).to(DEVICE)
    if inputs["input_ids"].shape[1] < 3:
        return -999.0
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
        logits = out.logits[0, :-1, :]
        labels = inputs["input_ids"][0, 1:]
        lps = F.log_softmax(logits, dim=-1)[range(len(labels)), labels]
    return float(lps.mean().cpu())


# ── 1. GLTR (Gehrmann et al. 2019) ──

def gltr_features(text, tok, model):
    inputs = tok(
        text, return_tensors="pt",
        truncation=True, max_length=CONFIG["max_length"],
    ).to(DEVICE)
    if inputs["input_ids"].shape[1] < 3:
        return [0.25, 0.25, 0.25, 0.25]

    with torch.no_grad():
        logits = model(**inputs).logits[0, :-1, :]

    shift_labels = inputs["input_ids"][0, 1:]
    sorted_ids = torch.argsort(logits, dim=-1, descending=True)

    ranks = []
    for i, tok_id in enumerate(shift_labels):
        rank = (sorted_ids[i] == tok_id).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)

    ranks = np.array(ranks)
    n = len(ranks) or 1
    return [
        (ranks < 10).sum() / n,
        (ranks < 100).sum() / n,
        (ranks < 1000).sum() / n,
        (ranks >= 1000).sum() / n,
    ]


def run_gltr(X_test, y_test, X_train, y_train, tok, model):
    print("\n" + "=" * 55)
    print("[1/4] GLTR (Gehrmann et al. 2019 - Harvard/MIT)")
    t0 = time.time()

    def extract(texts, tag):
        feats = []
        for i, t in enumerate(texts):
            if i % 50 == 0:
                print(f"  [{tag}] {i}/{len(texts)}")
            feats.append(gltr_features(t, tok, model))
        return np.array(feats)

    X_tr = extract(X_train, "Train")
    X_te = extract(X_test, "Test ")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=CONFIG["seed"])
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    metrics = compute_metrics(y_test, preds)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s | Accuracy: {metrics['accuracy']} | F1: {metrics['f1']}")
    return metrics


# ── 2. DetectGPT (Mitchell et al. 2023) ──

VOCAB_FILL = ["study", "model", "data", "method", "result", "system",
              "approach", "analysis", "performance", "paper", "show",
              "propose", "using", "based", "new", "high", "large",
              "training", "learning", "network", "feature", "task"]


def perturb_text(text):
    tokens = text.split()
    if len(tokens) < 4:
        return text
    n_mask = max(1, int(len(tokens) * CONFIG["pct_masked"]))
    pos = random.sample(range(len(tokens)), min(n_mask, len(tokens)))
    tokens2 = tokens[:]
    for p in pos:
        tokens2[p] = random.choice(VOCAB_FILL)
    return " ".join(tokens2)


def detectgpt_score(text, tok, model):
    orig_ll = get_log_likelihood(text, tok, model)
    pert_ll = [get_log_likelihood(perturb_text(text), tok, model)
               for _ in range(CONFIG["n_perturbations"])]
    mu = np.mean(pert_ll)
    std = np.std(pert_ll) + 1e-8
    return (orig_ll - mu) / std


def run_detectgpt(X_test, y_test, tok, model):
    print("\n" + "=" * 55)
    print(f"[2/4] DetectGPT (Mitchell et al. 2023) - {CONFIG['n_perturbations']} perturbations")
    t0 = time.time()

    scores = []
    for i, text in enumerate(X_test):
        if i % 50 == 0:
            print(f"  {i}/{len(X_test)}")
        scores.append(detectgpt_score(text, tok, model))

    preds = [1 if s > 0 else 0 for s in scores]
    metrics = compute_metrics(y_test, preds)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s | Accuracy: {metrics['accuracy']} | F1: {metrics['f1']}")
    return metrics


# ── 3. Fast-DetectGPT (Bao et al. ICLR 2024) ──

def fast_detectgpt_score(text, tok, model):
    inputs = tok(
        text, return_tensors="pt",
        truncation=True, max_length=CONFIG["max_length"],
    ).to(DEVICE)
    if inputs["input_ids"].shape[1] < 3:
        return 0.0

    with torch.no_grad():
        logits = model(**inputs).logits[0, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)

    shift_labels = inputs["input_ids"][0, 1:]
    actual_ll = log_probs[range(len(shift_labels)), shift_labels].mean().item()

    probs = torch.exp(log_probs).cpu()
    sampled = []
    for _ in range(CONFIG["n_samples"]):
        ids = torch.multinomial(probs, num_samples=1).squeeze(-1).to(DEVICE)
        s_ll = log_probs[range(len(ids)), ids].mean().item()
        sampled.append(s_ll)

    mu = np.mean(sampled)
    std = np.std(sampled) + 1e-8
    return (actual_ll - mu) / std


def run_fast_detectgpt(X_test, y_test, tok, model):
    print("\n" + "=" * 55)
    print("[3/4] Fast-DetectGPT (Bao et al. ICLR 2024)")
    t0 = time.time()

    scores = []
    for i, text in enumerate(X_test):
        if i % 100 == 0:
            print(f"  {i}/{len(X_test)}")
        scores.append(fast_detectgpt_score(text, tok, model))

    preds = [1 if s > 0 else 0 for s in scores]
    metrics = compute_metrics(y_test, preds)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s | Accuracy: {metrics['accuracy']} | F1: {metrics['f1']}")
    return metrics


# ── 4. RoBERTa fine-tuned (supervised) ──

class TextDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def run_finetuned_roberta(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 55)
    print("[4/4] RoBERTa fine-tuned on your ArXiv data")
    t0 = time.time()

    tok = RobertaTokenizer.from_pretrained("roberta-base")

    print("  Tokenizing train set ...")
    tr_enc = tok(X_train, truncation=True, padding="max_length",
                 max_length=CONFIG["max_length"], return_tensors="pt")
    print("  Tokenizing test set ...")
    te_enc = tok(X_test, truncation=True, padding="max_length",
                 max_length=CONFIG["max_length"], return_tensors="pt")

    tr_loader = DataLoader(TextDataset(tr_enc, y_train), batch_size=CONFIG["batch_size"],
                           shuffle=True, num_workers=0)
    te_loader = DataLoader(TextDataset(te_enc, y_test), batch_size=CONFIG["batch_size"],
                           shuffle=False, num_workers=0)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    total_steps = len(tr_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, best_state, patience = 0.0, None, 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0
        for batch in tr_loader:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            loss = model(input_ids=ids, attention_mask=mask, labels=lbls).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        model.eval()
        ep_preds, ep_labels = [], []
        with torch.no_grad():
            for batch in te_loader:
                out = model(input_ids=batch["input_ids"].to(DEVICE),
                            attention_mask=batch["attention_mask"].to(DEVICE))
                ep_preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
                ep_labels.extend(batch["labels"].numpy())

        ep_f1 = f1_score(ep_labels, ep_preds, average="weighted", zero_division=0)
        ep_acc = accuracy_score(ep_labels, ep_preds)
        avg_loss = epoch_loss / len(tr_loader)
        print(f"  Epoch {epoch:02d}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | "
              f"Acc: {ep_acc:.4f} | F1: {ep_f1:.4f}")

        if ep_f1 > best_f1:
            best_f1 = ep_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG["early_stopping"]:
                print(f"  [Early stopping] Epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in te_loader:
            out = model(input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE))
            preds.extend(torch.argmax(out.logits, 1).cpu().numpy())

    metrics = compute_metrics(y_test, preds)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s | Accuracy: {metrics['accuracy']} | F1: {metrics['f1']}")
    return metrics


# ── Final Table ──

def print_final_table(all_results, save_dir):
    print("\n\n" + "=" * 72)
    print("  BENCHMARK RESULTS - Baselines vs Your GNN Models")
    print("=" * 72)
    print(f"  {'Model':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("  " + "-" * 68)

    rows = []
    for name, metrics in all_results:
        if metrics is None:
            print(f"  {name:<40} FAILED")
            continue
        a, p, r, f, n = (
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], metrics["support"],
        )
        print(f"  {name:<40} {a:>6.4f} {p:>6.4f} {r:>6.4f} {f:>6.4f} {n:>5}")
        rows.append({"Model": name, **metrics})

    print("=" * 72)

    print("\n  [LaTeX rows - paste into Table 1]\n")
    print("  % -- Baseline Detectors ----")
    for name, metrics in all_results:
        if metrics is None:
            continue
        a, p, r, f, n = (
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], metrics["support"],
        )
        print(f"  {name} & {a:.4f} & {p:.4f} & {r:.4f} & {f:.4f} & {n} \\\\")

    if rows:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "benchmark_results.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\n  [Saved] {path}")


def main(args):
    set_seed(CONFIG["seed"])
    CONFIG["n_perturbations"] = args.n_perturbations

    texts, labels = load_data(args)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["seed"],
        stratify=labels,
    )
    print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")

    gpt2_tok, gpt2_model = load_gpt2()

    all_results = []

    metrics = run_gltr(X_test, y_test, X_train, y_train, gpt2_tok, gpt2_model)
    all_results.append(("GLTR (Gehrmann et al. 2019)", metrics))

    metrics = run_detectgpt(X_test, y_test, gpt2_tok, gpt2_model)
    all_results.append(("DetectGPT (Mitchell et al. 2023)", metrics))

    metrics = run_fast_detectgpt(X_test, y_test, gpt2_tok, gpt2_model)
    all_results.append(("Fast-DetectGPT (Bao et al. 2024)", metrics))

    del gpt2_model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    metrics = run_finetuned_roberta(X_train, X_test, y_train, y_test)
    all_results.append(("RoBERTa fine-tuned (supervised)", metrics))

    print_final_table(all_results, args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--human_file", type=str)
    parser.add_argument("--ai_file", type=str)
    parser.add_argument("--save_dir", type=str, default="./benchmark_output")
    parser.add_argument("--n_perturbations", type=int, default=CONFIG["n_perturbations"])
    args = parser.parse_args()
    main(args)
