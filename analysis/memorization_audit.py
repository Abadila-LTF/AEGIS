"""
Memorization audit for Ch4 hardening.

Computes similarity between each human abstract and its LLaMA rewrite:
  - BLEU-4 (sentence-level)
  - Character-level edit distance ratio (1 - normalized Levenshtein)
  - Word overlap (Jaccard)

Flags near-copies (BLEU > 0.8 or edit_sim > 0.9) as potential label noise.
Reports headline-number robustness: accuracy when near-copies are removed
from the test set of a logistic-regression surface baseline.
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def sentence_bleu4(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    brevity_penalty = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    precisions = []
    for n in range(1, 5):
        ref_ng = Counter(ngrams(ref_tokens, n))
        hyp_ng = Counter(ngrams(hyp_tokens, n))
        clipped = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        total = max(sum(hyp_ng.values()), 1)
        precisions.append(clipped / total)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(np.log(p) for p in precisions) / 4
    return brevity_penalty * np.exp(log_avg)


def edit_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    la, lb = len(a), len(b)
    if la > 2000 or lb > 2000:
        a, b = a[:2000], b[:2000]
        la, lb = len(a), len(b)

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def word_jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main():
    parser = argparse.ArgumentParser(description="Memorization audit")
    parser.add_argument("--data", required=True, help="Path to dataset CSV (interleaved human/AI pairs)")
    parser.add_argument("--bleu_threshold", type=float, default=0.8)
    parser.add_argument("--edit_threshold", type=float, default=0.9)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    assert len(df) % 2 == 0, "Dataset must have even number of rows (interleaved pairs)"
    n_pairs = len(df) // 2

    print(f"Auditing {n_pairs} human-AI pairs for memorization / near-copying...\n")

    bleu_scores = []
    edit_sims = []
    jaccard_scores = []

    for i in range(n_pairs):
        human_text = str(df.iloc[2 * i]["text"])
        ai_text = str(df.iloc[2 * i + 1]["text"])

        assert df.iloc[2 * i]["generated"] == 0, f"Row {2*i} should be human"
        assert df.iloc[2 * i + 1]["generated"] == 1, f"Row {2*i+1} should be AI"

        bleu = sentence_bleu4(human_text, ai_text)
        edit = edit_similarity(human_text, ai_text)
        jacc = word_jaccard(human_text, ai_text)

        bleu_scores.append(bleu)
        edit_sims.append(edit)
        jaccard_scores.append(jacc)

    bleu_scores = np.array(bleu_scores)
    edit_sims = np.array(edit_sims)
    jaccard_scores = np.array(jaccard_scores)

    print("=" * 60)
    print("SIMILARITY STATISTICS (human ↔ LLaMA rewrite)")
    print("=" * 60)
    for name, scores in [("BLEU-4", bleu_scores), ("Edit similarity", edit_sims), ("Word Jaccard", jaccard_scores)]:
        print(f"\n  {name}:")
        print(f"    mean={scores.mean():.4f}  std={scores.std():.4f}")
        print(f"    min={scores.min():.4f}  25%={np.percentile(scores, 25):.4f}  "
              f"50%={np.percentile(scores, 50):.4f}  75%={np.percentile(scores, 75):.4f}  "
              f"max={scores.max():.4f}")

    near_copy_bleu = bleu_scores > args.bleu_threshold
    near_copy_edit = edit_sims > args.edit_threshold
    near_copy_either = near_copy_bleu | near_copy_edit

    print(f"\n{'=' * 60}")
    print(f"NEAR-COPY DETECTION")
    print(f"{'=' * 60}")
    print(f"  BLEU > {args.bleu_threshold}: {near_copy_bleu.sum()} pairs ({100*near_copy_bleu.mean():.1f}%)")
    print(f"  Edit > {args.edit_threshold}: {near_copy_edit.sum()} pairs ({100*near_copy_edit.mean():.1f}%)")
    print(f"  Either:  {near_copy_either.sum()} pairs ({100*near_copy_either.mean():.1f}%)")

    if near_copy_either.sum() > 0:
        flagged_idx = np.where(near_copy_either)[0]
        print(f"\n  Flagged pair indices: {flagged_idx[:20].tolist()}" +
              (f" ... (+{len(flagged_idx)-20} more)" if len(flagged_idx) > 20 else ""))

        print(f"\n  Worst near-copies (top 5 by BLEU):")
        top5 = np.argsort(bleu_scores)[-5:][::-1]
        for idx in top5:
            human_text = str(df.iloc[2 * idx]["text"])
            ai_text = str(df.iloc[2 * idx + 1]["text"])
            print(f"\n    Pair {idx}: BLEU={bleu_scores[idx]:.4f}  Edit={edit_sims[idx]:.4f}")
            print(f"    Human: {human_text[:120]}...")
            print(f"    AI:    {ai_text[:120]}...")

    # Robustness check: remove near-copy pairs and re-run surface baseline
    clean_mask = ~near_copy_either
    n_clean = clean_mask.sum()
    n_removed = near_copy_either.sum()

    print(f"\n{'=' * 60}")
    print(f"ROBUSTNESS CHECK (surface baseline, near-copies removed)")
    print(f"{'=' * 60}")
    print(f"  Original pairs: {n_pairs}")
    print(f"  Removed:        {n_removed}")
    print(f"  Clean pairs:    {n_clean}")

    if n_removed == 0:
        print("\n  No near-copies found — headline numbers are clean.")
        return

    clean_human_idx = [2 * i for i in range(n_pairs) if clean_mask[i]]
    clean_ai_idx = [2 * i + 1 for i in range(n_pairs) if clean_mask[i]]
    clean_df = df.iloc[clean_human_idx + clean_ai_idx].reset_index(drop=True)

    from surface_baseline import extract_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    feat_dicts = clean_df["text"].apply(extract_features)
    feat_df = pd.DataFrame(feat_dicts.tolist())
    X = feat_df.values
    y = clean_df["generated"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aurocs = [], []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X_te)))
        aurocs.append(roc_auc_score(y[test_idx], clf.predict_proba(X_te)[:, 1]))

    print(f"\n  Surface baseline (all pairs):   Acc=0.8097  AUROC=0.8807")
    print(f"  Surface baseline (clean pairs): Acc={np.mean(accs):.4f}  AUROC={np.mean(aurocs):.4f}")
    delta_acc = np.mean(accs) - 0.8097
    print(f"  Delta:                          {delta_acc:+.4f} acc / {np.mean(aurocs)-0.8807:+.4f} AUROC")
    print(f"\n  Verdict: {'Near-copies inflate scores' if delta_acc < -0.01 else 'Headline numbers are robust to near-copy removal'}")


if __name__ == "__main__":
    main()
