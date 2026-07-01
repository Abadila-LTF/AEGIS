"""
Surface-features baseline for Ch4 hardening.

Logistic regression on 5 shallow features:
  1. word_count
  2. type_token_ratio (TTR)
  3. mean_word_length
  4. stopword_ratio
  5. punctuation_density

Reports accuracy, AUROC, FPR@5% TPR threshold, and prints a comparison
table against AEGIS M2's published numbers. Runs 3-fold stratified CV
to match the thesis rigor bar (no single-split luck).
"""

import argparse
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

STOPWORDS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are was "
    "were be been being have has had having do does did doing a an the and but "
    "if or because as until while of at by for with about against between "
    "through during before after above below to from up down in out on off "
    "over under again further then once here there when where why how all both "
    "each few more most other some such no nor not only own same so than too "
    "very s t can will just don should now d ll m o re ve y ain aren couldn "
    "didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn "
    "weren won wouldn".split()
)


def extract_features(text: str) -> dict:
    text = str(text)
    words = text.lower().split()
    n_words = len(words) or 1
    chars = list(text)
    n_chars = len(chars) or 1

    unique_words = len(set(words))
    ttr = unique_words / n_words

    mean_word_len = sum(len(w) for w in words) / n_words

    stop_count = sum(1 for w in words if w in STOPWORDS)
    stopword_ratio = stop_count / n_words

    punct_count = sum(1 for c in chars if c in '.,;:!?-()[]{}"\'/\\')
    punct_density = punct_count / n_chars

    return {
        "word_count": n_words,
        "ttr": ttr,
        "mean_word_length": mean_word_len,
        "stopword_ratio": stopword_ratio,
        "punctuation_density": punct_density,
    }


def fpr_at_tpr(y_true, y_score, tpr_threshold=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, tpr_threshold)
    if idx >= len(fpr):
        return fpr[-1]
    return fpr[idx]


def main():
    parser = argparse.ArgumentParser(description="Surface-features baseline")
    parser.add_argument("--data", required=True, help="Path to dataset CSV (text, generated)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples  (human={sum(df['generated']==0)}, AI={sum(df['generated']==1)})")

    print("\nExtracting surface features...")
    feat_dicts = df["text"].apply(extract_features)
    feat_df = pd.DataFrame(feat_dicts.tolist())
    X = feat_df.values
    y = df["generated"].values

    print("\nFeature summary by class:")
    feat_df["generated"] = y
    print(feat_df.groupby("generated").mean().round(4).to_string())
    feat_df = feat_df.drop(columns=["generated"])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    accs, aurocs, fprs_at_5 = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=args.seed)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_prob)
        fpr5 = fpr_at_tpr(y_test, y_prob, tpr_threshold=0.95)

        accs.append(acc)
        aurocs.append(auroc)
        fprs_at_5.append(fpr5)

        print(f"\nFold {fold}: Acc={acc:.4f}  AUROC={auroc:.4f}  FPR@95%TPR={fpr5:.4f}")

    print("\n" + "=" * 60)
    print("SURFACE BASELINE  (logistic regression, 5 features)")
    print("=" * 60)
    print(f"  Accuracy:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  AUROC:        {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  FPR@95%TPR:   {np.mean(fprs_at_5):.4f} ± {np.std(fprs_at_5):.4f}")
    print()

    # Final full-data model for feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(X_scaled, y)

    feature_names = list(feat_df.columns)
    coefs = clf.coef_[0]
    print("Feature coefficients (full-data model):")
    for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:25s}  {coef:+.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON TABLE (for thesis)")
    print("=" * 60)
    print(f"  {'Method':<30s}  {'Acc':>7s}  {'AUROC':>7s}  {'FPR@95':>8s}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*8}")
    print(f"  {'Surface LR (5 feat)':<30s}  {np.mean(accs):7.4f}  {np.mean(aurocs):7.4f}  {np.mean(fprs_at_5):8.4f}")
    print(f"  {'AEGIS M2 (73K params)':<30s}  {'0.9850':>7s}  {'  —':>7s}  {'   —':>8s}")
    print(f"  {'RoBERTa-FT (125M params)':<30s}  {'0.8250':>7s}  {'  —':>7s}  {'   —':>8s}")
    print()
    print("Note: AEGIS M2 AUROC/FPR@95% will be filled by metric_upgrade.py")


if __name__ == "__main__":
    main()
