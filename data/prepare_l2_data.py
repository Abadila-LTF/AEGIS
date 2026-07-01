"""
Prepare L2 English datasets for AEGIS fairness evaluation.

Supports multiple L2 corpora. Output: CSV with columns (text, l1, proficiency).

Supported sources:
  1. TOEFL11 (ETS corpus, requires LDC license)
  2. PELIC (Pittsburgh English Language Institute Corpus, free)
  3. ICNALE (International Corpus Network of Asian Learners, free with registration)
  4. Custom directory of text files organized by L1

Usage:
    # From TOEFL11 (LDC)
    python prepare_l2_data.py --source toefl11 --input /path/to/ETS_Corpus_of_Non-Native_Written_English --output l2_essays.csv

    # From PELIC (free download from https://eli-data-mining-group.github.io/Pelic-dataset/)
    python prepare_l2_data.py --source pelic --input /path/to/PELIC_compiled.csv --output l2_essays.csv

    # From a custom directory (subdirs named by L1)
    python prepare_l2_data.py --source custom --input /path/to/essays/ --output l2_essays.csv
    # Expected structure: essays/Arabic/*.txt, essays/Chinese/*.txt, etc.
"""

import os
import re
import glob
import argparse
import pandas as pd


def load_toefl11(input_path):
    """Load TOEFL11 corpus (ETS Corpus of Non-Native Written English).

    Expected structure:
        input_path/
          data/
            text/
              responses/
                original/
                  {lang}/
                    {essay_id}.txt
          index.csv  (with columns: Filename, Language, Score Level, ...)
    """
    index_path = os.path.join(input_path, "index.csv")
    if os.path.exists(index_path):
        idx = pd.read_csv(index_path)
        rows = []
        for _, r in idx.iterrows():
            lang = r.get("Language", r.get("Prompt Language", ""))
            score = r.get("Score Level", r.get("Proficiency", ""))
            fname = r.get("Filename", "")
            fpath = os.path.join(input_path, "data", "text", "responses", "original", fname)
            if not os.path.exists(fpath):
                fpath = os.path.join(input_path, fname)
            if os.path.exists(fpath):
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
                if len(text) > 50:
                    rows.append({"text": text, "l1": lang, "proficiency": str(score)})
        return pd.DataFrame(rows)

    # Fallback: directory-based structure
    rows = []
    text_dirs = glob.glob(os.path.join(input_path, "**", "*.txt"), recursive=True)
    for fpath in text_dirs:
        parts = fpath.split(os.sep)
        lang = "unknown"
        for p in parts:
            if p in ["ARA", "CHI", "FRE", "GER", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR"]:
                lang_map = {
                    "ARA": "Arabic", "CHI": "Chinese", "FRE": "French",
                    "GER": "German", "HIN": "Hindi", "ITA": "Italian",
                    "JPN": "Japanese", "KOR": "Korean", "SPA": "Spanish",
                    "TEL": "Telugu", "TUR": "Turkish",
                }
                lang = lang_map.get(p, p)
                break
        with open(fpath, encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if len(text) > 50:
            rows.append({"text": text, "l1": lang, "proficiency": ""})

    return pd.DataFrame(rows)


def load_pelic(input_path):
    """Load PELIC corpus (free, downloadable CSV).

    PELIC has columns like: anon_id, L1, level_id, text, ...
    Download from: https://eli-data-mining-group.github.io/Pelic-dataset/
    """
    df = pd.read_csv(input_path)

    text_col = None
    for candidate in ["text", "tok_text", "answer", "response", "essay"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Could not find text column in PELIC data")

    l1_col = None
    for candidate in ["L1", "l1", "native_language", "first_language"]:
        if candidate in df.columns:
            l1_col = candidate
            break
    if l1_col is None:
        raise ValueError("Could not find L1 column in PELIC data")

    prof_col = None
    for candidate in ["level_id", "proficiency", "level", "score"]:
        if candidate in df.columns:
            prof_col = candidate
            break

    rows = []
    for _, r in df.iterrows():
        text = str(r[text_col]).strip()
        if len(text) > 50:
            rows.append({
                "text": text,
                "l1": str(r[l1_col]).strip(),
                "proficiency": str(r[prof_col]).strip() if prof_col else "",
            })

    return pd.DataFrame(rows)


def load_custom(input_path):
    """Load from a directory structure: input_path/{L1_language}/*.txt"""
    rows = []
    for lang_dir in sorted(os.listdir(input_path)):
        lang_path = os.path.join(input_path, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        for txt_file in glob.glob(os.path.join(lang_path, "*.txt")):
            with open(txt_file, encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            if len(text) > 50:
                rows.append({"text": text, "l1": lang_dir, "proficiency": ""})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare L2 English data for fairness evaluation")
    parser.add_argument("--source", required=True, choices=["toefl11", "pelic", "custom"],
                        help="Which corpus format to load")
    parser.add_argument("--input", required=True, help="Path to corpus (directory or CSV)")
    parser.add_argument("--output", default="l2_essays.csv", help="Output CSV path")
    parser.add_argument("--min_samples", type=int, default=10,
                        help="Minimum samples per L1 group (groups below this are dropped)")
    args = parser.parse_args()

    if args.source == "toefl11":
        df = load_toefl11(args.input)
    elif args.source == "pelic":
        df = load_pelic(args.input)
    elif args.source == "custom":
        df = load_custom(args.input)

    print(f"Loaded {len(df)} essays from {args.source}")
    print(f"L1 distribution:\n{df['l1'].value_counts().to_string()}")

    # Filter small groups
    counts = df["l1"].value_counts()
    keep = counts[counts >= args.min_samples].index
    dropped = counts[counts < args.min_samples]
    if len(dropped) > 0:
        print(f"\nDropping groups with <{args.min_samples} samples: {dict(dropped)}")
    df = df[df["l1"].isin(keep)].reset_index(drop=True)

    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} essays ({df['l1'].nunique()} L1 groups) to {args.output}")


if __name__ == "__main__":
    main()
