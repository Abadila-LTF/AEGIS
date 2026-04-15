"""
Combine original human abstracts with LLaMA-rewritten AI texts into a
single labelled dataset for binary classification.

Input CSV must have columns: text, rewritten_text
Output CSV has columns: text, generated (0 = human, 1 = AI)

Usage:
    python prepare_dataset.py --input rewritten.csv --output dataset.csv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build labelled dataset")
    parser.add_argument("--input", required=True, help="CSV with text and rewritten_text columns")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    rows = []
    for _, row in df.iterrows():
        rows.append({"text": row["text"], "generated": 0})
        rows.append({"text": row["rewritten_text"], "generated": 1})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"Dataset created: {len(out_df)} samples ({len(df)} human + {len(df)} AI)")


if __name__ == "__main__":
    main()
