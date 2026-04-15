"""
Generate AI-rewritten versions of human abstracts using Ollama (LLaMA 3.1).

Requires Ollama running locally with the llama3.1 model pulled.

Usage:
    python llm_rewrite.py --input arxiv_abstracts.csv --output rewritten.csv
"""

import argparse
import pandas as pd
import ollama
from tqdm import tqdm


PROMPT_TEMPLATE = (
    "Please rewrite the following text, keeping it approximately the same length. "
    "Do not include any explanations or additional text; only provide the rewritten text.\n\n"
    "Text:\n{text}\n\nRewritten Text:"
)


def rewrite_text(text, model="llama3.1"):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
    )
    return response["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Rewrite abstracts with LLaMA 3.1")
    parser.add_argument("--input", required=True, help="CSV with human abstracts (text, generated)")
    parser.add_argument("--output", required=True, help="Output CSV with original + rewritten")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rewriting"):
        rewritten = rewrite_text(row["text"], model=args.model)
        results.append({"text": row["text"], "rewritten_text": rewritten})

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} pairs to {args.output}")


if __name__ == "__main__":
    main()
