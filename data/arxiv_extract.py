"""
Extract human-written abstracts from the arXiv metadata JSON snapshot.

Usage:
    python arxiv_extract.py --input /path/to/arxiv-metadata-oai-snapshot.json \
                            --output arxiv_abstracts.csv
"""

import argparse
import csv
import ijson


def extract_abstracts(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8", newline="") as f_out:

        writer = csv.DictWriter(f_out, fieldnames=["text", "generated"])
        writer.writeheader()

        for obj in ijson.items(f_in, "item"):
            abstract = obj.get("abstract", "").strip()
            if abstract:
                writer.writerow({"text": abstract, "generated": 0})

    print(f"Abstracts written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract arXiv abstracts")
    parser.add_argument("--input", required=True, help="Path to arXiv JSON snapshot")
    parser.add_argument("--output", default="arxiv_abstracts.csv", help="Output CSV path")
    args = parser.parse_args()
    extract_abstracts(args.input, args.output)
