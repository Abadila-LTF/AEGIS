# AEGIS: Graph Attention Networks for AI-Generated Text Detection

This repository contains the implementation code and datasets for the paper:

> **AEGIS: Preserving Academic Integrity via Graph Attention Networks for AI-Generated Text Detection**
>
> Abadila Alaktif, Meriyem Chergui, Faiq Gmira, Abdelkarim Ammoumou, Saadia Drissi



## Overview

AEGIS introduces a novel graph-based paradigm for AI-generated text detection that represents documents as semantic graphs and processes them using graph neural networks. Unlike all prior methods that treat text as flat token sequences, AEGIS models documents as semantic graphs and exploits relational and topological patterns that are structurally invisible to sequential classifiers.

**Key results on ArXiv-2K (LLaMA 3.1-generated scientific abstracts):**

| Model | Accuracy | F1 | Parameters | Size |
|-------|----------|-----|-----------|------|
| AEGIS Model 2 (Enhanced GAT) | **98.50%** | **0.9850** | 73K | 0.28 MB |
| AEGIS Model 3 (PPL-GAT) | 97.50% | 0.9750 | 71.5K | 0.27 MB |
| RoBERTa fine-tuned (best baseline) | 82.53% | 0.8250 | 124.6M | 475 MB |

AEGIS achieves a 16-point F1 margin over the best supervised baseline with **1,708x fewer parameters**.

## Repository Structure

```
aegis-repo/
├── models/                         # Core GNN architectures
│   ├── model1_gcn.py               # Model 1: Simple GCN + GloVe (proof of concept)
│   ├── model2_gat.py               # Model 2: Enhanced GAT + RoBERTa + spaCy (best)
│   └── model3_ppl_gat.py           # Model 3: Perplexity-aware GAT with learnable edge weights
├── data/                           # Dataset construction pipeline
│   ├── rewritten_texts_file_2K_1.csv  # ArXiv-2K dataset (4,000 samples)
│   ├── arxiv_extract.py            # Extract human abstracts from arXiv JSON metadata
│   ├── llm_rewrite.py              # Generate AI rewrites using Ollama / LLaMA 3.1
│   └── prepare_dataset.py          # Combine human + AI texts with binary labels
├── benchmarks/                     # Baseline comparison scripts
│   ├── benchmark_detectors.py      # GLTR, DetectGPT, Fast-DetectGPT, RoBERTa
│   ├── benchmark_zerogpt.py        # ZeroGPT commercial API benchmark
│   └── profile_computational.py    # Fair computational profiling (isolated processes)
├── analysis/                       # Visualization and structural analysis
│   ├── graph_structure_analysis.py # Clustering coefficients, modularity, t-tests
│   ├── tsne_visualization.py       # t-SNE embedding space visualization
│   └── gnn_utils.py                # GNN inference utilities
├── requirements.txt
└── .gitignore
```

## Models

### Model 1 — Simple GCN (AEGIS)
- **Node features:** GloVe 100d static embeddings
- **Graph construction:** Cosine similarity thresholding (> 0.5)
- **Architecture:** 2× GCNConv → global mean pool → linear classifier
- **Result:** 74.17% accuracy (proof of concept)

### Model 2 — Enhanced GAT (AEGIS II)
- **Node features:** RoBERTa 768d + POS tags (19d) + position (1d) + stopword/punctuation flags (2d) = **790d**
- **Graph construction:** k-NN (k=8) on feature vectors + spaCy dependency edges
- **Architecture:** 3× GATConv (8 heads) + residual connections + hierarchical pooling + multi-task learning
- **Result:** **98.50% accuracy**, 100% human recall

### Model 3 — Perplexity-aware GAT (AEGIS-PPL)
- **Node features:** RoBERTa 768d embeddings
- **Edge construction:** Learnable weighted combination of cosine similarity, attention scores, token entropy, and token perplexity
- **Architecture:** 3× GATConv (8 heads) + learnable edge component weights (α, β, γ, δ)
- **Result:** 97.50% accuracy

## Dataset

**ArXiv-2K** (`data/rewritten_texts_file_2K_1.csv`): 4,000 balanced samples — 2,000 human-written arXiv abstracts and 2,000 AI-generated counterparts produced by LLaMA 3.1 via Ollama.

| Column | Description |
|--------|-------------|
| `text` | The abstract text |
| `generated` | Label: 0 = human-written, 1 = AI-generated |

### Reproducing the dataset from scratch

1. Download the [arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) (JSON)
2. Extract human abstracts:
   ```bash
   python data/arxiv_extract.py --input /path/to/arxiv-metadata-oai-snapshot.json --output data/arxiv_abstracts.csv
   ```
3. Generate AI rewrites (requires [Ollama](https://ollama.ai) with `llama3.1`):
   ```bash
   python data/llm_rewrite.py --input data/arxiv_abstracts.csv --output data/rewritten.csv
   ```
4. Combine into labeled dataset:
   ```bash
   python data/prepare_dataset.py --input data/rewritten.csv --output data/dataset.csv
   ```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For Model 1, you also need NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Running the Models

Each model file is self-contained with its own training loop. Pass the dataset via `--data`:

```bash
# Model 1 (GCN baseline)
python models/model1_gcn.py --data data/rewritten_texts_file_2K_1.csv

# Model 2 (Enhanced GAT — best performance)
python models/model2_gat.py --data data/rewritten_texts_file_2K_1.csv

# Model 3 (Perplexity-aware GAT)
python models/model3_ppl_gat.py --data data/rewritten_texts_file_2K_1.csv
```

## Running Benchmarks

```bash
# Benchmark against GLTR, DetectGPT, Fast-DetectGPT, and RoBERTa
python benchmarks/benchmark_detectors.py --data data/rewritten_texts_file_2K_1.csv

# ZeroGPT API benchmark (requires API key)
python benchmarks/benchmark_zerogpt.py --dataset data/rewritten_texts_file_2K_1.csv --api_key YOUR_KEY

# Computational profiling (memory, latency, training time)
python benchmarks/profile_computational.py --data data/rewritten_texts_file_2K_1.csv
```

## Computational Profiling

All models were profiled on Apple M3 Pro (36 GB unified memory) using `torch.mps.current_allocated_memory()` to report GPU-allocated memory only. Each model was profiled in an isolated subprocess for fair comparison.

| Metric | Model 1 (GCN) | Model 2 (GAT) | Model 3 (PPL) | RoBERTa FT |
|--------|---------------|---------------|---------------|------------|
| Parameters | ~10K | ~73K | ~71K | ~124M |
| Size on disk | 0.04 MB | 0.28 MB | 0.27 MB | 475 MB |
| GPU memory (inference) | 0.4 MB | 1.0 MB | 1.0 MB | 475.5 MB |
| GPU memory (training) | 5.0 MB | 29.7 MB | 32.7 MB | 2,865 MB |
| End-to-end latency | 67 ms | 199 ms | 168 ms | 23 ms |

## Structural Findings

Graph analysis reveals fundamental topological differences between human and AI-authored text:

- **Clustering coefficient:** Human 0.4681 vs AI 0.4380 (p < 0.001)
- **Community modularity:** Human 0.6159 vs AI 0.5826 (p < 0.001)

Human writing exhibits richer local neighborhoods and sharper community boundaries — hallmarks of hierarchical planning — while AI text shows smoother, more homogeneous connectivity consistent with autoregressive token prediction.

## Citation

```bibtex
@article{alaktif2026aegis,
  title={AEGIS: Preserving Academic Integrity via Graph Attention Networks for AI-Generated Text Detection},
  author={Alaktif, Abadila and Chergui, Meriyem and Gmira, Faiq and Ammoumou, Abdelkarim and Drissi, Saadia},
  journal={-},
  year={2026}
}
```

## License

This project is released for academic and research purposes.
