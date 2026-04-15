"""
Fair Computational Profiling — each model in its own subprocess.

Reports model-only GPU memory (MPS-allocated, no shared runtime overhead).
Uses real texts from the dataset for preprocessing timing.

Usage:
    python profile_computational.py --data ../data/dataset.csv
"""

import subprocess
import sys
import json
import os
import argparse

PYTHON = sys.executable

WORKER_CODE = r'''
import os, sys, time, gc, json, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, ModuleList, Parameter
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import psutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

random.seed(42); np.random.seed(42); torch.manual_seed(42)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
CSV_PATH = sys.argv[2]
N_PREPROC = 30
MAX_TOKENS = 256
K_NEIGHBORS = 8

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def mps_mb():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory() / 1024**2
    return 0.0

def mps_driver_mb():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory() / 1024**2
    return 0.0

def reset_mps():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()

def sync():
    if DEVICE.type == "mps": torch.mps.synchronize()

# ── Model definitions ──

class GCN(torch.nn.Module):
    def __init__(self, num_features=100, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = Dropout(0.5)
        self.lin = Linear(hidden, 2)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index)); x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index)); x = self.dropout(x)
        return self.lin(global_mean_pool(x, batch))

class Model2_GAT(torch.nn.Module):
    def __init__(self, nf=790, hc=64, nl=3, dr=0.3):
        super().__init__()
        self.nl = nl
        self.input_proj = Linear(nf, hc)
        self.gats = ModuleList([GATConv(hc, hc//8, heads=8, concat=True, dropout=dr, add_self_loops=True, edge_dim=1) for _ in range(nl)])
        self.lns = ModuleList([LayerNorm(hc) for _ in range(nl)])
        self.pool_attn = Linear(hc, 1)
        self.lin1 = Linear(hc*2, hc); self.drop = Dropout(dr); self.lin2 = Linear(hc, 2)
        self.cr = Linear(hc, 1); self.rc = Linear(hc, 3)
    def forward(self, x, ei, ea, batch):
        x = self.input_proj(x)
        for i in range(self.nl):
            g = self.gats[i](x, ei, edge_attr=ea)
            x = g + x if i > 0 else g
            x = F.elu(self.lns[i](x))
            if i < self.nl - 1: x = self.drop(x)
        xp = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], 1)
        h = self.drop(F.relu(self.lin1(xp)))
        return self.lin2(h), self.cr(h), self.rc(h)

class Model3_PPL(torch.nn.Module):
    def __init__(self, nf=768, hc=64, nl=3, dr=0.3):
        super().__init__()
        self.nl = nl
        self.ecw = Parameter(torch.tensor([.25,.25,.25,.25]))
        self.input_proj = Linear(nf, hc)
        self.gats = ModuleList([GATConv(hc, hc//8, heads=8, concat=True, dropout=dr, add_self_loops=True, edge_dim=1) for _ in range(nl)])
        self.lns = ModuleList([LayerNorm(hc) for _ in range(nl)])
        self.lin1 = Linear(hc*2, hc); self.drop = Dropout(dr); self.lin2 = Linear(hc, 2)
        self.cr = Linear(hc, 1); self.rc = Linear(hc, 3)
    def forward(self, x, ei, ea4, batch):
        x = self.input_proj(x)
        w = F.softmax(self.ecw, dim=0)
        ea = (ea4 * w.unsqueeze(0)).sum(1, keepdim=True)
        for i in range(self.nl):
            g = self.gats[i](x, ei, edge_attr=ea)
            x = g + x if i > 0 else g
            x = F.elu(self.lns[i](x))
            if i < self.nl - 1: x = self.drop(x)
        xp = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], 1)
        h = self.drop(F.relu(self.lin1(xp)))
        return self.lin2(h), self.cr(h), self.rc(h)


def make_batch(model_name, bs=16, nn=128):
    from torch_geometric.data import Batch
    graphs = []
    for _ in range(bs):
        n = random.randint(50, nn)
        if model_name == "m1":
            x = torch.randn(n, 100)
            ei_list = []
            for i in range(n):
                for j in range(i+1, min(i+8, n)):
                    ei_list.extend([[i,j],[j,i]])
            ei = torch.tensor(ei_list, dtype=torch.long).t() if ei_list else torch.empty(2,0,dtype=torch.long)
            graphs.append(Data(x=x, edge_index=ei, y=torch.randint(0,2,(1,))))
        elif model_name == "m2":
            x = torch.randn(n, 790)
            nbrs = NearestNeighbors(n_neighbors=min(9,n)).fit(x.numpy())
            _, idx = nbrs.kneighbors(x.numpy())
            r,c=[],[]
            for i in range(n):
                for j in idx[i][1:]: r.extend([i,j]); c.extend([j,i])
            ei = torch.tensor([r,c], dtype=torch.long)
            ea = torch.ones(ei.size(1),1)
            graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, y=torch.randint(0,2,(1,))))
        elif model_name == "m3":
            x = torch.randn(n, 768)
            nbrs = NearestNeighbors(n_neighbors=min(9,n)).fit(x.numpy())
            _, idx = nbrs.kneighbors(x.numpy())
            r,c=[],[]
            for i in range(n):
                for j in idx[i][1:]: r.extend([i,j]); c.extend([j,i])
            ei = torch.tensor([r,c], dtype=torch.long)
            ea = torch.rand(ei.size(1),4)
            graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, y=torch.randint(0,2,(1,))))
    return Batch.from_data_list(graphs).to(DEVICE)


def profile_model(model_name):
    results = {}

    reset_mps()
    baseline_mps = mps_mb()

    if model_name == "m1":
        model = GCN(100, 64).to(DEVICE)
    elif model_name == "m2":
        model = Model2_GAT(790, 64, 3).to(DEVICE)
    elif model_name == "m3":
        model = Model3_PPL(768, 64, 3).to(DEVICE)
    elif model_name == "roberta":
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        tok = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(DEVICE)

    sync()
    after_load_mps = mps_mb()
    results["mem_model_load_mb"] = round(after_load_mps - baseline_mps, 1)
    results["params"] = sum(p.numel() for p in model.parameters())
    results["disk_mb"] = round(results["params"] * 4 / 1024**2, 2)

    # ── Inference timing ──
    model.eval()
    # warmup
    for _ in range(3):
        if model_name in ("m1","m2","m3"):
            b = make_batch(model_name, bs=1)
            with torch.no_grad():
                if model_name=="m1": model(b.x, b.edge_index, b.batch)
                else: model(b.x, b.edge_index, b.edge_attr, b.batch)
        else:
            inp = tok("Test sentence for warmup.", truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            with torch.no_grad():
                model(input_ids=inp["input_ids"].to(DEVICE), attention_mask=inp["attention_mask"].to(DEVICE))
    sync()

    reset_mps()
    before_inf_mps = mps_mb()
    peak_inf_mps = before_inf_mps

    inf_times = []
    for _ in range(30):
        if model_name in ("m1","m2","m3"):
            b = make_batch(model_name, bs=1)
            sync(); t0 = time.perf_counter()
            with torch.no_grad():
                if model_name=="m1": model(b.x, b.edge_index, b.batch)
                else: model(b.x, b.edge_index, b.edge_attr, b.batch)
            sync(); inf_times.append((time.perf_counter()-t0)*1000)
            cur = mps_mb()
            if cur > peak_inf_mps: peak_inf_mps = cur
        else:
            inp = tok("A sample scientific abstract about quantum field theory.", truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            sync(); t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids=inp["input_ids"].to(DEVICE), attention_mask=inp["attention_mask"].to(DEVICE))
            sync(); inf_times.append((time.perf_counter()-t0)*1000)
            cur = mps_mb()
            if cur > peak_inf_mps: peak_inf_mps = cur

    results["inference_ms"] = round(float(np.median(inf_times)), 2)
    results["mem_inference_mb"] = round(peak_inf_mps, 1)

    # ── Training step timing + peak MPS memory ──
    reset_mps()
    before_train_mps = mps_mb()
    peak_train_mps = before_train_mps

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if model_name!="roberta" else 2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    train_times = []
    for step in range(15):
        optimizer.zero_grad()
        if model_name in ("m1","m2","m3"):
            b = make_batch(model_name, bs=16)
            sync(); t0 = time.perf_counter()
            if model_name=="m1": out = model(b.x, b.edge_index, b.batch)
            else:
                out = model(b.x, b.edge_index, b.edge_attr, b.batch)
                out = out[0]
            loss = criterion(out, b.y.view(-1))
        else:
            inp = tok(["Sample text for training benchmark."]*16, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            labels = torch.randint(0,2,(16,)).to(DEVICE)
            sync(); t0 = time.perf_counter()
            loss = model(input_ids=inp["input_ids"].to(DEVICE), attention_mask=inp["attention_mask"].to(DEVICE), labels=labels).loss
        loss.backward()
        optimizer.step()
        sync()
        train_times.append((time.perf_counter()-t0)*1000)
        cur = mps_mb()
        if cur > peak_train_mps: peak_train_mps = cur

    results["train_step_ms"] = round(float(np.median(train_times)), 2)
    results["mem_training_mb"] = round(peak_train_mps, 1)
    results["mem_train_driver_mb"] = round(mps_driver_mb(), 1)

    # ── Preprocessing timing (real texts) ──
    df = pd.read_csv(CSV_PATH)
    texts = df["text"].astype(str).tolist()[:N_PREPROC]

    if model_name == "m1":
        pp_times = []
        for text in texts:
            t0 = time.perf_counter()
            tokens = text.lower().split()[:MAX_TOKENS]
            embs = np.random.randn(len(tokens), 100).astype(np.float32)
            n = len(embs)
            norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
            sim = norms @ norms.T
            np.fill_diagonal(sim, 0)
            _ = np.argwhere(sim > 0.5)
            pp_times.append((time.perf_counter()-t0)*1000)
        results["preproc_ms"] = round(float(np.median(pp_times)), 1)

    elif model_name == "m2":
        import spacy
        from transformers import AutoModel, AutoTokenizer
        nlp = spacy.load("en_core_web_sm")
        tok2 = AutoTokenizer.from_pretrained("roberta-base")
        lm = AutoModel.from_pretrained("roberta-base").to(DEVICE); lm.eval()
        pp_times = []
        for text in texts:
            t0 = time.perf_counter()
            doc = nlp(text[:10000])
            tokens = [t.text for t in doc][:MAX_TOKENS]
            inp = tok2(" ".join(tokens), return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(DEVICE)
            with torch.no_grad():
                embs = lm(**inp).last_hidden_state[0,:len(tokens)].cpu().numpy()
            if len(embs) > 1:
                NearestNeighbors(n_neighbors=min(9,len(embs))).fit(embs).kneighbors(embs)
            sync()
            pp_times.append((time.perf_counter()-t0)*1000)
        results["preproc_ms"] = round(float(np.median(pp_times)), 1)
        del lm

    elif model_name == "m3":
        import spacy
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        nlp = spacy.load("en_core_web_sm")
        tok3 = AutoTokenizer.from_pretrained("roberta-base")
        lm = AutoModelForMaskedLM.from_pretrained("roberta-base").to(DEVICE); lm.eval()
        vocab_size = tok3.vocab_size
        log_v = math.log(vocab_size)
        pp_times = []
        for text in texts:
            t0 = time.perf_counter()
            inp = tok3(text, return_tensors="pt", padding="max_length", truncation=True,
                       max_length=MAX_TOKENS, return_attention_mask=True).to(DEVICE)
            nt = inp["attention_mask"][0].sum().item()
            with torch.no_grad():
                out = lm(inp["input_ids"], attention_mask=inp["attention_mask"],
                         output_attentions=True, output_hidden_states=True)
                embs = out.hidden_states[-1][0,:nt].cpu().numpy()
                logits = out.logits[0,:nt]
                p = F.softmax(logits, dim=-1)
                lp = F.log_softmax(logits, dim=-1)
                ent = -(p * lp).sum(-1).cpu().numpy()
                _ = ent / log_v
                _ = np.exp(ent)
            if len(embs) > 1:
                NearestNeighbors(n_neighbors=min(9,len(embs)), metric="cosine").fit(embs).kneighbors(embs)
            sync()
            pp_times.append((time.perf_counter()-t0)*1000)
        results["preproc_ms"] = round(float(np.median(pp_times)), 1)
        del lm

    elif model_name == "roberta":
        pp_times = []
        for text in texts:
            t0 = time.perf_counter()
            _ = tok(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            pp_times.append((time.perf_counter()-t0)*1000)
        results["preproc_ms"] = round(float(np.median(pp_times)), 1)

    print(json.dumps(results))


if __name__ == "__main__":
    profile_model(sys.argv[1])
'''

def run_model(name, label):
    print(f"  Profiling {label} ...")
    result = subprocess.run(
        [PYTHON, "-u", "-c", WORKER_CODE, name, CSV_PATH],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1"}
    )
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-500:]}")
        return None
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    print(f"    No JSON output found. stdout: {result.stdout[-300:]}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Fair computational profiling")
    parser.add_argument("--data", required=True, help="Path to dataset CSV for preprocessing timing")
    args = parser.parse_args()

    global CSV_PATH
    CSV_PATH = os.path.abspath(args.data)

    print("=" * 75)
    print("  FAIR COMPUTATIONAL PROFILING (isolated processes)")
    print("=" * 75)

    models = [
        ("m1", "Model 1 (Simple GCN)"),
        ("m2", "Model 2 (Enhanced GAT)"),
        ("m3", "Model 3 (Perplexity-aware GAT)"),
        ("roberta", "RoBERTa fine-tuned"),
    ]

    results = {}
    for key, label in models:
        r = run_model(key, label)
        if r:
            results[key] = r
            print(f"    Done: {r['params']:,} params | "
                  f"GPU Inf: {r['mem_inference_mb']:.1f} MB | "
                  f"GPU Train: {r['mem_training_mb']:.1f} MB | "
                  f"Inf: {r['inference_ms']:.1f} ms | "
                  f"Preproc: {r['preproc_ms']:.1f} ms")

    if len(results) < 4:
        print("\nSome models failed. Printing what we have.\n")

    n_train = 2800
    bpe = n_train // 16

    print("\n\n" + "=" * 90)
    print("  COMPUTATIONAL PROFILING RESULTS")
    print("=" * 90)
    fmt = "  {:<35} {:>14} {:>14} {:>14} {:>14}"
    print(fmt.format("Metric", "Model 1 (GCN)", "Model 2 (GAT)", "Model 3 (PPL)", "RoBERTa FT"))
    print("  " + "-" * 86)

    def g(k, f): return results.get(k, {}).get(f, "—")

    print(fmt.format("Trainable parameters",
        f"~{g('m1','params')//1000}K" if g('m1','params')!="—" else "—",
        f"~{g('m2','params')//1000}K" if g('m2','params')!="—" else "—",
        f"~{g('m3','params')//1000}K" if g('m3','params')!="—" else "—",
        f"~{g('roberta','params')//1_000_000}M" if g('roberta','params')!="—" else "—"))

    print(fmt.format("Model size on disk",
        f"{g('m1','disk_mb')} MB", f"{g('m2','disk_mb')} MB",
        f"{g('m3','disk_mb')} MB", f"{g('roberta','disk_mb')} MB"))

    def fmt_mb(key, field):
        v = g(key, field)
        return f"{v:.1f} MB" if v != "—" else "—"

    print(fmt.format("Peak GPU mem (inference)",
        fmt_mb('m1','mem_inference_mb'), fmt_mb('m2','mem_inference_mb'),
        fmt_mb('m3','mem_inference_mb'), fmt_mb('roberta','mem_inference_mb')))

    print(fmt.format("Peak GPU mem (training)",
        fmt_mb('m1','mem_training_mb'), fmt_mb('m2','mem_training_mb'),
        fmt_mb('m3','mem_training_mb'), fmt_mb('roberta','mem_training_mb')))

    for key in ["m1","m2","m3","roberta"]:
        r = results.get(key, {})
        if "preproc_ms" in r and "inference_ms" in r:
            r["e2e_ms"] = round(r["preproc_ms"] + r["inference_ms"], 1)

    epochs = {"m1": 10, "m2": 20, "m3": 30, "roberta": 10}
    for key in results:
        r = results[key]
        ep = epochs.get(key, 30)
        r["train_total_min"] = round(r["train_step_ms"] * bpe * ep / 60000, 1)
        if key != "roberta":
            r["train_total_min"] = round((r["train_step_ms"] * bpe + r["preproc_ms"] * n_train) * ep / 60000, 1)

    print(fmt.format("Training time (total)",
        f"{g('m1','train_total_min')} min" if g('m1','train_total_min')!="—" else "—",
        f"{g('m2','train_total_min')} min" if g('m2','train_total_min')!="—" else "—",
        f"{g('m3','train_total_min')} min" if g('m3','train_total_min')!="—" else "—",
        f"{g('roberta','train_total_min')} min" if g('roberta','train_total_min')!="—" else "—"))

    print(fmt.format("Preprocessing (per sample)",
        f"{g('m1','preproc_ms')} ms", f"{g('m2','preproc_ms')} ms",
        f"{g('m3','preproc_ms')} ms", f"{g('roberta','preproc_ms')} ms"))

    print(fmt.format("Classifier inference (per sample)",
        f"{g('m1','inference_ms')} ms", f"{g('m2','inference_ms')} ms",
        f"{g('m3','inference_ms')} ms", f"{g('roberta','inference_ms')} ms"))

    print(fmt.format("End-to-end latency (per sample)",
        f"{g('m1','e2e_ms')} ms" if g('m1','e2e_ms')!="—" else "—",
        f"{g('m2','e2e_ms')} ms" if g('m2','e2e_ms')!="—" else "—",
        f"{g('m3','e2e_ms')} ms" if g('m3','e2e_ms')!="—" else "—",
        f"{g('roberta','e2e_ms')} ms" if g('roberta','e2e_ms')!="—" else "—"))

    print("=" * 90)

    print("\n  [LaTeX rows]")
    for key, label in [("m1","Model 1 (Simple GCN)"),("m2","Model 2 (Enhanced GAT)"),
                        ("m3","Model 3 (Perplexity-aware GAT)"),("roberta","RoBERTa (fine-tuned)")]:
        r = results.get(key, {})
        if not r: continue
        p = f"~{r['params']//1000}K" if r['params'] < 1_000_000 else f"~{r['params']//1_000_000}M"
        print(f"  {label} & {p} & {r['disk_mb']} MB & "
              f"{r['mem_inference_mb']:.1f} MB & {r['mem_training_mb']:.1f} MB & "
              f"{r.get('train_total_min','—')} min & {r['preproc_ms']} ms & "
              f"{r['inference_ms']} ms & {r.get('e2e_ms','—')} ms \\\\")


if __name__ == "__main__":
    main()
