"""
Baseline comparison for Experiment I (Graph Reachability).

Implements the graph baselines from the paper's comparison table and evaluates
them on the SAME data distribution and difficulty grid as the depth-recurrent
model:

  1. Fixed-Depth GNN          - L distinct message-passing layers (adjacency).
  2. Graph Transformer         - L distinct adjacency-masked Transformer layers
                                 (the paper's "Fixed-depth Transformer" row).
  3. Weight-Tied Transformer   - ONE shared Transformer layer applied T steps,
                                 standard residual update, timestep embedding,
                                 WITHOUT our stability recipe (no identity gate,
                                 no LayerScale, no silent-thinking-specific gate)
                                 and WITHOUT ACT halting. The genuine Universal
                                 Transformer (ACT halting + ponder cost) is
                                 graph/ut_act.py. In the paper this is the
                                 "w/o LayerScale + gate" ablation row of
                                 tab:baselines, not a standalone baseline.

Fixed-depth models do a single forward pass; the weight-tied model is trained
with a randomised step count and final-step-only loss (identical protocol to
ours), then evaluated with a sufficient step budget.

Reuses the dataset and constants from graph_experiment.py so the comparison is
apples-to-apples. Writes per-method results to baseline_results.json.
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_experiment as G  # dataset, vocab, constants

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

VOCAB_SIZE = G.VOCAB_SIZE
PAD_IDX = G.PAD_IDX


# ----------------------------------------------------------------------
# Shared perception interface + readout (matches the main model)
# ----------------------------------------------------------------------
class GraphIO(nn.Module):
    """Node embedding + source/target role embeddings + pairwise readout."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.source_role_emb = nn.Parameter(torch.randn(d_model) * 0.02)
        self.target_role_emb = nn.Parameter(torch.randn(d_model) * 0.02)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def embed(self, node_ids, src_idx, tgt_idx):
        B, N = node_ids.shape
        h = self.token_emb(node_ids)
        src_oh = F.one_hot(src_idx, num_classes=N).float().unsqueeze(-1)
        tgt_oh = F.one_hot(tgt_idx, num_classes=N).float().unsqueeze(-1)
        h = h + src_oh * self.source_role_emb + tgt_oh * self.target_role_emb
        return self.emb_drop(self.emb_norm(h))

    def readout(self, h, src_idx, tgt_idx):
        B = h.size(0)
        bi = torch.arange(B, device=h.device)
        cls = torch.cat([h[bi, src_idx], h[bi, tgt_idx]], dim=-1)
        return self.head(cls).squeeze(-1)


def _row_normalised(adj_mask):
    """(B,N,N) bool -> row-normalised float adjacency for mean aggregation."""
    A = adj_mask.float()
    deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return A / deg


def _additive_mask(adj_mask, nhead):
    """(B,N,N) bool -> (B*nhead,N,N) additive float mask for attention."""
    B, N, _ = adj_mask.shape
    m = torch.zeros(B, N, N, device=adj_mask.device)
    m.masked_fill_(~adj_mask, float("-inf"))
    return m.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(B * nhead, N, N)


# ----------------------------------------------------------------------
# Baseline 1: Fixed-Depth GNN (L distinct message-passing layers)
# ----------------------------------------------------------------------
class GNNLayer(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.msg = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, A_norm):
        agg = torch.bmm(A_norm, self.msg(self.norm1(h)))
        h = h + self.drop(F.gelu(agg))
        h = h + self.drop(self.ff(self.norm2(h)))
        return h


class FixedDepthGNN(nn.Module):
    def __init__(self, d=128, depth=6, dropout=0.1):
        super().__init__()
        self.io = GraphIO(d, dropout)
        self.layers = nn.ModuleList([GNNLayer(d, dropout) for _ in range(depth)])
        self.recurrent = False

    def forward(self, node_ids, adj_mask, src_idx, tgt_idx, n_steps=None):
        h = self.io.embed(node_ids, src_idx, tgt_idx)
        A = _row_normalised(adj_mask)
        for layer in self.layers:
            h = layer(h, A)
        return self.io.readout(h, src_idx, tgt_idx)


# ----------------------------------------------------------------------
# Baseline 2: Graph Transformer (L distinct adjacency-masked layers)
# ----------------------------------------------------------------------
class GraphTransformerFixed(nn.Module):
    def __init__(self, d=128, nhead=4, dim_ff=256, depth=6, dropout=0.1):
        super().__init__()
        self.io = GraphIO(d, dropout)
        self.nhead = nhead
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nhead, dim_ff, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.recurrent = False

    def forward(self, node_ids, adj_mask, src_idx, tgt_idx, n_steps=None):
        h = self.io.embed(node_ids, src_idx, tgt_idx)
        m = _additive_mask(adj_mask, self.nhead)
        for layer in self.layers:
            h = layer(h, src_mask=m)
        return self.io.readout(h, src_idx, tgt_idx)


# ----------------------------------------------------------------------
# Baseline 3: Weight-Tied Transformer (one shared layer, recurrent, no recipe;
# the genuine Universal Transformer with ACT halting is graph/ut_act.py)
# ----------------------------------------------------------------------
class WeightTiedTransformer(nn.Module):
    """Weight-tied Transformer block applied T times with a timestep embedding
    and a standard residual update -- i.e., our core WITHOUT the stability recipe
    (no identity-biased gate, no LayerScale)."""

    def __init__(self, d=128, nhead=4, dim_ff=256, dropout=0.1, max_steps=20):
        super().__init__()
        self.io = GraphIO(d, dropout)
        self.nhead = nhead
        self.layer = nn.TransformerEncoderLayer(d, nhead, dim_ff, dropout,
                                                batch_first=True, norm_first=True)
        self.depth_emb = nn.Embedding(max_steps, d)
        self.recurrent = True

    def forward(self, node_ids, adj_mask, src_idx, tgt_idx, n_steps=3):
        h = self.io.embed(node_ids, src_idx, tgt_idx)
        m = _additive_mask(adj_mask, self.nhead)
        for t in range(n_steps):
            e = self.depth_emb(torch.tensor(t, device=h.device))
            h = self.layer(h + e, src_mask=m)
        return self.io.readout(h, src_idx, tgt_idx)


# ----------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------
def train(model, loader, epochs, lr, step_range, max_think):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    for ep in range(1, epochs + 1):
        model.train()
        for node_ids, adj, src, tgt, labels in loader:
            node_ids, adj = node_ids.to(DEVICE), adj.to(DEVICE)
            src, tgt, labels = src.to(DEVICE), tgt.to(DEVICE), labels.to(DEVICE)
            if model.recurrent:
                n = random.randint(*step_range)
            else:
                n = None
            logits = model(node_ids, adj, src, tgt, n_steps=n)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()


@torch.no_grad()
def eval_acc(model, loader, n_steps):
    model.eval()
    correct = total = 0
    for node_ids, adj, src, tgt, labels in loader:
        node_ids, adj = node_ids.to(DEVICE), adj.to(DEVICE)
        src, tgt, labels = src.to(DEVICE), tgt.to(DEVICE), labels.to(DEVICE)
        logits = model(node_ids, adj, src, tgt, n_steps=n_steps)
        correct += ((logits > 0).float() == labels).sum().item()
        total += labels.numel()
    return correct / total if total else 0.0


def evaluate_grid(model, test_loaders, test_hops, max_think):
    """Return {hops: best_acc}. Adaptive models get a sufficient step budget."""
    out = {}
    for hops in test_hops:
        if model.recurrent:
            # sufficient steps: at least `hops`, capped at max_think; take best
            budgets = sorted(set([min(max_think, max(hops, 8)), max_think]))
            out[hops] = max(eval_acc(model, test_loaders[hops], s) for s in budgets)
        else:
            out[hops] = eval_acc(model, test_loaders[hops], None)
    return out


def summarise(grid, id_hops):
    id_vals = [grid[h] for h in grid if h <= id_hops]
    id_acc = float(np.mean(id_vals)) if id_vals else float("nan")
    ood_reach = max([h for h in grid if grid[h] >= 0.90], default=0)
    return id_acc, ood_reach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    D, NHEAD, DIM_FF = 128, 4, 256
    EPOCHS, LR, BATCH = 40, 3e-4, 64
    TRAIN_SAMPLES, TEST_PER = 12_000, 500
    TRAIN_HOP_RANGE, TRAIN_NODES = (1, 5), (8, 16)
    TRAIN_STEP_RANGE, MAX_THINK = (5, 8), 20
    MAX_NODES = 20
    TEST_HOPS = [1, 2, 3, 4, 6, 8, 10, 12]
    ID_HOPS = 5  # training range max

    print("=" * 70)
    print(f"  Graph Reachability -- BASELINES   device={DEVICE}")
    print("=" * 70)

    print("[data] generating ...", flush=True)
    train_ds = G.GraphReachabilityDataset(TRAIN_SAMPLES, TRAIN_HOP_RANGE,
                                          TRAIN_NODES, MAX_NODES)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    test_loaders = {}
    for hops in TEST_HOPS:
        n_min = max(hops + 2, 10); n_max = min(max(hops + 6, 16), MAX_NODES)
        n_min = min(n_min, n_max)
        ds = G.GraphReachabilityDataset(TEST_PER, (hops, hops), (n_min, n_max), MAX_NODES)
        test_loaders[hops] = DataLoader(ds, batch_size=BATCH, shuffle=False)

    builders = {
        "Fixed-Depth GNN":        lambda: FixedDepthGNN(D, depth=6),
        "Graph Transformer":      lambda: GraphTransformerFixed(D, NHEAD, DIM_FF, depth=6),
        "Weight-Tied Transformer":  lambda: WeightTiedTransformer(D, NHEAD, DIM_FF, max_steps=MAX_THINK),
    }

    results = {}
    for name, build in builders.items():
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        model = build().to(DEVICE)
        nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        print(f"\n[{name}] params={nparam:,} recurrent={model.recurrent} -- training ...", flush=True)
        train(model, train_loader, EPOCHS, LR, TRAIN_STEP_RANGE, MAX_THINK)
        grid = evaluate_grid(model, test_loaders, TEST_HOPS, MAX_THINK)
        id_acc, ood_reach = summarise(grid, ID_HOPS)
        dt = time.time() - t0
        results[name] = {"params": nparam, "recurrent": model.recurrent,
                         "grid": grid, "id_acc": id_acc, "ood_reach": ood_reach,
                         "seconds": round(dt, 1)}
        print(f"  per-hop acc: " + "  ".join(f"{h}:{grid[h]:.2f}" for h in TEST_HOPS))
        print(f"  -> ID acc={id_acc:.3f}  OOD reach (>=90%)={ood_reach} hops  ({dt:.0f}s)", flush=True)

    suffix = "" if seed == SEED else f"_s{seed}"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"baseline_results{suffix}.json")
    with open(out, "w") as f:
        json.dump({"task": "graph", "seed": seed, "test_hops": TEST_HOPS,
                   "id_hops_max": ID_HOPS, "results": results}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
