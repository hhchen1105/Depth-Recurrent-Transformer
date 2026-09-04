"""
Baseline comparison for the two sequence tasks:
  - Experiment II  (nested-expr) : Nested Boolean Expression Evaluation
  - Experiment III (family-reason): Relational Composition in Unstructured Text

Both are binary classification over a token sequence with a [CLS] readout and
RoPE positional encoding, so they share one baseline harness.

Baselines (parameter budget comparable to our single-block core; trained on the
same data distribution and final-step-only protocol):

  * Fixed-Depth Transformer (RoPE) : L distinct pre-norm RoPE encoder layers,
    a single forward pass. L is set to the maximum training difficulty so the
    model has just enough architectural depth for in-distribution instances but
    cannot add depth at test time.
  * Weight-Tied Transformer (RoPE) : ONE shared pre-norm RoPE layer applied T
    times with a timestep embedding and a standard residual update -- i.e. our
    recurrent core WITHOUT the stability recipe (no identity-biased gate, no
    LayerScale) AND without ACT halting. Trained with randomised T and
    final-step loss; evaluated with a sufficient step budget. (The genuine
    Universal Transformer, with ACT halting + ponder cost, is ut_act_seq.py.)
    In the paper this is the "w/o LayerScale + gate" ablation row of
    tab:baselines, not a standalone baseline.

Usage:
    python seq_baselines.py logic    # nested boolean expressions
    python seq_baselines.py family   # relational text
    python seq_baselines.py logic --full --seed 43   # extra seed

Writes <task>/baseline_results.json (default seed 42), or
<task>/baseline_results_s<seed>.json for any other --seed.

NOTE ON BUDGET: to keep the four-way study tractable on a laptop GPU (MPS), the
baselines use the reduced but fixed budget below (samples/epochs), identical
across every baseline so the comparison stays fair. The "ours" row in the paper
table is taken from the full-budget runs already saved in each experiment dir.
"""

import os
import sys
import json
import time
import random
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

HERE = os.path.dirname(os.path.abspath(__file__))

# -- Per-task configuration --------------------------------------------
TASKS = {
    "logic": dict(
        dirname="nested-expr", module="logic_experiment",
        dataset="BooleanExpressionDataset",
        d_model=256, nhead=8, dim_ff=1024,
        train_depths=[1, 2, 3, 4, 5, 6, 7, 8],
        test_depths=[2, 4, 6, 8, 10, 12, 14],
        id_depth_max=8, fixed_depth=8,
        train_step_range=(4, 16), max_think=24,
        max_len=128,
        full_samples=64_000, full_epochs=30,   # matches logic_experiment.py
    ),
    "family": dict(
        dirname="family-reason", module="family_experiment",
        dataset="FamilyReasoningDataset",
        d_model=256, nhead=8, dim_ff=1024,
        train_depths=[2, 3, 4, 5],
        test_depths=[2, 3, 4, 5, 6, 7, 8, 9],
        id_depth_max=5, fixed_depth=5,
        train_step_range=(1, 12), max_think=20,
        max_len=128,
        full_samples=60_000, full_epochs=40,   # matches family_experiment.py
    ),
}

# Reduced-but-fixed training budget (same for every baseline, both tasks).
# Sized so each baseline trains to convergence in a few minutes on MPS while the
# four-way comparison stays internally fair.
SAMPLES = 16_000
EPOCHS = 15
BATCH = 128
LR = 3e-4


def load_task_module(cfg):
    sys.path.insert(0, os.path.join(HERE, cfg["dirname"]))
    return importlib.import_module(cfg["module"])


# ----------------------------------------------------------------------
# Plain pre-norm RoPE encoder layer (no LayerScale, no gate) -- reuses the
# task module's RotaryEmbedding + apply_rotary_pos_emb for identical RoPE.
# ----------------------------------------------------------------------
class RoPEEncoderLayer(nn.Module):
    def __init__(self, mod, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.apply_rope = mod.apply_rotary_pos_emb
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _attn(self, x, cos, sin, pad_mask):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)
        m = None
        if pad_mask is not None:
            m = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
            m.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=m,
            dropout_p=self.drop.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o(out)

    def forward(self, h, cos, sin, pad_mask=None):
        h = h + self.drop(self._attn(self.norm1(h), cos, sin, pad_mask))
        h = h + self.ffn(self.norm2(h))
        return h


class SeqIO(nn.Module):
    """Token embedding + RoPE + CLS readout (matches the main models)."""

    def __init__(self, mod, vocab_size, d_model, nhead, max_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=mod.PAD_IDX)
        self.rope = mod.RotaryEmbedding(d_model // nhead, max_len=max_len)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

    def embed(self, token_ids):
        h = self.token_emb(token_ids)
        return self.emb_drop(self.emb_norm(h))

    def readout(self, h):
        return self.head(h[:, 0]).squeeze(-1)


class FixedDepthTransformer(nn.Module):
    def __init__(self, mod, vocab_size, d, nhead, dim_ff, depth, max_len):
        super().__init__()
        self.io = SeqIO(mod, vocab_size, d, nhead, max_len)
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(mod, d, nhead, dim_ff) for _ in range(depth)])
        self.recurrent = False

    def forward(self, token_ids, pad_mask=None, n_steps=None):
        h = self.io.embed(token_ids)
        cos, sin = self.io.rope(token_ids.size(1))
        for layer in self.layers:
            h = layer(h, cos, sin, pad_mask)
        return self.io.readout(h)


class WeightTiedTransformer(nn.Module):
    def __init__(self, mod, vocab_size, d, nhead, dim_ff, max_steps, max_len):
        super().__init__()
        self.io = SeqIO(mod, vocab_size, d, nhead, max_len)
        self.layer = RoPEEncoderLayer(mod, d, nhead, dim_ff)
        self.depth_emb = nn.Embedding(max_steps, d)
        self.recurrent = True

    def forward(self, token_ids, pad_mask=None, n_steps=3):
        h = self.io.embed(token_ids)
        cos, sin = self.io.rope(token_ids.size(1))
        for t in range(n_steps):
            e = self.depth_emb(torch.tensor(t, device=h.device))
            h = self.layer(h + e, cos, sin, pad_mask)
        return self.io.readout(h)


# ----------------------------------------------------------------------
def make_loader(mod, cfg, depths, n_samples, shuffle):
    ds_cls = getattr(mod, cfg["dataset"])
    ds = ds_cls(n_samples=n_samples, depths=depths, max_len=cfg["max_len"])
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, drop_last=shuffle)


def train(model, loader, epochs, step_range, log_prefix=""):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    for ep in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for token_ids, pad_mask, labels in loader:
            token_ids, pad_mask, labels = (token_ids.to(DEVICE), pad_mask.to(DEVICE),
                                           labels.to(DEVICE))
            n = random.randint(*step_range) if model.recurrent else None
            logits = model(token_ids, pad_mask, n_steps=n)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            correct += ((logits > 0).float() == labels).sum().item(); total += labels.numel()
        if ep % 3 == 0 or ep == 1:
            print(f"    {log_prefix} epoch {ep:2d}/{epochs}  train_acc={correct/total:.3f}", flush=True)


@torch.no_grad()
def eval_acc(model, loader, n_steps):
    model.eval()
    correct = total = 0
    for token_ids, pad_mask, labels in loader:
        token_ids, pad_mask, labels = (token_ids.to(DEVICE), pad_mask.to(DEVICE),
                                       labels.to(DEVICE))
        logits = model(token_ids, pad_mask, n_steps=n_steps)
        correct += ((logits > 0).float() == labels).sum().item(); total += labels.numel()
    return correct / total if total else 0.0


def evaluate_grid(model, test_loaders, cfg):
    out = {}
    for depth, loader in test_loaders.items():
        # adaptive models get a sufficient step budget (the max trained depth)
        n = cfg["max_think"] if model.recurrent else None
        out[depth] = eval_acc(model, loader, n)
    return out


def summarise(grid, id_depth_max):
    id_vals = [grid[d] for d in grid if d <= id_depth_max]
    id_acc = float(np.mean(id_vals)) if id_vals else float("nan")
    ood_reach = max([d for d in grid if grid[d] >= 0.90], default=0)
    return id_acc, ood_reach


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Sequence-task baselines (logic / family).")
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--full", action="store_true",
                    help="use the original per-task full budget (logic 64k/30, family 60k/40)")
    ap.add_argument("--samples", type=int, default=None, help="override training samples")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random seed; non-default seeds write baseline_results_s<seed>.json")
    args = ap.parse_args()

    task = args.task
    cfg = TASKS[task]
    seed = args.seed
    global SAMPLES, EPOCHS
    if args.full:
        SAMPLES, EPOCHS = cfg["full_samples"], cfg["full_epochs"]
    if args.samples is not None:
        SAMPLES = args.samples
    if args.epochs is not None:
        EPOCHS = args.epochs

    mod = load_task_module(cfg)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print("=" * 70)
    print(f"  {task.upper()} -- BASELINES   device={DEVICE}  seed={seed}  budget={SAMPLES} samples / {EPOCHS} ep")
    print("=" * 70)

    print("[data] generating train + per-depth test sets ...", flush=True)
    train_loader = make_loader(mod, cfg, cfg["train_depths"], SAMPLES, True)
    test_loaders = {}
    for d in cfg["test_depths"]:
        test_loaders[d] = make_loader(mod, cfg, [d], 600, False)

    vocab = mod.VOCAB_SIZE
    builders = {
        "Fixed-Depth Transformer": lambda: FixedDepthTransformer(
            mod, vocab, cfg["d_model"], cfg["nhead"], cfg["dim_ff"],
            cfg["fixed_depth"], cfg["max_len"]),
        "Weight-Tied Transformer": lambda: WeightTiedTransformer(
            mod, vocab, cfg["d_model"], cfg["nhead"], cfg["dim_ff"],
            cfg["max_think"], cfg["max_len"]),
    }

    results = {}
    for name, build in builders.items():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = build().to(DEVICE)
        nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        print(f"\n[{name}] params={nparam:,} recurrent={model.recurrent} -- training ...", flush=True)
        train(model, train_loader, EPOCHS, cfg["train_step_range"], log_prefix=name[:12])
        grid = evaluate_grid(model, test_loaders, cfg)
        id_acc, ood_reach = summarise(grid, cfg["id_depth_max"])
        dt = time.time() - t0
        results[name] = {"params": nparam, "recurrent": model.recurrent,
                         "grid": {str(k): v for k, v in grid.items()},
                         "id_acc": id_acc, "ood_reach": ood_reach, "seconds": round(dt, 1)}
        print(f"  per-depth acc: " + "  ".join(f"{d}:{grid[d]:.2f}" for d in cfg["test_depths"]))
        print(f"  -> ID acc={id_acc:.3f}  OOD reach (>=90%)={ood_reach}  ({dt:.0f}s)", flush=True)

    suffix = "" if seed == SEED else f"_s{seed}"
    out = os.path.join(HERE, cfg["dirname"], f"baseline_results{suffix}.json")
    with open(out, "w") as f:
        json.dump({"task": task, "seed": seed, "test_depths": cfg["test_depths"],
                   "id_depth_max": cfg["id_depth_max"], "budget": {"samples": SAMPLES, "epochs": EPOCHS},
                   "results": results}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
