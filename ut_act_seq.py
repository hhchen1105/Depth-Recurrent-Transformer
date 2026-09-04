"""
"Genuine" Universal Transformer with Adaptive Computation Time (ACT),
following Dehghani et al. (2019) / Graves (2016), for the two SEQUENCE tasks
(nested boolean logic, relational family text).

This is the sequence-task counterpart of graph/ut_act.py. The existing
seq_baselines.py::WeightTiedTransformer only borrows weight sharing + a depth
embedding: it runs a fixed, externally-chosen number of steps for the whole
batch and has no halting mechanism or ponder cost. This script implements the
actual ACT halting mechanism on the same RoPE-based encoder architecture that
seq_baselines.py uses (SeqIO + RoPEEncoderLayer), so the comparison isolates
the effect of ACT itself:

  - each token position predicts its own halting probability p^t at every step
    from its current hidden state;
  - a position halts once its cumulative halting probability crosses 1-eps;
  - the position's final representation is the probability-weighted mean of the
    states it passed through, using the ACT "remainder" for the last (halting)
    step so the weights sum to exactly 1 (Graves 2016; UT paper Sec. 2.2);
  - a ponder cost, mean over real (non-pad) positions of (N_i + R_i), scaled by
    TIME_PENALTY, is added to the task loss;
  - classification reads the CLS position (index 0) of the weighted state.

Budget / hyper-parameters follow seq_baselines.py exactly (--full = logic
64k/30, family 60k/40), so the ACT row is directly comparable to the
Weight-Tied Transformer row in the paper table.

Usage (run from the repo root, next to seq_baselines.py):
    python ut_act_seq.py logic  --full --seeds 42 43 44
    python ut_act_seq.py family --full --seeds 42 43 44

Writes ut_act_seq_<task>_results.json + ut_act_seq_<task>_checkpoint_seed<seed>.pt
here, and <task_dir>/act_results.json (picked up by make_baseline_table.py).
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

HERE = os.path.dirname(os.path.abspath(__file__))
# seq_baselines.py sits either alongside this file or one directory up.
for _p in (HERE, os.path.dirname(HERE)):
    if os.path.exists(os.path.join(_p, "seq_baselines.py")):
        sys.path.insert(0, _p)
        break

import seq_baselines as SB
from seq_baselines import DEVICE, RoPEEncoderLayer, SeqIO

MAX_STEPS_CAP = None     # per-task: set from cfg["max_think"]
EPS = 0.01               # ACT halting threshold slack (Graves 2016 default)
TIME_PENALTY = 0.01      # tau: weight on the ponder-cost term (matches ut_act.py)
BATCH = 128
LR = 3e-4


class ACTSeqUniversalTransformer(nn.Module):
    """RoPE Universal Transformer with per-position Adaptive Computation Time."""

    def __init__(self, mod, vocab_size, d, nhead, dim_ff, max_steps, max_len,
                 dropout=0.1, eps=EPS):
        super().__init__()
        self.io = SeqIO(mod, vocab_size, d, nhead, max_len, dropout)
        self.layer = RoPEEncoderLayer(mod, d, nhead, dim_ff, dropout)
        self.depth_emb = nn.Embedding(max_steps, d)
        self.halt = nn.Linear(d, 1)
        # Standard ACT init: a slight negative bias so the model does not halt
        # immediately at step 1 before it has learned anything useful.
        nn.init.constant_(self.halt.bias, -1.0)
        self.max_steps = max_steps
        self.eps = eps
        self.recurrent = True

    def forward(self, token_ids, pad_mask=None, n_steps=None):
        """n_steps is accepted for interface compatibility but ignored: ACT
        chooses per-position depth on its own, up to self.max_steps."""
        h = self.io.embed(token_ids)
        cos, sin = self.io.rope(token_ids.size(1))
        B, L, D = h.shape
        dev = h.device
        if pad_mask is None:
            pad_mask = torch.zeros(B, L, dtype=torch.bool, device=dev)
        real = (~pad_mask).float()                            # (B,L)

        halting_cum = torch.zeros(B, L, device=dev)
        remainders = torch.zeros(B, L, device=dev)
        n_updates = torch.zeros(B, L, device=dev)
        weighted_state = torch.zeros_like(h)
        still_running = ~pad_mask                             # pad halts at t=0

        for t in range(self.max_steps):
            if not still_running.any():
                break
            p = torch.sigmoid(self.halt(h)).squeeze(-1)       # (B,L)
            new_cum = halting_cum + p
            halt_now = still_running & (new_cum >= 1.0 - self.eps)
            keep_running = still_running & ~halt_now

            weight = torch.zeros(B, L, device=dev)
            weight = torch.where(halt_now, 1.0 - halting_cum, weight)
            weight = torch.where(keep_running, p, weight)

            halting_cum = torch.where(still_running, new_cum, halting_cum)
            remainders = torch.where(halt_now, 1.0 - (halting_cum - p), remainders)
            n_updates = n_updates + still_running.float()

            e = self.depth_emb(torch.tensor(t, device=dev))
            h_new = self.layer(h + e, cos, sin, pad_mask)

            weighted_state = weighted_state + weight.unsqueeze(-1) * h_new
            h = torch.where(still_running.unsqueeze(-1), h_new, h)
            still_running = keep_running

        denom = real.sum().clamp(min=1.0)
        ponder_cost = ((n_updates + remainders) * real).sum() / denom
        logits = self.io.readout(weighted_state)
        return logits, ponder_cost


@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = total = 0
    for token_ids, pad_mask, labels in loader:
        token_ids, pad_mask, labels = (token_ids.to(DEVICE), pad_mask.to(DEVICE),
                                       labels.to(DEVICE))
        logits, _ = model(token_ids, pad_mask)
        correct += ((logits > 0).float() == labels).sum().item()
        total += labels.numel()
    return correct / total if total else 0.0


@torch.no_grad()
def avg_ponder(model, loader):
    """Diagnostic: mean ACT steps used at the CLS position (mirrors the UT
    paper's Fig. 3 ponder-time analysis)."""
    model.eval()
    total_steps, n = 0.0, 0
    for token_ids, pad_mask, labels in loader:
        token_ids, pad_mask = token_ids.to(DEVICE), pad_mask.to(DEVICE)
        h = model.io.embed(token_ids)
        cos, sin = model.io.rope(token_ids.size(1))
        B, L, _ = h.shape
        halting_cum = torch.zeros(B, L, device=h.device)
        n_updates = torch.zeros(B, L, device=h.device)
        still_running = ~pad_mask
        for t in range(model.max_steps):
            if not still_running.any():
                break
            p = torch.sigmoid(model.halt(h)).squeeze(-1)
            new_cum = halting_cum + p
            halt_now = still_running & (new_cum >= 1.0 - model.eps)
            keep_running = still_running & ~halt_now
            halting_cum = torch.where(still_running, new_cum, halting_cum)
            n_updates = n_updates + still_running.float()
            e = model.depth_emb(torch.tensor(t, device=h.device))
            h_new = model.layer(h + e, cos, sin, pad_mask)
            h = torch.where(still_running.unsqueeze(-1), h_new, h)
            still_running = keep_running
        total_steps += n_updates[:, 0].sum().item()   # CLS position
        n += B
    return total_steps / n if n else 0.0


def train(model, loader, epochs, log_prefix=""):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    for ep in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for token_ids, pad_mask, labels in loader:
            token_ids, pad_mask, labels = (token_ids.to(DEVICE), pad_mask.to(DEVICE),
                                           labels.to(DEVICE))
            logits, ponder_cost = model(token_ids, pad_mask)
            loss = (F.binary_cross_entropy_with_logits(logits, labels)
                    + TIME_PENALTY * ponder_cost)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            correct += ((logits > 0).float() == labels).sum().item()
            total += labels.numel()
        if ep % 3 == 0 or ep == 1:
            print(f"    {log_prefix} epoch {ep:2d}/{epochs}  train_acc={correct/total:.3f}",
                  flush=True)


def run_seed(task, cfg, mod, seed, samples, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    global BATCH
    SB.BATCH = BATCH  # make_loader uses SB.BATCH

    train_loader = SB.make_loader(mod, cfg, cfg["train_depths"], samples, True)
    test_loaders = {d: SB.make_loader(mod, cfg, [d], 600, False)
                    for d in cfg["test_depths"]}

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = ACTSeqUniversalTransformer(
        mod, mod.VOCAB_SIZE, cfg["d_model"], cfg["nhead"], cfg["dim_ff"],
        cfg["max_think"], cfg["max_len"]).to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t0 = time.time()
    print(f"\n[ACT-UT {task} seed {seed}] params={nparam:,} "
          f"max_steps={cfg['max_think']} budget={samples}/{epochs} -- training ...",
          flush=True)
    train(model, train_loader, epochs, log_prefix=f"s{seed}")
    dt = time.time() - t0

    grid = {d: eval_acc(model, test_loaders[d]) for d in cfg["test_depths"]}
    ponder = {d: avg_ponder(model, test_loaders[d]) for d in cfg["test_depths"]}
    id_acc = float(np.mean([grid[d] for d in cfg["test_depths"]
                            if d <= cfg["id_depth_max"]]))

    torch.save(model.state_dict(),
               os.path.join(HERE, f"ut_act_seq_{task}_checkpoint_seed{seed}.pt"))
    print(f"  per-depth acc:    " + "  ".join(f"{d}:{grid[d]:.2f}" for d in cfg["test_depths"]))
    print(f"  per-depth ponder: " + "  ".join(f"{d}:{ponder[d]:.1f}" for d in cfg["test_depths"]))
    print(f"  -> ID acc={id_acc:.3f}  ({dt:.0f}s)", flush=True)
    return {"params": nparam, "seconds": round(dt, 1),
            "grid": {str(k): v for k, v in grid.items()},
            "ponder": {str(k): v for k, v in ponder.items()},
            "id_acc": id_acc}


def main():
    ap = argparse.ArgumentParser(description="ACT Universal Transformer, sequence tasks.")
    ap.add_argument("task", choices=list(SB.TASKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--full", action="store_true",
                    help="use the full per-task budget (logic 64k/30, family 60k/40)")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = SB.TASKS[args.task]
    mod = SB.load_task_module(cfg)

    samples = SB.SAMPLES
    epochs = SB.EPOCHS
    if args.full:
        samples, epochs = cfg["full_samples"], cfg["full_epochs"]
    if args.samples is not None:
        samples = args.samples
    if args.epochs is not None:
        epochs = args.epochs

    print("=" * 70)
    print(f"  {args.task.upper()} -- ACT UNIVERSAL TRANSFORMER   device={DEVICE}  "
          f"seeds={args.seeds}  budget={samples}/{epochs}  "
          f"eps={EPS}  time_penalty={TIME_PENALTY}")
    print("=" * 70)

    out_path = os.path.join(HERE, f"ut_act_seq_{args.task}_results.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    for seed in args.seeds:
        print("=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)
        results[str(seed)] = run_seed(args.task, cfg, mod, seed, samples, epochs)
        for p in (out_path,
                  os.path.join(SB.HERE, cfg["dirname"], "act_results.json")):
            with open(p, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {os.path.join(SB.HERE, cfg['dirname'], 'act_results.json')} "
          f"(read by make_baseline_table.py)")


if __name__ == "__main__":
    main()
