"""
"Genuine" Universal Transformer with Adaptive Computation Time (ACT),
following Dehghani et al. (2019) / Graves (2016), for the graph reachability
task.

The existing `baselines.py::UniversalTransformer` only borrows weight
sharing + a depth embedding from UT; it runs a fixed, externally-chosen
number of steps for the whole batch and has no halting mechanism or ponder
cost. This script implements the actual ACT halting mechanism:

  - each node (position) predicts its own halting probability p^t at every
    step from its current hidden state;
  - a node halts once its cumulative halting probability crosses 1-eps;
  - the node's final representation is the probability-weighted mean of the
    states it passed through, using the ACT "remainder" R for the last
    (halting) step so the weights sum to exactly 1 (Graves 2016, Eq. in
    Sec. 2; UT paper Sec. 2.2);
  - a ponder cost, sum_i (N_i + R_i) averaged over positions/batch and
    scaled by `TIME_PENALTY`, is added to the task loss.

This reuses the same GraphIO (embedding + pairwise readout) and the same
nn.TransformerEncoderLayer transition function as baselines.py's
UniversalTransformer, so the comparison isolates the effect of ACT itself.

Usage: python ut_act.py [--seeds 42 43 44]   (run from this graph/ directory)
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
import graph_experiment as G
from baselines import GraphIO, _additive_mask, DEVICE

D, NHEAD, DIM_FF = 128, 4, 256
EPOCHS, LR, BATCH = 40, 3e-4, 64
TRAIN_SAMPLES, TEST_PER = 12_000, 500
TRAIN_HOP_RANGE, TRAIN_NODES = (1, 5), (8, 16)
MAX_NODES = 20
MAX_STEPS = 20          # ACT step ceiling (matches MAX_THINK elsewhere)
EPS = 0.01               # ACT halting threshold slack (Graves 2016 default)
TIME_PENALTY = 0.01      # tau: weight on the ponder-cost term
TEST_HOPS = [1, 2, 3, 4, 6, 8, 10, 12]
ID_HOPS = 5

HERE = os.path.dirname(os.path.abspath(__file__))


class ACTUniversalTransformer(nn.Module):
    """UT with per-position Adaptive Computation Time halting."""

    def __init__(self, d=D, nhead=NHEAD, dim_ff=DIM_FF, dropout=0.1,
                 max_steps=MAX_STEPS, eps=EPS):
        super().__init__()
        self.io = GraphIO(d, dropout)
        self.nhead = nhead
        self.layer = nn.TransformerEncoderLayer(d, nhead, dim_ff, dropout,
                                                batch_first=True, norm_first=True)
        self.depth_emb = nn.Embedding(max_steps, d)
        self.halt = nn.Linear(d, 1)
        # Standard ACT init: start with a slight bias so the model doesn't
        # halt immediately at step 1 for every position before it has
        # learned anything useful.
        nn.init.constant_(self.halt.bias, -1.0)
        self.max_steps = max_steps
        self.eps = eps
        self.recurrent = True

    def forward(self, node_ids, adj_mask, src_idx, tgt_idx, n_steps=None):
        """n_steps is accepted for interface compatibility but ignored: ACT
        determines per-position depth on its own, up to self.max_steps."""
        B, N = node_ids.shape
        h = self.io.embed(node_ids, src_idx, tgt_idx)
        m = _additive_mask(adj_mask, self.nhead)

        halting_cum = torch.zeros(B, N, device=h.device)      # cumulative p
        remainders = torch.zeros(B, N, device=h.device)
        n_updates = torch.zeros(B, N, device=h.device)
        weighted_state = torch.zeros_like(h)
        still_running = torch.ones(B, N, dtype=torch.bool, device=h.device)

        for t in range(self.max_steps):
            if not still_running.any():
                break
            p = torch.sigmoid(self.halt(h)).squeeze(-1)          # (B,N)
            # positions that halt at this exact step
            new_cum = halting_cum + p
            halt_now = still_running & (new_cum >= 1.0 - self.eps)
            keep_running = still_running & ~halt_now

            # weight for this step: remainder for newly-halted, raw p for
            # positions that keep running; 0 for already-halted positions.
            weight = torch.zeros(B, N, device=h.device)
            weight = torch.where(halt_now, 1.0 - halting_cum, weight)
            weight = torch.where(keep_running, p, weight)

            halting_cum = torch.where(still_running, new_cum, halting_cum)
            remainders = torch.where(halt_now, 1.0 - (halting_cum - p), remainders)
            n_updates = n_updates + still_running.float()

            e = self.depth_emb(torch.tensor(t, device=h.device))
            h_new = self.layer(h + e, src_mask=m)

            weighted_state = weighted_state + weight.unsqueeze(-1) * h_new
            # only still-running positions get their state advanced; halted
            # positions keep their last computed h_new but no longer feed
            # into weighted_state after this step (weight=0 going forward).
            h = torch.where(still_running.unsqueeze(-1), h_new, h)
            still_running = keep_running

        ponder_cost = (n_updates + remainders).mean()
        logits = self.io.readout(weighted_state, src_idx, tgt_idx)
        return logits, ponder_cost


@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = total = 0
    for node_ids, adj, src, tgt, labels in loader:
        node_ids, adj = node_ids.to(DEVICE), adj.to(DEVICE)
        src, tgt, labels = src.to(DEVICE), tgt.to(DEVICE), labels.to(DEVICE)
        logits, _ = model(node_ids, adj, src, tgt)
        correct += ((logits > 0).float() == labels).sum().item()
        total += labels.numel()
    return correct / total if total else 0.0


@torch.no_grad()
def avg_ponder(model, loader):
    """Diagnostic: mean number of ACT steps used (sanity check, mirrors the
    original UT paper's Fig. 3 ponder-time analysis)."""
    model.eval()
    total_steps, n = 0.0, 0
    for node_ids, adj, src, tgt, labels in loader:
        node_ids, adj = node_ids.to(DEVICE), adj.to(DEVICE)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        B, N = node_ids.shape
        h = model.io.embed(node_ids, src, tgt)
        m = _additive_mask(adj, model.nhead)
        halting_cum = torch.zeros(B, N, device=h.device)
        n_updates = torch.zeros(B, N, device=h.device)
        still_running = torch.ones(B, N, dtype=torch.bool, device=h.device)
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
            h_new = model.layer(h + e, src_mask=m)
            h = torch.where(still_running.unsqueeze(-1), h_new, h)
            still_running = keep_running
        # average ponder time at the source/target nodes specifically
        bi = torch.arange(B, device=h.device)
        st = (n_updates[bi, src] + n_updates[bi, tgt]) / 2
        total_steps += st.sum().item()
        n += B
    return total_steps / n if n else 0.0


def train(model, loader, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    for ep in range(1, epochs + 1):
        model.train()
        for node_ids, adj, src, tgt, labels in loader:
            node_ids, adj = node_ids.to(DEVICE), adj.to(DEVICE)
            src, tgt, labels = src.to(DEVICE), tgt.to(DEVICE), labels.to(DEVICE)
            logits, ponder_cost = model(node_ids, adj, src, tgt)
            loss = F.binary_cross_entropy_with_logits(logits, labels) + TIME_PENALTY * ponder_cost
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()


def build_loaders():
    train_ds = G.GraphReachabilityDataset(TRAIN_SAMPLES, TRAIN_HOP_RANGE,
                                          TRAIN_NODES, MAX_NODES)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    test_loaders = {}
    for hops in TEST_HOPS:
        n_min = max(hops + 2, 10); n_max = min(max(hops + 6, 16), MAX_NODES)
        n_min = min(n_min, n_max)
        ds = G.GraphReachabilityDataset(TEST_PER, (hops, hops), (n_min, n_max), MAX_NODES)
        test_loaders[hops] = DataLoader(ds, batch_size=BATCH, shuffle=False)
    return train_loader, test_loaders


def run_seed(seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    train_loader, test_loaders = build_loaders()
    model = ACTUniversalTransformer().to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.time()
    train(model, train_loader, EPOCHS, LR)
    dt = time.time() - t0

    grid = {h: eval_acc(model, test_loaders[h]) for h in TEST_HOPS}
    ponder = {h: avg_ponder(model, test_loaders[h]) for h in TEST_HOPS}
    id_acc = float(np.mean([grid[h] for h in TEST_HOPS if h <= ID_HOPS]))

    torch.save(model.state_dict(), os.path.join(HERE, f"ut_act_checkpoint_seed{seed}.pt"))
    return {"params": nparam, "seconds": round(dt, 1), "grid": grid,
            "ponder": ponder, "id_acc": id_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    print(f"device={DEVICE}  seeds={args.seeds}  max_steps={MAX_STEPS}  "
          f"eps={EPS}  time_penalty={TIME_PENALTY}")
    results = {}
    for seed in args.seeds:
        print("=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)
        r = run_seed(seed)
        results[seed] = r
        print(f"  params={r['params']:,}  ({dt if (dt:=r['seconds']) else 0}s)")
        print(f"  ID acc={r['id_acc']:.3f}")
        print(f"  per-hop acc:   " + "  ".join(f"{h}:{r['grid'][h]:.2f}" for h in TEST_HOPS))
        print(f"  per-hop ponder:" + "  ".join(f"{h}:{r['ponder'][h]:.1f}" for h in TEST_HOPS))

    graph_dir = os.path.dirname(os.path.abspath(G.__file__))
    for p in (os.path.join(HERE, "ut_act_results.json"),
              os.path.join(graph_dir, "act_results.json")):
        with open(p, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nSaved -> {os.path.join(HERE, 'ut_act_results.json')}")
    print(f"Saved -> {os.path.join(graph_dir, 'act_results.json')} "
          f"(read by make_baseline_table.py)")


if __name__ == "__main__":
    main()
