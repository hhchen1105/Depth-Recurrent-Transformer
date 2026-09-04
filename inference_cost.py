"""
Measured inference cost of the depth-recurrent reasoning core (paper Fig. fig:inference-cost).

A single pre-norm Transformer block (d=256, h=8, d_ff=1024, L=128, plus the
identity-biased gate and depth embedding of our core) is applied T times. We
measure peak CUDA memory and per-query wall-clock latency as the step count T
and the batch size B grow, and overlay the analytical key-value-cache memory
that a token-generating backbone of depth N would need for the same T
(Section "Computational and Memory Complexity").

GPU job: run via run_inference_cost.sh, which measures and then plots. To only
redraw the figure from an existing inference_cost.json (no GPU needed), run
    python inference_cost.py --plot-only
"""

import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D_MODEL, NHEAD, DIM_FF, L = 256, 8, 1024, 128
T_LIST = [1, 2, 4, 8, 16, 32, 64]
B_LIST = [1, 32, 128, 512]
N_BACKBONE = 24          # representative reasoning-LLM backbone depth for the overlay
BYTES_PER = 2            # fp16 KV cache
REPS = 30


def _build_block():
    """Import torch lazily so --plot-only works on a CPU-only login node."""
    import torch
    import torch.nn as nn

    class ReasoningBlock(nn.Module):
        """One pre-norm Transformer block + depth embedding + identity-biased
        gate, matching the core used in the sequence-task experiments."""

        def __init__(self, d=D_MODEL, h=NHEAD, ff=DIM_FF, max_steps=128):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, h, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
            self.depth_emb = nn.Embedding(max_steps, d)
            self.gate = nn.Linear(2 * d, d)
            nn.init.constant_(self.gate.bias, -2.0)

        def forward(self, h, t):
            x = h + self.depth_emb(torch.tensor(t, device=h.device))
            xn = self.ln1(x)
            a, _ = self.attn(xn, xn, xn, need_weights=False)
            x = x + a
            x = x + self.ff(self.ln2(x))
            z = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
            return z * x + (1.0 - z) * h

    return ReasoningBlock()


def measure(block, B, T, device):
    import torch

    with torch.no_grad():
        h0 = torch.randn(B, L, D_MODEL, device=device)
        h = h0
        for t in range(min(T, 4)):          # warmup
            h = block(h, t)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            h = h0
            for t in range(T):
                h = block(h, t)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / REPS * 1e3
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return peak_mb, latency_ms


def kv_cache_mb(B, T, d=D_MODEL, N=N_BACKBONE):
    """Analytical KV-cache memory for generating T tokens after an L-token prompt
    through an N-layer backbone: 2 * B * (L + T) * N * d * bytes."""
    return BYTES_PER * 2 * B * (L + T) * N * d / 1e6


def make_figure(out):
    """Restricted to the serving-scale batch sizes (B >= 32); at B=1 the core's
    fixed workspace floor dwarfs the KV-cache so the memory comparison is not
    meaningful there, and B=1 latency is within a few percent of B=32 anyway
    (the GPU is not saturated). Left: the growing key-value-cache term a
    depth-N token backbone would carry, as a multiple of the recurrent core's
    own (T-independent) footprint -- the core is the flat y=1 line. Right:
    latency per query, linear in T."""
    t_list = out["T_list"]
    b_list = [B for B in out["B_list"] if B >= 32]
    # Font sizes target the final render: the figure is included at \columnwidth
    # (~3.5 in), i.e. scaled by ~0.49x, so ~18/15 pt here land near the 8/7 pt
    # the heatmap figures render at -- the paper's camera-ready floor. Panel
    # titles are dropped (the axis labels and caption carry that text) so the
    # two panels fit the single column at this type size.
    FS_LABEL, FS_TICK, FS_LEG = 18, 15, 14
    fig, (axm, axl) = plt.subplots(1, 2, figsize=(7.6, 3.3),
                                   constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(b_list)))
    cmap = dict(zip(b_list, colors))

    axm.axhline(1.0, color="black", lw=2.2)   # the core: caption notes y=1 is it
    for B in b_list:
        core_flat = max(out["recurrent"][str(B)]["peak_mb"])
        ratio = [kv / core_flat for kv in out["token_kv_cache_mb"][str(B)]]
        axm.plot(t_list, ratio, "o--", color=cmap[B], lw=2.2, ms=6,
                 label=f"$B{{=}}{B}$")
    axm.set_xlabel("reasoning steps $T$", fontsize=FS_LABEL)
    axm.set_ylabel("peak memory / core", fontsize=FS_LABEL)
    axm.set_ylim(bottom=0)
    axm.tick_params(labelsize=FS_TICK)
    axm.legend(fontsize=FS_LEG, loc="lower right", handlelength=1.6,
               title="token KV-cache", title_fontsize=FS_LEG)
    axm.grid(alpha=0.3)

    for B in b_list:
        axl.plot(t_list, out["recurrent"][str(B)]["latency_ms"], "o-",
                 color=cmap[B], lw=2.2, ms=6, label=f"$B{{=}}{B}$")
    axl.set_xlabel("reasoning steps $T$", fontsize=FS_LABEL)
    axl.set_ylabel("latency / query (ms)", fontsize=FS_LABEL)
    axl.tick_params(labelsize=FS_TICK)
    axl.legend(fontsize=FS_LEG, loc="upper left", handlelength=1.6)
    axl.grid(alpha=0.3)

    fig.savefig("inference_cost.pdf", dpi=150, bbox_inches="tight", pad_inches=0.03)
    fig.savefig("inference_cost.png", dpi=150, bbox_inches="tight", pad_inches=0.03)
    print("saved inference_cost.{pdf,png}")


def main():
    import torch

    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda")
    block = _build_block().to(device).eval()

    out = {
        "config": dict(d_model=D_MODEL, nhead=NHEAD, dim_ff=DIM_FF, seq_len=L,
                       n_backbone=N_BACKBONE, reps=REPS,
                       gpu=torch.cuda.get_device_name(0)),
        "T_list": T_LIST, "B_list": B_LIST,
        "recurrent": {}, "token_kv_cache_mb": {},
    }
    for B in B_LIST:
        mem_r, lat_r = [], []
        for T in T_LIST:
            p, ms = measure(block, B, T, device)
            mem_r.append(round(p, 1))
            lat_r.append(round(ms, 3))
            print(f"  B={B:>4} T={T:>3}   peak={p:8.1f} MB   latency={ms:8.3f} ms",
                  flush=True)
        out["recurrent"][str(B)] = dict(peak_mb=mem_r, latency_ms=lat_r)
        out["token_kv_cache_mb"][str(B)] = [round(kv_cache_mb(B, T), 1) for T in T_LIST]

    with open("inference_cost.json", "w") as f:
        json.dump(out, f, indent=2)

    make_figure(out)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        with open("inference_cost.json") as f:
            make_figure(json.load(f))
    else:
        main()
