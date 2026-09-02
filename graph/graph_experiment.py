"""
Vertical Chain-of-Thought via Depth-Recurrent Transformers:
Solving Long-Path Graph Reachability by Thinking Deeper.

Uses adjacency matrix masking to constrain attention to graph edges,
turning the Transformer into a GNN where 1 thinking step = 1 hop.

Dependencies: torch, networkx, matplotlib, numpy
"""

import os
import sys
import random
import itertools

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------
# 0. Reproducibility
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 1. Vocabulary (node identity only)
# ----------------------------------------------------------------------
NODE_NAMES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]  # A-Z

PAD_TOKEN = "<PAD>"
VOCAB = {PAD_TOKEN: 0}
for name in NODE_NAMES:
    VOCAB[name] = len(VOCAB)
VOCAB_SIZE = len(VOCAB)
PAD_IDX = VOCAB[PAD_TOKEN]


# ----------------------------------------------------------------------
# 2. Graph generation & Dataset
# ----------------------------------------------------------------------

def _generate_reachable_graph(num_nodes: int, path_length: int):
    """
    Generate a random directed graph guaranteeing a path of exactly
    `path_length` hops. Distractor edges are added but never create
    a shortcut (preserving the exact shortest-path length).
    """
    assert path_length >= 1
    assert num_nodes >= path_length + 1

    nodes = random.sample(NODE_NAMES[:num_nodes], num_nodes)
    path_nodes = random.sample(nodes, path_length + 1)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for i in range(len(path_nodes) - 1):
        G.add_edge(path_nodes[i], path_nodes[i + 1])

    source, target = path_nodes[0], path_nodes[-1]

    # Add distractor edges, rejecting any that create a shorter path
    n_distractors = random.randint(num_nodes // 2, num_nodes * 2)
    for _ in range(n_distractors):
        u, v = random.sample(nodes, 2)
        G.add_edge(u, v)
        try:
            if nx.shortest_path_length(G, source, target) < path_length:
                G.remove_edge(u, v)
        except nx.NetworkXNoPath:
            pass

    sp = nx.shortest_path_length(G, source, target)
    return G, source, target, sp


def _generate_unreachable_graph(num_nodes: int):
    """Generate a directed graph with an unreachable (source, target) pair."""
    nodes = random.sample(NODE_NAMES[:num_nodes], num_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    n_edges = random.randint(num_nodes // 2, num_nodes * 2)
    for _ in range(n_edges):
        u, v = random.sample(nodes, 2)
        G.add_edge(u, v)

    random.shuffle(nodes)
    for s, t in itertools.combinations(nodes, 2):
        if not nx.has_path(G, s, t):
            return G, s, t

    # Fallback: partition into two disconnected components
    split = num_nodes // 2
    G_new = nx.DiGraph()
    G_new.add_nodes_from(nodes)
    for _ in range(n_edges):
        half = random.choice([nodes[:split], nodes[split:]])
        if len(half) < 2:
            continue
        u, v = random.sample(half, 2)
        G_new.add_edge(u, v)

    source = random.choice(nodes[:split])
    target = random.choice(nodes[split:])
    return G_new, source, target


class GraphReachabilityDataset(Dataset):
    """
    Binary classification: is `target` reachable from `source`?

    Returns per sample:
      - node_ids:  (max_nodes,) token IDs for each node, padded
      - adj_mask:  (max_nodes, max_nodes) boolean adjacency + self-loops
      - src_idx:   scalar index of source node
      - tgt_idx:   scalar index of target node
      - label:     scalar float, 0 or 1
    """

    def __init__(
        self,
        n_samples: int = 4000,
        hop_range: tuple[int, int] = (1, 3),
        num_nodes_range: tuple[int, int] = (6, 12),
        max_nodes: int = 20,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.samples: list[tuple] = []  # (G, source, target, label)

        n_pos = n_samples // 2
        n_neg = n_samples - n_pos

        for _ in range(n_pos):
            hops = random.randint(*hop_range)
            n_nodes = random.randint(
                max(hop_range[1] + 2, num_nodes_range[0]),
                num_nodes_range[1],
            )
            G, s, t, sp = _generate_reachable_graph(n_nodes, hops)
            self.samples.append((G, s, t, 1))

        for _ in range(n_neg):
            n_nodes = random.randint(*num_nodes_range)
            G, s, t = _generate_unreachable_graph(n_nodes)
            self.samples.append((G, s, t, 0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        G, source, target, label = self.samples[idx]
        nodes = sorted(G.nodes())
        n = len(nodes)
        node_to_idx = {name: i for i, name in enumerate(nodes)}

        # Node IDs (padded)
        node_ids = [VOCAB[name] for name in nodes] + [PAD_IDX] * (self.max_nodes - n)

        # Adjacency mask: mask[i, j] = True iff edge j->i OR i==j
        # Self-loops on ALL positions (including padding) for numerical stability
        adj = torch.zeros(self.max_nodes, self.max_nodes, dtype=torch.bool)
        for i in range(self.max_nodes):
            adj[i, i] = True
        for u, v in G.edges():
            ui, vi = node_to_idx[u], node_to_idx[v]
            adj[vi, ui] = True  # edge u->v: node v attends to node u

        src_idx = node_to_idx[source]
        tgt_idx = node_to_idx[target]

        return (
            torch.tensor(node_ids, dtype=torch.long),
            adj,
            torch.tensor(src_idx, dtype=torch.long),
            torch.tensor(tgt_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


# ----------------------------------------------------------------------
# 3. Model: GraphThinkingTransformer with Adjacency Masking
# ----------------------------------------------------------------------

class ThinkingBlock(nn.Module):
    """
    A single TransformerEncoderLayer applied recurrently with:
      - Adjacency-constrained attention (1 step = 1 hop)
      - Gated recurrence
      - Depth embedding per step
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float, max_steps: int):
        super().__init__()
        self.nhead = nhead
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.depth_emb = nn.Embedding(max_steps, d_model)
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        # Bias gate toward "keep old state" (sigmoid(-2) ~ 0.12) so signals
        # survive across many recurrent steps instead of vanishing.
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, h: torch.Tensor, step: int, adj_mask: torch.Tensor):
        """
        h        : (B, N, D)  - current node hidden states
        step     : int        - recurrence depth index
        adj_mask : (B, N, N)  - boolean, True = allowed to attend
        """
        B, N, D = h.shape

        # Inject depth signal
        depth = self.depth_emb(torch.tensor(step, device=h.device))
        h_input = h + depth.unsqueeze(0).unsqueeze(0)

        # Convert boolean mask to additive float mask: True->0, False->-inf
        attn_mask = torch.zeros(B, N, N, device=h.device, dtype=h.dtype)
        attn_mask.masked_fill_(~adj_mask, float('-inf'))
        # Expand for multi-head: (B, N, N) -> (B*nhead, N, N)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        attn_mask = attn_mask.reshape(B * self.nhead, N, N)

        h_new = self.layer(h_input, src_mask=attn_mask)

        # Gated update
        z = torch.sigmoid(self.gate_proj(torch.cat([h_new, h], dim=-1)))
        h_out = z * h_new + (1 - z) * h
        return h_out


class GraphThinkingTransformer(nn.Module):
    """
    Depth-Recurrent Transformer for binary graph reachability
    with adjacency matrix masking (1 layer = 1 hop of message passing).
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_nodes: int = 20,
        max_thinking_steps: int = 20,
        n_thinking_steps: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_thinking_steps = n_thinking_steps

        # Node identity embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        # Role embeddings: mark which node is source / target
        self.source_role_emb = nn.Parameter(torch.randn(d_model) * 0.02)
        self.target_role_emb = nn.Parameter(torch.randn(d_model) * 0.02)

        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Recurrent Thinking Block (single shared layer)
        self.thinking_block = ThinkingBlock(d_model, nhead, dim_ff, dropout, max_thinking_steps)

        # Classification head: concatenated source + target representations
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, node_ids, adj_mask, src_idx, tgt_idx, n_steps=None,
                return_all=False):
        """
        node_ids : (B, N)    - padded node token IDs
        adj_mask : (B, N, N) - boolean adjacency mask
        src_idx  : (B,)      - source node index per sample
        tgt_idx  : (B,)      - target node index per sample
        n_steps  : override thinking steps for test-time scaling
        return_all : if True, return logits from EVERY step, shape (T, B)
                     (used only by the --per-step-loss ablation)
        Returns  : (B,) logits from the FINAL thinking step only
        """
        B, N = node_ids.shape
        steps = n_steps if n_steps is not None else self.n_thinking_steps

        # Embed nodes
        h = self.token_emb(node_ids)  # (B, N, D)

        # Add role embeddings at source and target positions
        src_oh = F.one_hot(src_idx, num_classes=N).float().unsqueeze(-1)  # (B, N, 1)
        tgt_oh = F.one_hot(tgt_idx, num_classes=N).float().unsqueeze(-1)
        h = h + src_oh * self.source_role_emb + tgt_oh * self.target_role_emb

        h = self.emb_drop(self.emb_norm(h))

        # Recurrent thinking with adjacency masking
        batch_idx = torch.arange(B, device=h.device)

        def readout(state):
            src_repr = state[batch_idx, src_idx]  # (B, D)
            tgt_repr = state[batch_idx, tgt_idx]  # (B, D)
            cls_repr = torch.cat([src_repr, tgt_repr], dim=-1)  # (B, 2*D)
            return self.head(cls_repr).squeeze(-1)  # (B,)

        all_logits = []
        for t in range(steps):
            h = self.thinking_block(h, step=t, adj_mask=adj_mask)
            if return_all:
                all_logits.append(readout(h))

        if return_all:
            return torch.stack(all_logits)  # (T, B)
        return readout(h)


# ----------------------------------------------------------------------
# 4. Training utilities
# ----------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, scheduler, step_range=(5, 8),
                    per_step_loss=False):
    """
    Final-step loss only (default): the model must "think silently" for N
    steps and is only rewarded if the final answer is correct.
    Randomised depth for robustness.

    per_step_loss=True switches to intermediate supervision: the loss is
    averaged over the readout at EVERY thinking step (ablation baseline).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for node_ids, adj_mask, src_idx, tgt_idx, labels in loader:
        node_ids = node_ids.to(DEVICE)
        adj_mask = adj_mask.to(DEVICE)
        src_idx = src_idx.to(DEVICE)
        tgt_idx = tgt_idx.to(DEVICE)
        labels = labels.to(DEVICE)

        # Randomised thinking depth
        n_steps = random.randint(*step_range)

        if per_step_loss:
            logits_all = model(node_ids, adj_mask, src_idx, tgt_idx,
                               n_steps=n_steps, return_all=True)
            loss = torch.stack([
                F.binary_cross_entropy_with_logits(lg, labels)
                for lg in logits_all]).mean()
            logits = logits_all[-1]
        else:
            logits = model(node_ids, adj_mask, src_idx, tgt_idx, n_steps=n_steps)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        total_loss += loss.item() * node_ids.size(0)
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += node_ids.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, n_steps: int):
    model.eval()
    correct, total = 0, 0
    for node_ids, adj_mask, src_idx, tgt_idx, labels in loader:
        node_ids = node_ids.to(DEVICE)
        adj_mask = adj_mask.to(DEVICE)
        src_idx = src_idx.to(DEVICE)
        tgt_idx = tgt_idx.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(node_ids, adj_mask, src_idx, tgt_idx, n_steps=n_steps)
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += node_ids.size(0)
    return correct / total if total > 0 else 0.0


# ----------------------------------------------------------------------
# 5. Main experiment
# ----------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Graph reachability experiment.")
    ap.add_argument("--per-step-loss", action="store_true",
                    help="train with intermediate supervision (loss averaged "
                         "over every thinking step) instead of final-step-only; "
                         "results are written to perstep_* files")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random seed (default 42); non-default seeds write "
                         "seed-suffixed output files")
    ap.add_argument("--emb-ablation", action="store_true",
                    help="after training, save the checkpoint and run the "
                         "depth-embedding OOD confound diagnostic "
                         "(Part A replacement ablation + Part B gate logging); "
                         "writes embedding_ablation_results.json / gate_stats.png")
    args = ap.parse_args()
    per_step = args.per_step_loss

    # Module import seeds with SEED=42; reseed here so --seed takes effect
    # for data generation and initialisation alike.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    seed_suffix = "" if args.seed == SEED else f"_s{args.seed}"

    # -- Hyper-parameters --
    D_MODEL          = 128
    NHEAD            = 4
    DIM_FF           = 256
    DROPOUT          = 0.1
    MAX_NODES        = 20
    MAX_THINK        = 20
    TRAIN_THINK      = 5
    BATCH_SIZE       = 64
    EPOCHS           = 40
    LR               = 3e-4
    TRAIN_SAMPLES    = 12_000
    TEST_SAMPLES_PER = 500

    TRAIN_HOP_RANGE  = (1, 5)
    TRAIN_NODES      = (8, 16)
    TRAIN_STEP_RANGE = (5, 8)   # >= max train hops so model always has enough steps

    TEST_HOPS        = [1, 2, 3, 4, 6, 8, 10, 12]
    TEST_STEPS       = [1, 2, 3, 5, 8, 12, 15, 20]

    print("=" * 70)
    print("  Vertical CoT: Graph Reachability with Adjacency Masking")
    print("  (1 thinking step = 1 hop of message passing)")
    print(f"  Supervision   : {'PER-STEP (intermediate)' if per_step else 'final-step only'}")
    print("=" * 70)
    print(f"  Device        : {DEVICE}")
    print(f"  Train hops    : {TRAIN_HOP_RANGE}")
    print(f"  Train steps   : {TRAIN_STEP_RANGE}")
    print(f"  Test hops     : {TEST_HOPS}")
    print(f"  Test steps    : {TEST_STEPS}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Max nodes     : {MAX_NODES}")
    print(f"  Vocab size    : {VOCAB_SIZE}")
    print("=" * 70)

    # -- Dataset --
    print("\n[1/4] Generating training data ...")
    train_ds = GraphReachabilityDataset(
        n_samples=TRAIN_SAMPLES,
        hop_range=TRAIN_HOP_RANGE,
        num_nodes_range=TRAIN_NODES,
        max_nodes=MAX_NODES,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"       {len(train_ds)} training samples generated.")

    # -- OOD test sets (one per hop length) --
    print("[2/4] Generating OOD test data ...")
    test_loaders = {}
    for hops in TEST_HOPS:
        n_min = max(hops + 2, 10)
        n_max = min(max(hops + 6, 16), MAX_NODES)
        n_min = min(n_min, n_max)
        ds = GraphReachabilityDataset(
            n_samples=TEST_SAMPLES_PER,
            hop_range=(hops, hops),
            num_nodes_range=(n_min, n_max),
            max_nodes=MAX_NODES,
        )
        test_loaders[hops] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"       {len(TEST_HOPS)} test buckets x {TEST_SAMPLES_PER} samples each.")

    # -- Model --
    print("[3/4] Building model ...")
    model = GraphThinkingTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        dim_ff=DIM_FF,
        dropout=DROPOUT,
        max_nodes=MAX_NODES,
        max_thinking_steps=MAX_THINK,
        n_thinking_steps=TRAIN_THINK,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

    # -- Training --
    print("[4/4] Training ...\n")
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(model, train_loader, optimiser, scheduler,
                                     step_range=TRAIN_STEP_RANGE,
                                     per_step_loss=per_step)
        if epoch % 5 == 0 or epoch == 1:
            id_acc = evaluate(model, train_loader, n_steps=TRAIN_THINK)
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  train_acc={acc:.3f}  id_eval={id_acc:.3f}")

    # -- Save checkpoint (weights only; small) --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(out_dir, f"graph_model{seed_suffix}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Checkpoint saved to {ckpt_path}")

    # -- Evaluation Matrix --
    print("\n" + "=" * 70)
    print("  EVALUATION: Accuracy(hops, thinking_steps)")
    print("  Hypothesis: need steps >= hops for high accuracy")
    print("=" * 70)

    results = np.zeros((len(TEST_HOPS), len(TEST_STEPS)))
    for i, hops in enumerate(TEST_HOPS):
        for j, steps in enumerate(TEST_STEPS):
            acc = evaluate(model, test_loaders[hops], n_steps=steps)
            results[i, j] = acc

    # -- Print table --
    header = "  Hops \\ Steps |" + "".join(f"  {s:>3}  " for s in TEST_STEPS) + "|"
    sep    = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for i, hops in enumerate(TEST_HOPS):
        row = f"  {hops:>11} |"
        for j in range(len(TEST_STEPS)):
            val = results[i, j]
            row += f" {val:.3f} "
        row += "|"
        print(row)
    print(sep)

    # -- Heatmap --
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(results, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(TEST_STEPS)))
    ax.set_xticklabels(TEST_STEPS)
    ax.set_yticks(range(len(TEST_HOPS)))
    ax.set_yticklabels(TEST_HOPS)
    ax.set_xlabel("Thinking Steps (Compute Depth)", fontsize=12)
    ax.set_ylabel("Path Hops (Task Difficulty)", fontsize=12)
    loss_desc = "per-step loss" if per_step else "final-step loss"
    ax.set_title("Depth-Recurrent Transformer with Adjacency Masking\n"
                 f"(Trained on {TRAIN_HOP_RANGE[0]}-{TRAIN_HOP_RANGE[1]} hops; {loss_desc}; gate bias -2)",
                 fontsize=13, fontweight="bold")

    for i in range(len(TEST_HOPS)):
        for j in range(len(TEST_STEPS)):
            val = results[i, j]
            color = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=11)

    plt.tight_layout()
    fig_name = (f"perstep_results{seed_suffix}.pdf" if per_step
                else f"graph_results{seed_suffix}.pdf")
    plt.savefig(fig_name, dpi=150)
    print(f"\n  Heatmap saved to {fig_name}")

    # Save the raw grid so tables can be regenerated without re-training.
    import json
    grid_name = (f"perstep_results{seed_suffix}.json" if per_step
                 else f"graph_grid{seed_suffix}.json")
    with open(grid_name, "w") as f:
        json.dump({"supervision": "per_step" if per_step else "final_step",
                   "seed": args.seed,
                   "test_hops": TEST_HOPS, "test_steps": TEST_STEPS,
                   "grid": results.tolist()}, f, indent=2)
    print(f"  Grid saved to {grid_name}")

    # Headline ablation numbers: shortcut behaviour at step 1 on the deepest
    # OOD bucket, and OOD accuracy given a sufficient step budget.
    hop12 = TEST_HOPS.index(12)
    step1 = TEST_STEPS.index(1)
    suff = [j for j, s in enumerate(TEST_STEPS) if s >= 12]
    print(f"  Step-1 acc on 12-hop: {results[hop12, step1]:.3f}")
    print(f"  Best sufficient-step acc on 12-hop: {results[hop12, suff].max():.3f}")
    for hops in (6, 8, 10, 12):
        i = TEST_HOPS.index(hops)
        j = [k for k, s in enumerate(TEST_STEPS) if s >= hops]
        print(f"  Best acc at {hops:2d} hops (steps >= hops): {results[i, j].max():.3f}")

    # -- Summary --
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)
    for i, hops in enumerate(TEST_HOPS):
        insufficient = [results[i, j] for j, s in enumerate(TEST_STEPS) if s < hops]
        sufficient   = [results[i, j] for j, s in enumerate(TEST_STEPS) if s >= hops]
        avg_insuf = np.mean(insufficient) if insufficient else float('nan')
        avg_suf   = np.mean(sufficient) if sufficient else float('nan')
        print(f"  {hops:>2}-hop  |  steps<hops: {avg_insuf:.3f}  steps>=hops: {avg_suf:.3f}"
              f"  delta = {avg_suf - avg_insuf:+.3f}")
    print("=" * 70)
    print("  Expected: steps<hops ~ 0.50 (random), steps>=hops >> 0.50")
    print("=" * 70)

    # -- Depth-embedding OOD confound diagnostic (Part A + Part B) --
    if args.emb_ablation:
        sys.path.insert(0, os.path.join(out_dir, ".."))
        from emb_ablation_common import run_embedding_ablation
        # Training samples n_steps in TRAIN_STEP_RANGE=(5, 8); the recurrence
        # loop runs `for t in range(n_steps)`, so depth-embedding rows 0..7
        # are trained and rows 8..MAX_THINK-1 are never touched by a gradient.
        t_max_trained = TRAIN_STEP_RANGE[1]
        id_hops = [h for h in TEST_HOPS if h <= TRAIN_HOP_RANGE[1]]
        run_embedding_ablation(
            model=model,
            evaluate_fn=evaluate,
            test_loaders=test_loaders,
            difficulties=TEST_HOPS,
            steps=TEST_STEPS,
            t_max_trained=t_max_trained,
            id_difficulties=id_hops,
            out_dir=out_dir,
            task_name="graph",
            seed=args.seed,
            difficulty_label="path_hops",
            extra_meta={
                "train_step_range": list(TRAIN_STEP_RANGE),
                "train_hop_range": list(TRAIN_HOP_RANGE),
                "test_samples_per_bucket": TEST_SAMPLES_PER,
            },
        )


if __name__ == "__main__":
    main()
