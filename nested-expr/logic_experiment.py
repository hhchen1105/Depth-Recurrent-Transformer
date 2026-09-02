"""
Depth-Recurrent Transformers for Nested Boolean Expression Evaluation.

Demonstrates that a recurrent Transformer can solve hierarchical/recursive
tasks by "thinking" for more steps -- each step unpeels one layer of nesting.

Hypothesis: Standard Transformers struggle with deep nesting because they lack
the computational depth to simulate a stack.  A Recurrent Transformer can solve
deeper nesting (OOD) by thinking for more steps (unwrapping layer by layer).

Architecture reuses the graph reachability reasoning core:
  - Shared-weight ThinkingBlock with gated recurrence
  - Negative gate bias initialisation (-2.0)
  - Final-step loss only (no intermediate supervision)
  - Bidirectional (encoder) attention (full expression is visible)

Dependencies: torch, matplotlib, numpy
"""

import sys
import os
import random

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------
# 1. Vocabulary (character-level)
# ----------------------------------------------------------------------
TOKENS = {
    "<PAD>": 0,
    "<CLS>": 1,
    "T": 2,
    "F": 3,
    "&": 4,
    "|": 5,
    "!": 6,
    "(": 7,
    ")": 8,
}
PAD_IDX = TOKENS["<PAD>"]
CLS_IDX = TOKENS["<CLS>"]
VOCAB_SIZE = len(TOKENS)
CHAR_TO_IDX = {k: v for k, v in TOKENS.items() if k not in ("<PAD>", "<CLS>")}


# ----------------------------------------------------------------------
# 2. Boolean Expression Generation & Evaluation
# ----------------------------------------------------------------------

def _atom():
    """Generate a depth-0 boolean expression (no parentheses)."""
    r = random.random()
    if r < 0.5:
        return random.choice(["T", "F"])
    elif r < 0.75:
        return "!" + random.choice(["T", "F"])
    else:
        a = random.choice(["T", "F"])
        b = random.choice(["T", "F"])
        op = random.choice(["&", "|"])
        return a + op + b


def generate_expr(target_depth):
    """
    Generate a valid boolean expression with exactly `target_depth`
    maximum nesting depth (= max depth of nested parentheses).

    Strategy: recursively build one mandatory deep branch wrapped in
    parentheses, with a shallow sibling to keep expression size compact.
    """
    if target_depth == 0:
        return _atom()

    op = random.choice(["&", "|"])

    # One child must achieve target_depth - 1 (parentheses around it add +1)
    deep_child = generate_expr(target_depth - 1)

    # Other child: bias heavily toward shallow to keep expressions compact.
    # Weights [2^(d-1), 2^(d-2), ..., 1] strongly favour depth 0.
    if target_depth > 1:
        weights = [2 ** (target_depth - 1 - d) for d in range(target_depth)]
        other_depth = random.choices(range(target_depth), weights=weights, k=1)[0]
    else:
        other_depth = 0
    other_child = generate_expr(other_depth)

    # Randomly order children
    if random.random() < 0.5:
        inner = deep_child + op + other_child
    else:
        inner = other_child + op + deep_child

    result = "(" + inner + ")"

    # Optionally prepend NOT (25 % chance)
    if random.random() < 0.25:
        result = "!" + result

    return result


def nesting_depth(expr_str):
    """Compute maximum parenthesis nesting depth."""
    depth = 0
    max_depth = 0
    for c in expr_str:
        if c == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ")":
            depth -= 1
    return max_depth


def evaluate_expr(expr_str):
    """
    Evaluate a boolean expression via recursive descent parsing.
    Operator precedence: NOT > AND > OR.
    Returns: bool
    """
    tokens = list(expr_str)
    pos = [0]  # mutable pointer

    def parse_or():
        left = parse_and()
        while pos[0] < len(tokens) and tokens[pos[0]] == "|":
            pos[0] += 1
            right = parse_and()
            left = left or right
        return left

    def parse_and():
        left = parse_not()
        while pos[0] < len(tokens) and tokens[pos[0]] == "&":
            pos[0] += 1
            right = parse_not()
            left = left and right
        return left

    def parse_not():
        if pos[0] < len(tokens) and tokens[pos[0]] == "!":
            pos[0] += 1
            return not parse_not()
        return parse_primary()

    def parse_primary():
        tok = tokens[pos[0]]
        if tok == "(":
            pos[0] += 1          # skip '('
            result = parse_or()
            pos[0] += 1          # skip ')'
            return result
        elif tok == "T":
            pos[0] += 1
            return True
        elif tok == "F":
            pos[0] += 1
            return False
        else:
            raise ValueError(f"Unexpected token '{tok}' at position {pos[0]}")

    result = parse_or()
    assert pos[0] == len(tokens), (
        f"Parsing incomplete: consumed {pos[0]}/{len(tokens)} tokens in '{expr_str}'"
    )
    return result


def tokenize(expr_str, max_len):
    """Convert expression string to token IDs: [CLS] + expr + [PAD...]."""
    ids = [CLS_IDX] + [CHAR_TO_IDX[c] for c in expr_str]
    length = len(ids)
    if length < max_len:
        ids += [PAD_IDX] * (max_len - length)
    else:
        ids = ids[:max_len]
    return ids


# ----------------------------------------------------------------------
# 3. Dataset
# ----------------------------------------------------------------------

class BooleanExpressionDataset(Dataset):
    """
    Dataset of random boolean expressions with controlled nesting depth.

    Each sample: (token_ids, pad_mask, label)
      - token_ids : (max_len,) long tensor
      - pad_mask  : (max_len,) bool tensor, True = padding position
      - label     : scalar float, 0.0 (False) or 1.0 (True)
    """

    def __init__(self, n_samples, depths, max_len=128, max_attempts_factor=100):
        super().__init__()
        self.max_len = max_len
        self.samples = []  # (token_ids, label, depth, expr_str)

        samples_per_depth = n_samples // len(depths)

        for depth in depths:
            count = 0
            attempts = 0
            max_attempts = samples_per_depth * max_attempts_factor
            while count < samples_per_depth and attempts < max_attempts:
                attempts += 1
                expr = generate_expr(depth)

                # Validate nesting depth
                actual_depth = nesting_depth(expr)
                if actual_depth != depth:
                    continue

                # Validate length (need room for CLS token)
                if len(expr) + 1 > max_len:
                    continue

                # Validate evaluation
                try:
                    result = evaluate_expr(expr)
                except Exception:
                    continue

                label = 1.0 if result else 0.0
                token_ids = tokenize(expr, max_len)
                self.samples.append((token_ids, label, depth, expr))
                count += 1

            if count < samples_per_depth:
                print(f"  WARNING: only generated {count}/{samples_per_depth} "
                      f"samples for depth {depth}")

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, label, depth, expr = self.samples[idx]
        ids = torch.tensor(token_ids, dtype=torch.long)
        pad_mask = ids == PAD_IDX
        return ids, pad_mask, torch.tensor(label, dtype=torch.float32)


# ----------------------------------------------------------------------
# 4. Model: LogicThinkingTransformer  (Scale-Up with RoPE & LayerScale)
# ----------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).
    Pre-computes sin/cos tables for the per-head dimension."""

    def __init__(self, head_dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        # Pre-compute cache  -- (max_len, head_dim/2)
        t = torch.arange(max_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())

    def forward(self, seq_len):
        """Returns (cos, sin) each of shape (1, 1, seq_len, head_dim/2)."""
        c = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        s = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return c, s


def apply_rotary_pos_emb(x, cos, sin):
    """Apply RoPE to a (B, nhead, L, head_dim) tensor."""
    d = cos.shape[-1]                       # head_dim / 2
    x1, x2 = x[..., :d], x[..., d:]        # split halves
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


class LayerScale(nn.Module):
    """Per-channel learnable scaling (Touvron et al., 2021)."""

    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class ThinkingBlock(nn.Module):
    """
    Custom pre-norm Transformer block applied recurrently with:
      - RoPE (rotary position embeddings in Q/K)
      - LayerScale (stabilises deep recurrence, init 1e-4)
      - Gated recurrence with negative gate bias (-2.0)
      - Depth embedding per step
    Full bidirectional attention (no causal / adjacency masking).
    """

    def __init__(self, d_model, nhead, dim_ff, dropout, max_steps):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

        # Pre-norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # LayerScale
        self.ls1 = LayerScale(d_model, init_value=1e-4)
        self.ls2 = LayerScale(d_model, init_value=1e-4)

        # Depth embedding & gated recurrence
        self.depth_emb = nn.Embedding(max_steps, d_model)
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        # Bias gate toward "keep old state" -- sigmoid(-2) ~ 0.12
        nn.init.constant_(self.gate_proj.bias, -2.0)

    # -- attention with RoPE ------------------------------------------
    def _attention(self, x, cos, sin, pad_mask):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Use PyTorch's optimised SDPA (MPS-accelerated)
        # pad_mask: (B, L) True = pad.  Expand to (B, 1, 1, L) for broadcast
        sdpa_mask = None
        if pad_mask is not None:
            sdpa_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
            sdpa_mask.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

    # -- forward ------------------------------------------------------
    def forward(self, h, step, pad_mask=None, cos=None, sin=None):
        # Inject depth signal
        depth = self.depth_emb(torch.tensor(step, device=h.device))
        h_input = h + depth.unsqueeze(0).unsqueeze(0)

        # Pre-norm self-attention + LayerScale + residual
        x = self.norm1(h_input)
        h_new = h_input + self.ls1(self._attention(x, cos, sin, pad_mask))

        # Pre-norm FFN + LayerScale + residual
        x = self.norm2(h_new)
        h_new = h_new + self.ls2(self.ffn(x))

        # Gated update
        z = torch.sigmoid(self.gate_proj(torch.cat([h_new, h], dim=-1)))
        return z * h_new + (1 - z) * h


class LogicThinkingTransformer(nn.Module):
    """
    Depth-Recurrent Transformer for boolean expression evaluation.

    Scaled-up architecture (d=256, ff=1024, 8 heads) with:
      - RoPE for relative-position awareness inside attention
      - LayerScale for stable deep recurrence
      - Gated recurrence with gate bias -2.0
    Classification via CLS token at position 0.
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nhead=8,
        dim_ff=1024,
        dropout=0.1,
        max_seq_len=128,
        max_thinking_steps=28,
        n_thinking_steps=10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_thinking_steps = n_thinking_steps

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        # RoPE handles position inside attention -- no additive PE needed
        self.rope = RotaryEmbedding(d_model // nhead, max_len=max_seq_len)

        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Recurrent Thinking Block (single shared layer)
        self.thinking_block = ThinkingBlock(
            d_model, nhead, dim_ff, dropout, max_thinking_steps
        )

        # Classification head: CLS representation -> binary
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, token_ids, pad_mask=None, n_steps=None):
        """
        token_ids : (B, L)  -- [CLS] + expression tokens + [PAD...]
        pad_mask  : (B, L)  -- True at PAD positions
        n_steps   : override thinking steps for test-time scaling
        Returns   : (B,) logits (from the FINAL thinking step only)
        """
        B, L = token_ids.shape
        steps = n_steps if n_steps is not None else self.n_thinking_steps

        # Embed tokens (RoPE applied inside attention, not here)
        h = self.token_emb(token_ids)
        h = self.emb_drop(self.emb_norm(h))

        # Compute RoPE sin/cos once -- reused across all thinking steps
        cos, sin = self.rope(L)

        # Recurrent thinking (bidirectional attention, padding-masked)
        for t in range(steps):
            h = self.thinking_block(h, step=t, pad_mask=pad_mask,
                                    cos=cos, sin=sin)

        # Classify from CLS token (position 0)
        cls_repr = h[:, 0]  # (B, D)
        return self.head(cls_repr).squeeze(-1)  # (B,)


# ----------------------------------------------------------------------
# 5. Training utilities
# ----------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, scheduler, step_range=(4, 8)):
    """
    Final-step loss only: the model must "think silently" for N steps
    and is only rewarded if the final answer is correct.
    Randomised depth for robustness.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for token_ids, pad_mask, labels in loader:
        token_ids = token_ids.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        # Randomised thinking depth
        n_steps = random.randint(*step_range)

        logits = model(token_ids, pad_mask, n_steps=n_steps)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        total_loss += loss.item() * token_ids.size(0)
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += token_ids.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, n_steps):
    model.eval()
    correct, total = 0, 0
    for token_ids, pad_mask, labels in loader:
        token_ids = token_ids.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(token_ids, pad_mask, n_steps=n_steps)
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += token_ids.size(0)
    return correct / total if total > 0 else 0.0


# ----------------------------------------------------------------------
# 6. Main experiment
# ----------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Nested boolean expression experiment.")
    ap.add_argument("--emb-ablation", action="store_true",
                    help="after training, save the checkpoint and run the "
                         "depth-embedding OOD confound diagnostic "
                         "(Part A replacement ablation + Part B gate logging); "
                         "writes embedding_ablation_results.json / gate_stats.png")
    args = ap.parse_args()

    # -- Hyper-parameters (Scale-Up & Converge) --
    D_MODEL          = 256       # 2x wider  -> more working memory
    NHEAD            = 8         # 8 heads, head_dim = 32
    DIM_FF           = 1024      # 4x d_model
    DROPOUT          = 0.1
    MAX_SEQ_LEN      = 128       # fits depth <= 8 easily; depth 14 via retries
    MAX_THINK        = 28        # depth embeddings cover up to 27
    TRAIN_THINK      = 10        # ID eval uses 10 steps (covers depth 1-8+)
    BATCH_SIZE       = 128
    EPOCHS           = 30        # converges by ~25 (99.6% ID eval)
    LR               = 3e-4
    TRAIN_SAMPLES    = 64_000    # 8 000 per depth level (high diversity)
    TEST_SAMPLES_PER = 500

    TRAIN_STEP_RANGE = (4, 16)   # trains depth embeddings 0-15; covers depth 8
    TRAIN_DEPTHS     = [1, 2, 3, 4, 5, 6, 7, 8]   # <- curriculum fix
    TEST_DEPTHS      = [2, 4, 6, 8, 10, 12, 14]
    TEST_STEPS       = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]

    print("=" * 70)
    print("  Depth-Recurrent Transformer: Nested Boolean Expression Evaluation")
    print("  (Each thinking step unpeels one layer of nesting)")
    print("=" * 70)
    print(f"  Device           : {DEVICE}")
    print(f"  Train depths     : {TRAIN_DEPTHS}")
    print(f"  Train steps      : {TRAIN_STEP_RANGE}")
    print(f"  Test depths      : {TEST_DEPTHS}")
    print(f"  Test steps       : {TEST_STEPS}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Max sequence len : {MAX_SEQ_LEN}")
    print(f"  Vocab size       : {VOCAB_SIZE}")
    print("=" * 70)

    # -- Dataset --
    print("\n[1/4] Generating training data ...")
    train_ds = BooleanExpressionDataset(
        n_samples=TRAIN_SAMPLES,
        depths=TRAIN_DEPTHS,
        max_len=MAX_SEQ_LEN,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Label distribution
    n_true = sum(1 for s in train_ds.samples if s[1] == 1.0)
    n_false = len(train_ds) - n_true
    print(f"       {len(train_ds)} training samples  (True={n_true}, False={n_false})")

    # Show example expressions
    print("       Example expressions:")
    shown = set()
    for token_ids, label, depth, expr in train_ds.samples:
        if depth not in shown:
            shown.add(depth)
            val = "T" if label == 1.0 else "F"
            print(f"         depth={depth}: {expr}  ->  {val}")
        if len(shown) == len(TRAIN_DEPTHS):
            break

    # -- OOD test sets (one per depth level) --
    print("[2/4] Generating test data ...")
    test_loaders = {}
    for depth in TEST_DEPTHS:
        ds = BooleanExpressionDataset(
            n_samples=TEST_SAMPLES_PER,
            depths=[depth],
            max_len=MAX_SEQ_LEN,
        )
        test_loaders[depth] = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False
        )
    print(f"       {len(TEST_DEPTHS)} test buckets x {TEST_SAMPLES_PER} samples each.")

    # -- Model --
    print("[3/4] Building model ...")
    model = LogicThinkingTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        dim_ff=DIM_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        max_thinking_steps=MAX_THINK,
        n_thinking_steps=TRAIN_THINK,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=total_steps
    )

    # -- Training --
    print("[4/4] Training ...\n")
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(
            model, train_loader, optimiser, scheduler,
            step_range=TRAIN_STEP_RANGE,
        )
        if epoch % 5 == 0 or epoch == 1:
            id_acc = evaluate(model, train_loader, n_steps=TRAIN_THINK)
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  "
                f"train_acc={acc:.3f}  id_eval={id_acc:.3f}"
            )

    # -- Save checkpoint (weights only; small) --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(out_dir, "logic_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Checkpoint saved to {ckpt_path}")

    # -- Evaluation Matrix --
    print("\n" + "=" * 70)
    print("  EVALUATION: Accuracy(nesting_depth, thinking_steps)")
    print("  Hypothesis: deeper nesting requires more thinking steps")
    print("=" * 70)

    results = np.zeros((len(TEST_DEPTHS), len(TEST_STEPS)))
    for i, depth in enumerate(TEST_DEPTHS):
        for j, steps in enumerate(TEST_STEPS):
            acc = evaluate(model, test_loaders[depth], n_steps=steps)
            results[i, j] = acc

    # -- Print table --
    header = "  Depth \\ Steps |" + "".join(f"  {s:>3}  " for s in TEST_STEPS) + "|"
    sep = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for i, depth in enumerate(TEST_DEPTHS):
        row = f"  {depth:>12} |"
        for j in range(len(TEST_STEPS)):
            row += f" {results[i, j]:.3f} "
        row += "|"
        print(row)
    print(sep)

    # -- Heatmap --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(results, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(TEST_STEPS)))
    ax.set_xticklabels(TEST_STEPS)
    ax.set_yticks(range(len(TEST_DEPTHS)))
    ax.set_yticklabels(TEST_DEPTHS)
    ax.set_xlabel("Thinking Steps (Compute Depth)", fontsize=12)
    ax.set_ylabel("Nesting Depth (Task Difficulty)", fontsize=12)
    ax.set_title(
        "Depth-Recurrent Transformer: Nested Boolean Expression Evaluation\n"
        f"(d={D_MODEL}, RoPE+LayerScale; depth {TRAIN_DEPTHS[0]}-{TRAIN_DEPTHS[-1]}; "
        f"gate bias -2)",
        fontsize=13,
        fontweight="bold",
    )

    for i in range(len(TEST_DEPTHS)):
        for j in range(len(TEST_STEPS)):
            val = results[i, j]
            color = "white" if val < 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=11)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "logic_results.pdf")
    plt.savefig(fig_path, dpi=150)
    print(f"\n  Heatmap saved to {fig_path}")

    # -- Summary --
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)
    for i, depth in enumerate(TEST_DEPTHS):
        insufficient = [
            results[i, j] for j, s in enumerate(TEST_STEPS) if s < depth
        ]
        sufficient = [
            results[i, j] for j, s in enumerate(TEST_STEPS) if s >= depth
        ]
        avg_insuf = np.mean(insufficient) if insufficient else float("nan")
        avg_suf = np.mean(sufficient) if sufficient else float("nan")
        print(
            f"  depth={depth:>2}  |  steps<depth: {avg_insuf:.3f}  "
            f"steps>=depth: {avg_suf:.3f}  delta = {avg_suf - avg_insuf:+.3f}"
        )
    print("=" * 70)
    print("  Expected: steps<depth ~ 0.50 (random), steps>=depth >> 0.50")
    print("=" * 70)

    # -- Depth-embedding OOD confound diagnostic (Part A + Part B) --
    if args.emb_ablation:
        sys.path.insert(0, os.path.join(out_dir, ".."))
        from emb_ablation_common import run_embedding_ablation
        # Training samples n_steps in TRAIN_STEP_RANGE=(4, 16); the recurrence
        # loop runs `for t in range(n_steps)`, so depth-embedding rows 0..15
        # are trained and rows 16..MAX_THINK-1 are never touched by a gradient.
        t_max_trained = TRAIN_STEP_RANGE[1]
        id_depths = [d for d in TEST_DEPTHS if d <= TRAIN_DEPTHS[-1]]
        run_embedding_ablation(
            model=model,
            evaluate_fn=evaluate,
            test_loaders=test_loaders,
            difficulties=TEST_DEPTHS,
            steps=TEST_STEPS,
            t_max_trained=t_max_trained,
            id_difficulties=id_depths,
            out_dir=out_dir,
            task_name="logic",
            seed=42,
            difficulty_label="nesting_depth",
            extra_meta={
                "train_step_range": list(TRAIN_STEP_RANGE),
                "train_depths": TRAIN_DEPTHS,
                "test_samples_per_bucket": TEST_SAMPLES_PER,
            },
        )


if __name__ == "__main__":
    main()
