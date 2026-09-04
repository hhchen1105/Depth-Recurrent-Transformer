"""
Depth-Recurrent Transformers for Family Relationship Reasoning.

Demonstrates that a recurrent Transformer can solve relational composition
tasks by "thinking" for more steps -- each step composes one hop of a
relationship chain (inspired by CLUTRR).

Hypothesis: Inferring an unstated family relationship from a chain of K
parent/child facts requires composing K binary relations.  A Recurrent
Transformer can solve longer chains (OOD) by thinking for more steps,
with accuracy degrading when thinking steps < chain length.

Chain generation uses an 'Apex (Common Ancestor)' routing strategy:
chains go UP the family tree (parent hops) then DOWN (child hops),
following a (+1)*(-1)* pattern.  This produces offsets in [-K, +K]
including 0 (sibling), making the task a genuine addition/subtraction
problem rather than a simple counting task.

Architecture reuses the setup from Nested Boolean Expression Eval:
  - Shared-weight ThinkingBlock with gated recurrence
  - RoPE for relative-position awareness
  - LayerScale for stable deep recurrence
  - Negative gate bias initialisation (-2.0)
  - Final-step loss only (no intermediate supervision)
  - Bidirectional (encoder) attention

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
SEED = 42   # default; override with --seed. Seeding happens in main().


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------
# 1. Vocabulary & Relations (word-level)
#    Strict 1-to-1 bijection: every offset +/-1..+/-12 has a unique token.
#    No catch-all categories -- eliminates label collisions.
# ----------------------------------------------------------------------

NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tina",
]

# Build relation names for offsets +/-1 .. +/-12
# +1 parent, +2 grandparent, +3 great-grandparent,
# +4 great-great-grandparent, +K "great^(K-2)-grandparent"
# Negative mirror: child, grandchild, ...

def _make_relation_name(offset):
    """Return a unique relation string for a signed integer offset."""
    assert offset != 0
    mag = abs(offset)
    if mag == 1:
        base = "parent" if offset > 0 else "child"
    elif mag == 2:
        base = "grandparent" if offset > 0 else "grandchild"
    else:
        prefix = "great-" * (mag - 2)
        base = prefix + ("grandparent" if offset > 0 else "grandchild")
    return base

MAX_OFFSET = 12  # covers chains up to depth 12

OFFSET_TO_RELATION = {}
RELATION_TO_OFFSET = {}
for _off in range(1, MAX_OFFSET + 1):
    for _sign in [+1, -1]:
        o = _sign * _off
        name = _make_relation_name(o)
        OFFSET_TO_RELATION[o] = name
        RELATION_TO_OFFSET[name] = o

# Offset 0 = same generation (sibling / cousin via common ancestor)
OFFSET_TO_RELATION[0] = "sibling"
RELATION_TO_OFFSET["sibling"] = 0


def offset_to_relation(offset):
    """Convert a generation offset to a relation string (including 0 -> sibling)."""
    return OFFSET_TO_RELATION[offset]  # KeyError = bug (offset out of range)


# Build word-level vocabulary
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]"]
RELATION_WORDS = list(RELATION_TO_OFFSET.keys())
FUNCTION_WORDS = ["is", "the", "of", "query", ":", "."]

ALL_WORDS = SPECIAL_TOKENS + NAMES + RELATION_WORDS + FUNCTION_WORDS
WORD_TO_ID = {w: i for i, w in enumerate(ALL_WORDS)}
ID_TO_WORD = {i: w for w, i in WORD_TO_ID.items()}

PAD_IDX = WORD_TO_ID["[PAD]"]
CLS_IDX = WORD_TO_ID["[CLS]"]
SEP_IDX = WORD_TO_ID["[SEP]"]
VOCAB_SIZE = len(ALL_WORDS)


def encode(text):
    """Encode a whitespace-separated text string to a list of token IDs."""
    return [WORD_TO_ID[w] for w in text.split()]


def decode(ids):
    """Decode a list of token IDs back to a string."""
    return " ".join(ID_TO_WORD[i] for i in ids if i != PAD_IDX)


# ----------------------------------------------------------------------
# 2. Data Generation & Dataset
# ----------------------------------------------------------------------

def generate_chain(depth):
    """
    Generate an 'Apex (Common Ancestor)' chain of `depth` hops.

    Returns: (entities, facts, cumulative_offset).

    The chain follows a (+1)* (-1)* pattern: first `u` parent-hops up to a
    common ancestor (the apex), then `d` child-hops back down, where
    u + d = depth.  The cumulative offset from entity_0 to entity_K is u - d,
    which can be positive, negative, or zero (sibling).

    All K+1 entities are unique, preventing self-reference.
    """
    entities = random.sample(NAMES, depth + 1)

    # Choose how many up-steps vs down-steps
    u = random.randint(0, depth)  # number of parent (up) hops
    d = depth - u                 # number of child (down) hops

    facts = []
    # Up phase: parent hops (entity_i is parent of entity_{i+1})
    for i in range(u):
        facts.append(f"{entities[i]} is the parent of {entities[i+1]} .")
    # Down phase: child hops (entity_i is child of entity_{i+1})
    for i in range(u, depth):
        facts.append(f"{entities[i]} is the child of {entities[i+1]} .")

    cumulative_offset = u - d  # entity_0's generation relative to entity_K

    return entities, facts, cumulative_offset


def generate_hard_negative(correct_offset):
    """
    Generate a hard negative offset from {correct +/- 2, correct +/- 4}.

    Uses even deltas only so that the negative always has the same parity
    as the correct offset.  This prevents a parity shortcut: the apex
    pattern forces offset = 2u - K (same parity as K), so odd deltas
    would produce structurally impossible offsets that are trivially
    rejectable.
    """
    candidates = []
    for delta in [2, -2, 4, -4]:  # even deltas only -- preserves parity
        neg_offset = correct_offset + delta
        if -MAX_OFFSET <= neg_offset <= MAX_OFFSET:
            candidates.append(neg_offset)
    return random.choice(candidates)


def generate_distractors(num_distractors, exclude_names):
    """
    Generate distractor sentences using parent/child/sibling words with
    names NOT in the chain.  Deduplicates entity pairs to avoid
    contradictory relations (e.g. "A is parent of B" + "A is sibling of B").
    """
    available = [n for n in NAMES if n not in exclude_names]
    distractors = []
    used_pairs = set()
    for _ in range(num_distractors):
        if len(available) < 2:
            break
        pair = random.sample(available, 2)
        pair_key = frozenset(pair)
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        rel = random.choice(["parent", "child", "sibling"])
        distractors.append(f"{pair[0]} is the {rel} of {pair[1]} .")
    return distractors


def generate_sample(depth, label, max_len=128):
    """
    Generate one family reasoning sample.

    Args:
        depth: chain length (number of hops)
        label: 1 (correct relation) or 0 (hard negative)
        max_len: maximum sequence length

    Returns: (text, label) or None if generation fails
    """
    # Generate chain (apex pattern -- offset can be any integer in [-depth, depth])
    entities, facts, offset = generate_chain(depth)

    # Determine query relation
    if label == 1:
        query_relation = offset_to_relation(offset)
    else:
        neg_offset = generate_hard_negative(offset)
        query_relation = offset_to_relation(neg_offset)

    # Generate distractors
    chain_names = set(entities)
    num_distractors = random.randint(2, 4)
    distractors = generate_distractors(num_distractors, chain_names)

    # Shuffle facts and distractors together
    all_sentences = facts + distractors
    random.shuffle(all_sentences)

    # Build full text: [CLS] + sentences + query : A is the R of B [SEP]
    query = (f"query : {entities[0]} is the {query_relation} "
             f"of {entities[-1]} [SEP]")
    text = "[CLS] " + " ".join(all_sentences) + " " + query

    # Check length
    tokens = text.split()
    if len(tokens) > max_len:
        return None

    return text, label


class FamilyReasoningDataset(Dataset):
    """
    Dataset of family relationship reasoning problems.

    Each sample: (token_ids, pad_mask, label)
      - token_ids : (max_len,) long tensor
      - pad_mask  : (max_len,) bool tensor, True = padding position
      - label     : scalar float, 0.0 (wrong relation) or 1.0 (correct relation)
    """

    def __init__(self, n_samples, depths, max_len=128):
        super().__init__()
        self.max_len = max_len
        self.samples = []  # (token_ids, label, depth, text)

        samples_per_depth = n_samples // len(depths)

        for depth in depths:
            count = 0
            attempts = 0
            max_attempts = samples_per_depth * 100
            while count < samples_per_depth and attempts < max_attempts:
                attempts += 1
                label = 1 if count % 2 == 0 else 0  # 50/50 balance
                result = generate_sample(depth, label, max_len)
                if result is None:
                    continue

                text, lbl = result
                token_ids = encode(text)

                # Pad
                if len(token_ids) < max_len:
                    token_ids += [PAD_IDX] * (max_len - len(token_ids))
                else:
                    token_ids = token_ids[:max_len]

                self.samples.append((token_ids, float(lbl), depth, text))
                count += 1

            if count < samples_per_depth:
                print(f"  WARNING: only generated {count}/{samples_per_depth} "
                      f"samples for depth {depth}")

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, label, depth, text = self.samples[idx]
        ids = torch.tensor(token_ids, dtype=torch.long)
        pad_mask = ids == PAD_IDX
        return ids, pad_mask, torch.tensor(label, dtype=torch.float32)


# ----------------------------------------------------------------------
# 3. Model: FamilyThinkingTransformer (RoPE + LayerScale)
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
    d = cos.shape[-1]
    x1, x2 = x[..., :d], x[..., d:]
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
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def _attention(self, x, cos, sin, pad_mask):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

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

    def forward(self, h, step, pad_mask=None, cos=None, sin=None):
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


class FamilyThinkingTransformer(nn.Module):
    """
    Depth-Recurrent Transformer for family relationship reasoning.

    Scaled architecture (d=256, ff=1024, 8 heads) with:
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
        max_thinking_steps=20,
        n_thinking_steps=8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_thinking_steps = n_thinking_steps

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.rope = RotaryEmbedding(d_model // nhead, max_len=max_seq_len)

        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.thinking_block = ThinkingBlock(
            d_model, nhead, dim_ff, dropout, max_thinking_steps
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, token_ids, pad_mask=None, n_steps=None):
        """
        token_ids : (B, L)  -- [CLS] + fact/distractor sentences + query [SEP] + [PAD...]
        pad_mask  : (B, L)  -- True at PAD positions
        n_steps   : override thinking steps for test-time scaling
        Returns   : (B,) logits (from the FINAL thinking step only)
        """
        B, L = token_ids.shape
        steps = n_steps if n_steps is not None else self.n_thinking_steps

        h = self.token_emb(token_ids)
        h = self.emb_drop(self.emb_norm(h))

        cos, sin = self.rope(L)

        for t in range(steps):
            h = self.thinking_block(h, step=t, pad_mask=pad_mask,
                                    cos=cos, sin=sin)

        cls_repr = h[:, 0]
        return self.head(cls_repr).squeeze(-1)


# ----------------------------------------------------------------------
# 4. Training utilities
# ----------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, scheduler, step_range=(1, 12)):
    """
    Final-step loss only: randomised thinking depth per batch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for token_ids, pad_mask, labels in loader:
        token_ids = token_ids.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        labels = labels.to(DEVICE)

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
# 5. Main experiment
# ----------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Relational composition (family) experiment.")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random seed; the accuracy grid is always saved to "
                         "family_results_s<seed>.npy (seed 42 also to "
                         "family_results.npy), and non-default seeds write "
                         "family_results_s<seed>.pdf instead of family_results.pdf")
    args = ap.parse_args()
    seed = args.seed
    _seed_everything(seed)

    # -- Hyper-parameters --
    D_MODEL          = 256
    NHEAD            = 8
    DIM_FF           = 1024
    DROPOUT          = 0.1
    MAX_SEQ_LEN      = 128
    MAX_THINK        = 20       # depth embeddings cover 0-19
    TRAIN_THINK      = 8        # ID eval uses 8 steps
    BATCH_SIZE       = 128
    EPOCHS           = 40
    LR               = 3e-4

    TRAIN_STEP_RANGE = (1, 12)
    TRAIN_DEPTHS     = [2, 3, 4, 5]
    TEST_ID_DEPTHS   = [2, 3, 4, 5]
    TEST_OOD_DEPTHS  = [6, 7, 8, 9]
    TEST_DEPTHS      = TEST_ID_DEPTHS + TEST_OOD_DEPTHS
    TEST_STEPS       = [1, 2, 4, 6, 8, 10, 12, 16, 20]

    TRAIN_SAMPLES    = 60_000   # 15K per depth level
    TEST_SAMPLES_PER = 2_000

    print("=" * 70)
    print("  Depth-Recurrent Transformer: Family Relationship Reasoning")
    print("  (Each thinking step composes one hop of a relationship chain)")
    print("=" * 70)
    print(f"  Device           : {DEVICE}")
    print(f"  Vocab size       : {VOCAB_SIZE}")
    print(f"  Train depths     : {TRAIN_DEPTHS}")
    print(f"  Train steps      : {TRAIN_STEP_RANGE}")
    print(f"  Test depths (ID) : {TEST_ID_DEPTHS}")
    print(f"  Test depths (OOD): {TEST_OOD_DEPTHS}")
    print(f"  Test steps       : {TEST_STEPS}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Max sequence len : {MAX_SEQ_LEN}")
    print("=" * 70)

    # -- Dataset --
    print("\n[1/4] Generating training data ...")
    train_ds = FamilyReasoningDataset(
        n_samples=TRAIN_SAMPLES,
        depths=TRAIN_DEPTHS,
        max_len=MAX_SEQ_LEN,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Label distribution
    n_pos = sum(1 for s in train_ds.samples if s[1] == 1.0)
    n_neg = len(train_ds) - n_pos
    print(f"       {len(train_ds)} training samples  (pos={n_pos}, neg={n_neg})")

    # Show example samples for data quality check
    print("\n       -- Sample data (10 examples) --")
    for i in range(min(10, len(train_ds.samples))):
        token_ids, label, depth, text = train_ds.samples[i]
        tag = "CORRECT" if label == 1.0 else "WRONG"
        print(f"       [{tag:>7}] depth={depth}  {text}")
    print()

    # Label balance per depth
    for depth in TRAIN_DEPTHS:
        depth_samples = [s for s in train_ds.samples if s[2] == depth]
        pos = sum(1 for s in depth_samples if s[1] == 1.0)
        neg = len(depth_samples) - pos
        print(f"       depth={depth}: {len(depth_samples)} samples "
              f"(pos={pos}, neg={neg})")

    # -- Test sets --
    print("\n[2/4] Generating test data ...")
    test_loaders = {}
    for depth in TEST_DEPTHS:
        ds = FamilyReasoningDataset(
            n_samples=TEST_SAMPLES_PER,
            depths=[depth],
            max_len=MAX_SEQ_LEN,
        )
        test_loaders[depth] = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False
        )
    print(f"       {len(TEST_DEPTHS)} test buckets x {TEST_SAMPLES_PER} samples each.")

    # -- Model --
    print("\n[3/4] Building model ...")
    model = FamilyThinkingTransformer(
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
    print("\n[4/4] Training ...\n")
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(
            model, train_loader, optimiser, scheduler,
            step_range=TRAIN_STEP_RANGE,
        )
        if epoch % 2 == 0 or epoch == 1:
            id_acc = evaluate(model, train_loader, n_steps=TRAIN_THINK)
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  "
                f"train_acc={acc:.3f}  id_eval={id_acc:.3f}"
            )

    # -- Save checkpoint --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_name = "family_model.pt" if seed == SEED else f"family_model_s{seed}.pt"
    ckpt_path = os.path.join(out_dir, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Checkpoint saved to {ckpt_path}")

    # -- Evaluation Matrix --
    print("\n" + "=" * 70)
    print("  EVALUATION: Accuracy(chain_depth, thinking_steps)")
    print("  Hypothesis: longer chains require more thinking steps")
    print("=" * 70)

    results = np.zeros((len(TEST_DEPTHS), len(TEST_STEPS)))
    for i, depth in enumerate(TEST_DEPTHS):
        for j, steps in enumerate(TEST_STEPS):
            acc = evaluate(model, test_loaders[depth], n_steps=steps)
            results[i, j] = acc

    # -- Save results array (per seed; seed 42 also to the canonical name for
    #    plot_results.py / make_baseline_table.py) --
    np.save(os.path.join(out_dir, f"family_results_s{seed}.npy"), results)
    if seed == SEED:
        np.save(os.path.join(out_dir, "family_results.npy"), results)

    # -- Print table --
    header = "  Depth \\ Steps |" + "".join(f"  {s:>3}  " for s in TEST_STEPS) + "|"
    sep = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for i, depth in enumerate(TEST_DEPTHS):
        marker = " " if depth in TEST_ID_DEPTHS else "*"
        row = f"  {depth:>12}{marker}|"
        for j in range(len(TEST_STEPS)):
            row += f" {results[i, j]:.3f} "
        row += "|"
        print(row)
        if depth == TEST_ID_DEPTHS[-1]:
            print("  " + "-" * (len(header) - 2))
    print(sep)
    print("  (* = OOD depth, not seen during training)")

    # -- Heatmap --
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(results, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(TEST_STEPS)))
    ax.set_xticklabels(TEST_STEPS)
    ax.set_yticks(range(len(TEST_DEPTHS)))
    ax.set_yticklabels(TEST_DEPTHS)
    ax.set_xlabel("Thinking Steps (Compute Depth)", fontsize=12)
    ax.set_ylabel("Chain Depth (Number of Hops)", fontsize=12)
    ax.set_title(
        "Depth-Recurrent Transformer: Family Relationship Reasoning\n"
        f"(d={D_MODEL}, RoPE+LayerScale; train depth {TRAIN_DEPTHS[0]}-"
        f"{TRAIN_DEPTHS[-1]}; gate bias -2)",
        fontsize=13,
        fontweight="bold",
    )

    # Horizontal line separating ID from OOD
    id_ood_boundary = len(TEST_ID_DEPTHS) - 0.5
    ax.axhline(y=id_ood_boundary, color="white", linewidth=2.5, linestyle="--")
    ax.text(len(TEST_STEPS) - 0.5, id_ood_boundary - 0.35, "ID",
            ha="right", va="bottom", fontsize=10, color="white",
            fontweight="bold")
    ax.text(len(TEST_STEPS) - 0.5, id_ood_boundary + 0.35, "OOD",
            ha="right", va="top", fontsize=10, color="white",
            fontweight="bold")

    for i in range(len(TEST_DEPTHS)):
        for j in range(len(TEST_STEPS)):
            val = results[i, j]
            color = "white" if val < 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=11)

    plt.tight_layout()
    fig_name = "family_results.pdf" if seed == SEED else f"family_results_s{seed}.pdf"
    fig_path = os.path.join(out_dir, fig_name)
    plt.savefig(fig_path, dpi=150)
    print(f"\n  Heatmap saved to {fig_path}")

    # -- Summary --
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)
    for i, depth in enumerate(TEST_DEPTHS):
        tag = "ID " if depth in TEST_ID_DEPTHS else "OOD"
        insufficient = [
            results[i, j] for j, s in enumerate(TEST_STEPS) if s < depth
        ]
        sufficient = [
            results[i, j] for j, s in enumerate(TEST_STEPS) if s >= depth
        ]
        avg_insuf = np.mean(insufficient) if insufficient else float("nan")
        avg_suf = np.mean(sufficient) if sufficient else float("nan")
        print(
            f"  [{tag}] depth={depth}  |  steps<depth: {avg_insuf:.3f}  "
            f"steps>=depth: {avg_suf:.3f}  delta = {avg_suf - avg_insuf:+.3f}"
        )
    print("=" * 70)
    print("  Expected: steps<depth ~ 0.50 (random), steps>=depth >> 0.50")
    print("=" * 70)


if __name__ == "__main__":
    main()
