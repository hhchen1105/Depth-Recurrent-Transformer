"""
Depth-embedding OOD confound diagnostic for the family-reasoning task.

Eval-only: loads the existing `family_model.pt` checkpoint, reuses the
`FamilyThinkingTransformer` class, the `FamilyReasoningDataset`, and the
`evaluate()` function from `family_experiment.py`, and runs the
embedding-replacement ablation (Part A) + gate-behaviour logging (Part B)
defined in `../emb_ablation_common.py`.

Nothing is retrained. Outputs (in this directory):
  embedding_ablation_results.json
  gate_stats.png
  embedding_ablation_summary.png

Run under Slurm (GPU):  sbatch run_emb_ablation_family.sh
"""

import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import family_experiment as fe
from family_experiment import (
    FamilyThinkingTransformer, FamilyReasoningDataset, evaluate,
    VOCAB_SIZE, DEVICE,
)
from emb_ablation_common import run_embedding_ablation


def main():
    # -- Config: must match family_experiment.main() --
    D_MODEL, NHEAD, DIM_FF, DROPOUT = 256, 8, 1024, 0.1
    MAX_SEQ_LEN = 128
    MAX_THINK = 20
    TRAIN_THINK = 8

    TRAIN_STEP_RANGE = (1, 12)     # -> iterations t in 0..11 trained
    T_MAX_TRAINED = TRAIN_STEP_RANGE[1]   # first never-trained row index = 12

    TRAIN_DEPTHS = [2, 3, 4, 5]
    TEST_ID_DEPTHS = [2, 3, 4, 5]
    TEST_OOD_DEPTHS = [6, 7, 8, 9]
    TEST_DEPTHS = TEST_ID_DEPTHS + TEST_OOD_DEPTHS
    TEST_STEPS = [1, 2, 4, 6, 8, 10, 12, 16, 20]
    TEST_SAMPLES_PER = 2_000

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ckpt_path = os.path.join(_HERE, "family_model.pt")
    print(f"[load] checkpoint : {ckpt_path}")
    print(f"[load] device     : {DEVICE}")

    model = FamilyThinkingTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, nhead=NHEAD, dim_ff=DIM_FF, dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        max_thinking_steps=MAX_THINK,
        n_thinking_steps=TRAIN_THINK,
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # sanity: how "untouched" do the untrained rows look?
    W = model.thinking_block.depth_emb.weight.data
    print(f"[emb] trained rows 0..{T_MAX_TRAINED - 1}: "
          f"norm/row = {W[:T_MAX_TRAINED].norm(dim=1).mean():.3f}")
    print(f"[emb] untrained rows {T_MAX_TRAINED}..{W.shape[0] - 1}: "
          f"norm/row = {W[T_MAX_TRAINED:].norm(dim=1).mean():.3f}  "
          f"std = {W[T_MAX_TRAINED:].std():.3f}  "
          f"(random init ~ sqrt(d)={D_MODEL ** 0.5:.3f}, std ~ 1.0)")

    # -- Test loaders (fresh; larger than the paper run for stable means) --
    print("\n[data] generating OOD test buckets ...")
    test_loaders = {}
    for depth in TEST_DEPTHS:
        ds = FamilyReasoningDataset(
            n_samples=TEST_SAMPLES_PER, depths=[depth], max_len=MAX_SEQ_LEN
        )
        test_loaders[depth] = DataLoader(ds, batch_size=128, shuffle=False)
        print(f"       depth {depth}: {len(ds)} samples")

    run_embedding_ablation(
        model=model,
        evaluate_fn=evaluate,
        test_loaders=test_loaders,
        difficulties=TEST_DEPTHS,
        steps=TEST_STEPS,
        t_max_trained=T_MAX_TRAINED,
        id_difficulties=TEST_ID_DEPTHS,
        out_dir=_HERE,
        task_name="family",
        seed=SEED,
        difficulty_label="chain_depth",
        extra_meta={
            "checkpoint": "family_model.pt",
            "train_step_range": list(TRAIN_STEP_RANGE),
            "train_depths": TRAIN_DEPTHS,
            "test_samples_per_bucket": TEST_SAMPLES_PER,
        },
    )


if __name__ == "__main__":
    main()
