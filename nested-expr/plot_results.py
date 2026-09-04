"""Regenerate the nested boolean expression heatmap (PDF + PNG) from saved results."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_DEPTHS = [2, 4, 6, 8, 10, 12, 14]
TEST_STEPS  = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]

# Final-run accuracy grid (depth 1-8 training, d=256, RoPE + LayerScale).
# Written by logic_experiment.py as logic_grid_s42.npy and transcribed here so
# the styled PDF can be rebuilt without retraining. The paper's baseline table
# reports the 3-seed mean; this figure shows the seed-42 run.
results = np.array([
    [0.64, 0.83, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # depth=2
    [0.53, 0.75, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # depth=4
    [0.57, 0.68, 0.90, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00],  # depth=6
    [0.60, 0.72, 0.93, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],  # depth=8
    [0.56, 0.66, 0.88, 0.94, 0.95, 0.96, 0.96, 0.97, 0.97, 0.96],  # depth=10
    [0.58, 0.67, 0.87, 0.94, 0.96, 0.96, 0.96, 0.96, 0.95, 0.95],  # depth=12
    [0.57, 0.68, 0.86, 0.90, 0.92, 0.94, 0.93, 0.93, 0.92, 0.93],  # depth=14
])

D_MODEL          = 256
TRAIN_DEPTHS     = list(range(1, 9))  # 1-8
TRAIN_STEP_RANGE = (4, 16)
TEST_ID_DEPTHS   = [2, 4, 6, 8]

# Font sizes below are deliberately oversized: the figure is scaled to about 0.27x
# in a two-column layout, so at final size the on-page text lands near 8pt.
fig, ax = plt.subplots(figsize=(13, 8))
im = ax.imshow(results, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

ax.set_xticks(range(len(TEST_STEPS)))
ax.set_xticklabels(TEST_STEPS)
ax.set_yticks(range(len(TEST_DEPTHS)))
ax.set_yticklabels(TEST_DEPTHS)
ax.set_xlabel("Thinking Steps (Compute Depth)", fontsize=30)
ax.set_ylabel("Nesting Depth (Task Difficulty)", fontsize=30)
ax.tick_params(axis="both", labelsize=26)
# Horizontal line: depth ID/OOD boundary (task difficulty).
# Region labels sit in the right margin, outside the heatmap cells.
id_ood_boundary = len(TEST_ID_DEPTHS) - 0.5
ax.axhline(y=id_ood_boundary, color="white", linewidth=2.5, linestyle="--")
_yax = ax.get_yaxis_transform()
_id_center  = (-0.5 + id_ood_boundary) / 2
_ood_center = (id_ood_boundary + len(TEST_DEPTHS) - 0.5) / 2
ax.text(1.015, _id_center, "ID", transform=_yax, ha="left", va="center",
        fontsize=27, color="red", fontweight="bold", clip_on=False)
ax.text(1.015, _ood_center, "OOD", transform=_yax, ha="left", va="center",
        fontsize=27, color="red", fontweight="bold", clip_on=False)

# Vertical lines: step ID/OOD boundaries. Region labels sit above the heatmap.
step_left  = next(i for i, s in enumerate(TEST_STEPS) if s >= TRAIN_STEP_RANGE[0]) - 0.5
step_right = next(i for i, s in enumerate(TEST_STEPS) if s >  TRAIN_STEP_RANGE[1]) - 0.5
ax.axvline(x=step_left,  color="white", linewidth=2.5, linestyle="--")
ax.axvline(x=step_right, color="white", linewidth=2.5, linestyle="--")
_xax = ax.get_xaxis_transform()
ax.text(step_left / 2 - 0.25, 1.03, "OOD", transform=_xax, ha="center", va="bottom",
        fontsize=27, color="red", fontweight="bold", clip_on=False)
ax.text((step_left + step_right) / 2, 1.03, "ID", transform=_xax, ha="center", va="bottom",
        fontsize=27, color="red", fontweight="bold", clip_on=False)
ax.text((step_right + len(TEST_STEPS) - 0.5) / 2, 1.03, "OOD", transform=_xax,
        ha="center", va="bottom", fontsize=27, color="red", fontweight="bold", clip_on=False)

for i in range(len(TEST_DEPTHS)):
    for j in range(len(TEST_STEPS)):
        val = results[i, j]
        color = "white" if val < 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=25, color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
cbar.set_label("Accuracy", fontsize=30)
cbar.ax.tick_params(labelsize=26)

plt.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logic_results.pdf")
plt.savefig(out, dpi=150)
plt.savefig(out.replace(".pdf", ".png"), dpi=150)
print(f"Saved to {out} (+ .png)")
