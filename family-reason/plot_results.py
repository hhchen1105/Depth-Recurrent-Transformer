"""Regenerate the family relationship reasoning heatmap (PDF + PNG) from saved results."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_DEPTHS    = [2, 3, 4, 5]
TEST_ID_DEPTHS  = [2, 3, 4, 5]
TEST_OOD_DEPTHS = [6, 7, 8, 9]
TEST_DEPTHS     = TEST_ID_DEPTHS + TEST_OOD_DEPTHS
TEST_STEPS      = [1, 2, 4, 6, 8, 10, 12, 16, 20]

# Values from extended evaluation (test steps up to 20)
results = np.array([
    [0.801, 0.856, 0.858, 0.861, 0.861, 0.863, 0.862, 0.860, 0.861],  # depth=2
    [0.724, 0.834, 0.839, 0.833, 0.835, 0.834, 0.833, 0.829, 0.825],  # depth=3
    [0.658, 0.811, 0.833, 0.835, 0.827, 0.824, 0.826, 0.824, 0.822],  # depth=4
    [0.638, 0.791, 0.812, 0.819, 0.818, 0.819, 0.817, 0.806, 0.806],  # depth=5
    [0.536, 0.661, 0.682, 0.687, 0.688, 0.690, 0.691, 0.691, 0.687],  # depth=6
    [0.527, 0.632, 0.651, 0.657, 0.665, 0.667, 0.671, 0.671, 0.674],  # depth=7
    [0.460, 0.507, 0.524, 0.530, 0.543, 0.550, 0.552, 0.568, 0.571],  # depth=8
    [0.448, 0.492, 0.518, 0.525, 0.531, 0.535, 0.543, 0.548, 0.557],  # depth=9
])

D_MODEL          = 256
TRAIN_STEP_RANGE = (1, 12)

# Font sizes below are deliberately oversized: the figure is scaled to about 0.35x
# in a two-column layout, so at final size the on-page text lands near 8pt.
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(results, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

ax.set_xticks(range(len(TEST_STEPS)))
ax.set_xticklabels(TEST_STEPS)
ax.set_yticks(range(len(TEST_DEPTHS)))
ax.set_yticklabels(TEST_DEPTHS)
ax.tick_params(axis="both", labelsize=20)
ax.set_xlabel("Thinking Steps (Compute Depth)", fontsize=23)
ax.set_ylabel("Chain Depth (Number of Hops)", fontsize=23)
# Horizontal line: depth ID/OOD boundary (task difficulty).
# Region labels sit in the right margin, outside the heatmap cells.
id_ood_boundary = len(TEST_ID_DEPTHS) - 0.5
ax.axhline(y=id_ood_boundary, color="white", linewidth=2.5, linestyle="--")
_yax = ax.get_yaxis_transform()
_id_center  = (-0.5 + id_ood_boundary) / 2
_ood_center = (id_ood_boundary + len(TEST_DEPTHS) - 0.5) / 2
ax.text(1.015, _id_center, "ID", transform=_yax, ha="left", va="center",
        fontsize=21, color="red", fontweight="bold", clip_on=False)
ax.text(1.015, _ood_center, "OOD", transform=_yax, ha="left", va="center",
        fontsize=21, color="red", fontweight="bold", clip_on=False)

# Vertical line: step ID/OOD boundary (only right; training starts at step 1).
# Region labels sit above the heatmap.
step_right = next(i for i, s in enumerate(TEST_STEPS) if s > TRAIN_STEP_RANGE[1]) - 0.5
ax.axvline(x=step_right, color="white", linewidth=2.5, linestyle="--")
_xax = ax.get_xaxis_transform()
ax.text((step_right - 0.5) / 2, 1.03, "ID", transform=_xax, ha="center", va="bottom",
        fontsize=21, color="red", fontweight="bold", clip_on=False)
ax.text((step_right + len(TEST_STEPS) - 0.5) / 2, 1.03, "OOD", transform=_xax,
        ha="center", va="bottom", fontsize=20, color="red", fontweight="bold", clip_on=False)

for i in range(len(TEST_DEPTHS)):
    for j in range(len(TEST_STEPS)):
        val = results[i, j]
        color = "white" if val < 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=19, color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.10)
cbar.set_label("Accuracy", fontsize=23)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "family_results.pdf")
plt.savefig(out, dpi=150)
plt.savefig(out.replace(".pdf", ".png"), dpi=150)
print(f"Saved to {out} (+ .png)")
