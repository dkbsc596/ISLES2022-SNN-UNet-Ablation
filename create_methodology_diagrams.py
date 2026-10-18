import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "results/publication_results/publication_figures/methodology"
os.makedirs(OUT_DIR, exist_ok=True)


def box(ax, x, y, w, h, text, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        fill=False
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.4
        )
    )


# ============================================================
# FIGURE 1
# Overall Experimental Methodology
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(
    6, 9.5,
    "Overall Experimental Methodology",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold"
)

# Row 1
box(ax, 0.5, 7.8, 2.2, 0.8, "ISLES-2022\n250 subjects")
box(ax, 3.1, 7.8, 2.2, 0.8, "Patient-level split\n174 / 38 / 38")
box(ax, 5.7, 7.8, 2.2, 0.8, "DWI + ADC + FLAIR\n3 input channels")
box(ax, 8.3, 7.8, 3.0, 0.8, "Preprocessing &\nspatial alignment")

arrow(ax, 2.7, 8.2, 3.1, 8.2)
arrow(ax, 5.3, 8.2, 5.7, 8.2)
arrow(ax, 7.9, 8.2, 8.3, 8.2)

# Row 2
box(ax, 1.0, 5.8, 2.5, 0.9, "3D patch extraction\n128 × 128 × 64")
box(ax, 4.0, 5.8, 2.5, 0.9, "Controlled models\nA0 / A1 / A2 / A3")
box(ax, 7.0, 5.8, 2.5, 0.9, "Patch-level\nprediction")
box(ax, 10.0, 5.8, 1.5, 0.9, "3D\nreconstruction")

arrow(ax, 9.8, 7.8, 2.25, 6.7)
arrow(ax, 3.5, 6.25, 4.0, 6.25)
arrow(ax, 6.5, 6.25, 7.0, 6.25)
arrow(ax, 9.5, 6.25, 10.0, 6.25)

# Row 3
box(ax, 1.0, 3.5, 2.5, 0.9, "Complete patient\nlesion volume")
box(ax, 4.0, 3.5, 2.5, 0.9, "Binary lesion\nmask")
box(ax, 7.0, 3.5, 2.5, 0.9, "Patient-level\nmetrics")
box(ax, 10.0, 3.5, 1.5, 0.9, "Statistical\nanalysis")

arrow(ax, 10.75, 5.8, 2.25, 4.4)
arrow(ax, 3.5, 3.95, 4.0, 3.95)
arrow(ax, 6.5, 3.95, 7.0, 3.95)
arrow(ax, 9.5, 3.95, 10.0, 3.95)

# Metrics
box(
    ax, 2.0, 1.1, 8.0, 1.0,
    "Dice  |  IoU  |  Precision  |  Recall  |  HD95\n"
    "Bootstrap 95% CI  |  Wilcoxon test  |  Holm correction  |  Effect size",
    fontsize=10
)

arrow(ax, 8.25, 3.5, 6.0, 2.1)

fig.savefig(
    os.path.join(OUT_DIR, "Figure_Methodology_1_Overall_Workflow.png"),
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(OUT_DIR, "Figure_Methodology_1_Overall_Workflow.pdf"),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 2
# Controlled A0-A3 Architecture
# ============================================================

fig, ax = plt.subplots(figsize=(13, 8))

ax.set_xlim(0, 13)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(
    6.5, 9.5,
    "Controlled 3D U-Net Architecture and Ablation Design",
    ha="center",
    fontsize=18,
    fontweight="bold"
)

# Main architecture
box(ax, 0.5, 7.0, 2.0, 1.0, "Input\nDWI + ADC + FLAIR\n3 × 128×128×64")
box(ax, 3.2, 7.0, 2.0, 1.0, "Encoder\n16 → 32 → 64 → 128")
box(ax, 6.0, 7.0, 2.0, 1.0, "Bottleneck\n128 channels")
box(ax, 8.8, 7.0, 2.0, 1.0, "Decoder\n128 → 64 → 32 → 16")
box(ax, 11.3, 7.0, 1.2, 1.0, "Output\n1 channel")

arrow(ax, 2.5, 7.5, 3.2, 7.5)
arrow(ax, 5.2, 7.5, 6.0, 7.5)
arrow(ax, 8.0, 7.5, 8.8, 7.5)
arrow(ax, 10.8, 7.5, 11.3, 7.5)

# Attention
box(
    ax, 4.0, 4.8, 2.5, 0.9,
    "Attention gates\nA2 + A3",
    fontsize=10
)

# Residual
box(
    ax, 7.5, 4.8, 2.5, 0.9,
    "Residual spiking blocks\nA3",
    fontsize=10
)

arrow(ax, 4.8, 7.0, 5.0, 5.7)
arrow(ax, 8.5, 7.0, 8.7, 5.7)

# Temporal
box(
    ax, 1.0, 3.0, 3.0, 0.9,
    "Temporal processing\nT = 4 time steps",
    fontsize=10
)

box(
    ax, 5.0, 3.0, 3.0, 0.9,
    "LIF neurons\nτ = 2.0",
    fontsize=10
)

box(
    ax, 9.0, 3.0, 3.0, 0.9,
    "Temporal mean\n→ lesion probability",
    fontsize=10
)

arrow(ax, 2.5, 4.8, 2.5, 3.9)
arrow(ax, 4.0, 3.45, 5.0, 3.45)
arrow(ax, 8.0, 3.45, 9.0, 3.45)

# Ablation table
ax.text(
    6.5, 2.25,
    "Controlled Ablation Configurations",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

headers = ["Model", "SNN", "Attention", "Residual"]

x_positions = [1.2, 4.0, 6.2, 8.7]
for x, h in zip(x_positions, headers):
    ax.text(x, 1.8, h, ha="center", fontsize=10, fontweight="bold")

rows = [
    ("A0", "No", "No", "No"),
    ("A1", "Yes", "No", "No"),
    ("A2", "Yes", "Yes", "No"),
    ("A3", "Yes", "Yes", "Yes"),
]

ys = [1.45, 1.15, 0.85, 0.55]

for y, row in zip(ys, rows):
    for x, value in zip(x_positions, row):
        ax.text(x, y, value, ha="center", fontsize=9)

fig.savefig(
    os.path.join(OUT_DIR, "Figure_Methodology_2_Architecture_Ablation.png"),
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(OUT_DIR, "Figure_Methodology_2_Architecture_Ablation.pdf"),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 3
# Patch-to-Patient Evaluation
# ============================================================

fig, ax = plt.subplots(figsize=(13, 7))

ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(
    6.5, 7.5,
    "Patch-to-Patient Reconstruction and Evaluation",
    ha="center",
    fontsize=18,
    fontweight="bold"
)

box(
    ax, 0.5, 5.4, 2.0, 1.0,
    "Patient MRI volume\nDWI + ADC + FLAIR"
)

box(
    ax, 3.2, 5.4, 2.0, 1.0,
    "3D patches\n128 × 128 × 64"
)

box(
    ax, 5.9, 5.4, 2.0, 1.0,
    "Model prediction\n1-channel mask"
)

box(
    ax, 8.6, 5.4, 2.0, 1.0,
    "Patch aggregation\n& reconstruction"
)

box(
    ax, 11.2, 5.4, 1.3, 1.0,
    "Patient\nvolume"
)

arrow(ax, 2.5, 5.9, 3.2, 5.9)
arrow(ax, 5.2, 5.9, 5.9, 5.9)
arrow(ax, 7.9, 5.9, 8.6, 5.9)
arrow(ax, 10.6, 5.9, 11.2, 5.9)

# Ground truth
box(
    ax, 3.0, 2.8, 2.5, 1.0,
    "Ground-truth\nlesion volume"
)

box(
    ax, 6.0, 2.8, 2.5, 1.0,
    "Predicted\nlesion volume"
)

box(
    ax, 9.0, 2.8, 3.0, 1.0,
    "Patient-level comparison\nPrediction vs. ground truth"
)

arrow(ax, 11.85, 5.4, 7.25, 3.8)
arrow(ax, 5.5, 3.3, 6.0, 3.3)
arrow(ax, 8.5, 3.3, 9.0, 3.3)

# Metrics
box(
    ax, 2.0, 0.7, 9.0, 1.0,
    "Dice  |  IoU  |  Precision  |  Recall  |  HD95\n"
    "38 held-out patients  |  Bootstrap 95% CI  |  Paired statistical comparison",
    fontsize=10
)

arrow(ax, 10.5, 2.8, 6.5, 1.7)

fig.savefig(
    os.path.join(OUT_DIR, "Figure_Methodology_3_Patient_Level_Evaluation.png"),
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(OUT_DIR, "Figure_Methodology_3_Patient_Level_Evaluation.pdf"),
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 70)
print("METHODOLOGY FIGURES CREATED")
print("=" * 70)

for filename in sorted(os.listdir(OUT_DIR)):
    print(filename)

print("=" * 70)
