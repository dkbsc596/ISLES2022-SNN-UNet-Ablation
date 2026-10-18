import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"

LOG_ROOT = os.path.join(
    ROOT, "results", "publication_results", "logs"
)

FINAL_ROOT = os.path.join(
    ROOT, "results", "publication_results", "final_test"
)

EFF_ROOT = os.path.join(
    ROOT, "results", "publication_results", "energy_efficiency"
)

FIG_ROOT = os.path.join(
    ROOT, "results", "publication_results",
    "publication_figures"
)

os.makedirs(FIG_ROOT, exist_ok=True)

MODELS = ["A0", "A1", "A2", "A3"]

MODEL_LABELS = {
    "A0": "A0: 3D CNN U-Net",
    "A1": "A1: 3D SNN U-Net",
    "A2": "A2: SNN U-Net + Attention",
    "A3": "A3: SNN U-Net + Attention + Residual",
}


# ============================================================
# LOAD TRAINING HISTORIES
# ============================================================

histories = {}

for model in MODELS:

    path = os.path.join(
        LOG_ROOT,
        model,
        "training_history.json"
    )

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{model}: training history is not a list"
        )

    histories[model] = data

    print(
        f"{model}: loaded {len(data)} epochs"
    )


# ============================================================
# FIGURE 1
# TRAINING CONVERGENCE
# ============================================================

plt.figure(figsize=(10, 6))

for model in MODELS:

    history = histories[model]

    epochs = [
        int(row["epoch"])
        for row in history
        if "epoch" in row and
        "patient_val_dice" in row
    ]

    dice = [
        float(row["patient_val_dice"])
        for row in history
        if "epoch" in row and
        "patient_val_dice" in row
    ]

    plt.plot(
        epochs,
        dice,
        linewidth=2,
        label=MODEL_LABELS[model]
    )

plt.xlabel("Epoch")
plt.ylabel("Patient-level Validation Dice")
plt.title("Training Convergence of Controlled 3D Models")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = os.path.join(
    FIG_ROOT,
    "Figure_1_Training_Convergence.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# ============================================================
# LOAD BOOTSTRAP RESULTS
# ============================================================

ci_path = os.path.join(
    FINAL_ROOT,
    "bootstrap_95CI.csv"
)

ci_df = pd.read_csv(ci_path)

print("\nBootstrap columns:")
print(list(ci_df.columns))


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_ci(model, metric):

    rows = ci_df[
        (ci_df["model"] == model) &
        (ci_df["metric"] == metric)
    ]

    if len(rows) == 0:
        raise ValueError(
            f"No CI data found for "
            f"{model} / {metric}"
        )

    row = rows.iloc[0]

    return (
        float(row["mean"]),
        float(row["ci95_lower"]),
        float(row["ci95_upper"])
    )


# ============================================================
# FIGURE 2
# DICE WITH 95% CI
# ============================================================

means = []
lower_errors = []
upper_errors = []

for model in MODELS:

    mean, lower, upper = get_ci(
        model,
        "dice"
    )

    means.append(mean)
    lower_errors.append(mean - lower)
    upper_errors.append(upper - mean)

x = np.arange(len(MODELS))

plt.figure(figsize=(9, 6))

plt.errorbar(
    x,
    means,
    yerr=[
        lower_errors,
        upper_errors
    ],
    fmt="o",
    capsize=6,
    markersize=8,
    linewidth=2
)

plt.xticks(
    x,
    MODELS
)

plt.ylabel("Patient-level Dice")
plt.xlabel("Model")

plt.title(
    "Held-out Test Dice with 95% Bootstrap Confidence Intervals"
)

plt.ylim(0.5, 0.9)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

for i, value in enumerate(means):

    plt.text(
        i,
        value + 0.025,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

path = os.path.join(
    FIG_ROOT,
    "Figure_2_Dice_95CI.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# ============================================================
# FIGURE 3
# MULTI-METRIC COMPARISON
# ============================================================

metric_names = [
    ("dice", "Dice"),
    ("iou", "IoU"),
    ("precision", "Precision"),
    ("recall", "Recall")
]

x = np.arange(len(MODELS))

width = 0.18

plt.figure(figsize=(11, 6))

for i, (metric_key, metric_label) in enumerate(
    metric_names
):

    values = [
        get_ci(model, metric_key)[0]
        for model in MODELS
    ]

    offset = (i - 1.5) * width

    plt.bar(
        x + offset,
        values,
        width,
        label=metric_label
    )

plt.xticks(
    x,
    MODELS
)

plt.ylabel("Metric Value")
plt.xlabel("Model")

plt.title(
    "Held-out Test Patient-level Performance"
)

plt.ylim(0, 1.0)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

path = os.path.join(
    FIG_ROOT,
    "Figure_3_MultiMetric_Comparison.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# ============================================================
# FIGURE 4
# HD95
# ============================================================

hd95_values = []
hd95_lower = []
hd95_upper = []

for model in MODELS:

    mean, lower, upper = get_ci(
        model,
        "hd95_mm"
    )

    hd95_values.append(mean)
    hd95_lower.append(mean - lower)
    hd95_upper.append(upper - mean)

x = np.arange(len(MODELS))

plt.figure(figsize=(9, 6))

plt.errorbar(
    x,
    hd95_values,
    yerr=[
        hd95_lower,
        hd95_upper
    ],
    fmt="o",
    capsize=6,
    markersize=8,
    linewidth=2
)

plt.xticks(
    x,
    MODELS
)

plt.ylabel("HD95 (mm)")
plt.xlabel("Model")

plt.title(
    "Held-out Test HD95 with 95% Bootstrap Confidence Intervals"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

for i, value in enumerate(hd95_values):

    plt.text(
        i,
        value + 0.8,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()

path = os.path.join(
    FIG_ROOT,
    "Figure_4_HD95_Comparison.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# ============================================================
# LOAD COMPUTATIONAL EFFICIENCY
# ============================================================

eff_path = os.path.join(
    EFF_ROOT,
    "energy_efficiency_summary.csv"
)

eff_df = pd.read_csv(eff_path)

print("\nEfficiency columns:")
print(list(eff_df.columns))


# ============================================================
# FIND LATENCY COLUMN
# ============================================================

latency_candidates = [
    "latency_mean_ms",
    "mean_latency_ms",
    "latency_ms",
    "mean_latency",
    "latency"
]

latency_column = None

for col in latency_candidates:

    if col in eff_df.columns:

        latency_column = col
        break

if latency_column is None:

    raise ValueError(
        "Could not find latency column. "
        "Available columns: "
        + str(list(eff_df.columns))
    )


# ============================================================
# FIND MODEL COLUMN
# ============================================================

if "model" not in eff_df.columns:

    raise ValueError(
        "Efficiency CSV does not contain "
        "'model' column."
    )


# ============================================================
# FIGURE 5
# ACCURACY VS LATENCY
# ============================================================

dice_values = []
latency_values = []

for model in MODELS:

    dice = get_ci(
        model,
        "dice"
    )[0]

    row = eff_df[
        eff_df["model"] == model
    ]

    if len(row) == 0:

        raise ValueError(
            f"No efficiency data found for {model}"
        )

    row = row.iloc[0]

    latency = float(
        row[latency_column]
    )

    dice_values.append(dice)
    latency_values.append(latency)


plt.figure(figsize=(9, 6))

plt.scatter(
    latency_values,
    dice_values,
    s=100
)

for i, model in enumerate(MODELS):

    plt.annotate(
        model,
        (
            latency_values[i],
            dice_values[i]
        ),
        xytext=(8, 8),
        textcoords="offset points"
    )

plt.xlabel(
    "Mean Inference Latency (ms)"
)

plt.ylabel(
    "Held-out Test Dice"
)

plt.title(
    "Accuracy–Latency Trade-off"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    FIG_ROOT,
    "Figure_5_Accuracy_vs_Latency.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# ============================================================
# FINAL FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PUBLICATION FIGURE GENERATION COMPLETE")
print("=" * 70)

for filename in sorted(
    os.listdir(FIG_ROOT)
):

    full_path = os.path.join(
        FIG_ROOT,
        filename
    )

    if os.path.isfile(full_path):

        size_kb = (
            os.path.getsize(full_path)
            / 1024
        )

        print(
            f"{filename:45s} "
            f"{size_kb:10.1f} KB"
        )

print("=" * 70)

print(
    "Output directory:"
)

print(FIG_ROOT)

print("=" * 70)
