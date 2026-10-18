#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"

FINAL_TEST = os.path.join(
    ROOT, "results/publication_results/final_test"
)

ENERGY = os.path.join(
    ROOT, "results/publication_results/energy_efficiency"
)

OUT = os.path.join(
    ROOT, "results/publication_results/publication_figures"
)

os.makedirs(OUT, exist_ok=True)

MODEL_LABELS = {
    "A0": "3D U-Net",
    "A1": "3D SNN-U-Net",
    "A2": "3D Attention SNN-U-Net",
    "A3": "3D Attention-Residual SNN-U-Net",
}

MODEL_ORDER = ["A0", "A1", "A2", "A3"]


def load_patient_results():
    path = os.path.join(
        FINAL_TEST, "patient_level_results.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    print("\nPatient-level results:")
    print("  File:", path)
    print("  Rows:", len(df))
    print("  Columns:", list(df.columns))

    return df


def identify_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    # Partial matching
    for c in df.columns:
        cl = c.lower()
        for candidate in candidates:
            if candidate.lower() in cl:
                return c

    return None


def normalize_model_column(df):
    model_col = identify_column(
        df,
        ["model", "configuration", "experiment"]
    )

    if model_col is None:
        raise ValueError(
            "Could not identify model column. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    def normalize(x):
        x = str(x).strip()

        if x in MODEL_LABELS:
            return x

        # Publication labels
        for key, label in MODEL_LABELS.items():
            if x == label:
                return key

        # Common descriptive variants
        xl = x.lower()

        if "attention" in xl and "residual" in xl:
            return "A3"
        if "attention" in xl:
            return "A2"
        if "snn" in xl:
            return "A1"
        if "unet" in xl or "u-net" in xl:
            return "A0"

        return x

    df["_model"] = df[model_col].map(normalize)

    return df


def save_figure(fig, basename):
    png = os.path.join(OUT, basename + ".png")
    pdf = os.path.join(OUT, basename + ".pdf")

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# ============================================================
# LOAD DATA
# ============================================================

df = load_patient_results()
df = normalize_model_column(df)

dice_col = identify_column(
    df,
    ["dice", "test_dice", "patient_dice"]
)

hd95_col = identify_column(
    df,
    ["hd95", "test_hd95", "hd95_mm", "patient_hd95"]
)

subject_col = identify_column(
    df,
    ["subject", "subject_id", "patient", "patient_id"]
)

if dice_col is None:
    raise ValueError(
        f"Dice column not found. Columns: {list(df.columns)}"
    )

if hd95_col is None:
    raise ValueError(
        f"HD95 column not found. Columns: {list(df.columns)}"
    )

print("\nDetected columns:")
print("  Model :", "_model")
print("  Dice  :", dice_col)
print("  HD95  :", hd95_col)
print("  Subject:", subject_col)


# ============================================================
# VALIDATE DATA
# ============================================================

for model in MODEL_ORDER:
    sub = df[df["_model"] == model]

    print(
        f"{model} | "
        f"{MODEL_LABELS[model]} | "
        f"n={len(sub)} | "
        f"Dice mean={sub[dice_col].mean():.6f} | "
        f"HD95 mean={sub[hd95_col].mean():.6f}"
    )

    if len(sub) != 38:
        print(
            f"WARNING: expected 38 patients for {model}, "
            f"found {len(sub)}"
        )


# ============================================================
# FIGURE 7
# PATIENT-LEVEL DICE DISTRIBUTION
# ============================================================

dice_data = [
    df[df["_model"] == model][dice_col]
    .dropna()
    .astype(float)
    .values
    for model in MODEL_ORDER
]

fig, ax = plt.subplots(
    figsize=(11, 6.5)
)

positions = np.arange(1, 5)

bp = ax.boxplot(
    dice_data,
    positions=positions,
    widths=0.55,
    patch_artist=False,
    showmeans=True,
    meanline=False
)

# Individual patient points
rng = np.random.default_rng(42)

for i, values in enumerate(dice_data, start=1):
    jitter = rng.uniform(
        -0.09,
        0.09,
        size=len(values)
    )

    ax.scatter(
        np.full(len(values), i) + jitter,
        values,
        s=18,
        alpha=0.55
    )

ax.set_xticks(positions)
ax.set_xticklabels(
    [MODEL_LABELS[m] for m in MODEL_ORDER],
    rotation=15,
    ha="right"
)

ax.set_ylabel("Patient-level Dice coefficient")
ax.set_xlabel("Model")
ax.set_title(
    "Patient-level Dice Distribution on the Held-out Test Set"
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.set_ylim(
    max(0, min(np.min(v) for v in dice_data) - 0.05),
    min(1.0, max(np.max(v) for v in dice_data) + 0.05)
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_7_Patient_Level_Dice_Distribution"
)


# ============================================================
# SAVE FIGURE 7 DATA
# ============================================================

dice_rows = []

for model in MODEL_ORDER:
    values = df[
        df["_model"] == model
    ][dice_col].dropna().astype(float)

    dice_rows.append({
        "model": model,
        "publication_model": MODEL_LABELS[model],
        "n": len(values),
        "mean_dice": values.mean(),
        "median_dice": values.median(),
        "std_dice": values.std(ddof=1),
        "q1_dice": values.quantile(0.25),
        "q3_dice": values.quantile(0.75),
        "min_dice": values.min(),
        "max_dice": values.max(),
    })

pd.DataFrame(dice_rows).to_csv(
    os.path.join(
        OUT,
        "Figure_7_Patient_Level_Dice_Summary.csv"
    ),
    index=False
)


# ============================================================
# FIGURE 8
# PATIENT-LEVEL HD95 DISTRIBUTION
# ============================================================

hd95_data = [
    df[df["_model"] == model][hd95_col]
    .dropna()
    .astype(float)
    .values
    for model in MODEL_ORDER
]

fig, ax = plt.subplots(
    figsize=(11, 6.5)
)

bp = ax.boxplot(
    hd95_data,
    positions=positions,
    widths=0.55,
    patch_artist=False,
    showmeans=True,
    meanline=False
)

rng = np.random.default_rng(42)

for i, values in enumerate(hd95_data, start=1):
    jitter = rng.uniform(
        -0.09,
        0.09,
        size=len(values)
    )

    ax.scatter(
        np.full(len(values), i) + jitter,
        values,
        s=18,
        alpha=0.55
    )

ax.set_xticks(positions)
ax.set_xticklabels(
    [MODEL_LABELS[m] for m in MODEL_ORDER],
    rotation=15,
    ha="right"
)

ax.set_ylabel("HD95 (mm)")
ax.set_xlabel("Model")
ax.set_title(
    "Patient-level HD95 Distribution on the Held-out Test Set"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_8_Patient_Level_HD95_Distribution"
)


# ============================================================
# SAVE FIGURE 8 DATA
# ============================================================

hd95_rows = []

for model in MODEL_ORDER:
    values = df[
        df["_model"] == model
    ][hd95_col].dropna().astype(float)

    hd95_rows.append({
        "model": model,
        "publication_model": MODEL_LABELS[model],
        "n": len(values),
        "mean_hd95_mm": values.mean(),
        "median_hd95_mm": values.median(),
        "std_hd95_mm": values.std(ddof=1),
        "q1_hd95_mm": values.quantile(0.25),
        "q3_hd95_mm": values.quantile(0.75),
        "min_hd95_mm": values.min(),
        "max_hd95_mm": values.max(),
    })

pd.DataFrame(hd95_rows).to_csv(
    os.path.join(
        OUT,
        "Figure_8_Patient_Level_HD95_Summary.csv"
    ),
    index=False
)


# ============================================================
# FIGURE 9
# ACCURACY VS SOP PROXY
# ============================================================

energy_path = os.path.join(
    ENERGY,
    "energy_efficiency_summary.csv"
)

if not os.path.exists(energy_path):
    raise FileNotFoundError(energy_path)

energy = pd.read_csv(energy_path)

energy_model_col = identify_column(
    energy,
    ["model"]
)

sop_col = identify_column(
    energy,
    ["sop_proxy"]
)

latency_col = identify_column(
    energy,
    ["latency_mean_ms"]
)

params_col = identify_column(
    energy,
    ["parameters"]
)

if energy_model_col is None or sop_col is None:
    raise ValueError(
        "Required efficiency columns not found. "
        f"Columns: {list(energy.columns)}"
    )

# Mean test Dice per model
dice_summary = (
    df.groupby("_model")[dice_col]
    .mean()
    .to_dict()
)

eff_rows = []

for model in MODEL_ORDER:
    match = energy[
        energy[energy_model_col].astype(str).str.strip() == model
    ]

    if len(match) != 1:
        raise ValueError(
            f"Expected one efficiency row for {model}, "
            f"found {len(match)}"
        )

    row = match.iloc[0]

    eff_rows.append({
        "model": model,
        "publication_model": MODEL_LABELS[model],
        "test_dice": dice_summary[model],
        "sop_proxy": float(row[sop_col]),
        "latency_mean_ms": (
            float(row[latency_col])
            if latency_col is not None
            else np.nan
        ),
        "parameters": (
            int(row[params_col])
            if params_col is not None
            else np.nan
        ),
    })

eff_df = pd.DataFrame(eff_rows)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

for _, row in eff_df.iterrows():

    ax.scatter(
        row["sop_proxy"] / 1e9,
        row["test_dice"],
        s=90
    )

    ax.annotate(
        row["publication_model"],
        (
            row["sop_proxy"] / 1e9,
            row["test_dice"]
        ),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=9
    )

ax.set_xlabel(
    "SOP proxy (billions of equivalent operations)"
)

ax.set_ylabel(
    "Mean patient-level test Dice"
)

ax.set_title(
    "Segmentation Accuracy versus Spike-based Computational Proxy"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_9_Dice_vs_SOP_Proxy"
)

eff_df.to_csv(
    os.path.join(
        OUT,
        "Figure_9_Dice_vs_SOP_Proxy_Data.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ADDITIONAL PUBLICATION FIGURES COMPLETE")
print("=" * 70)

for name in [
    "Figure_7_Patient_Level_Dice_Distribution",
    "Figure_8_Patient_Level_HD95_Distribution",
    "Figure_9_Dice_vs_SOP_Proxy",
]:
    print(f"PASS: {name}.png")
    print(f"PASS: {name}.pdf")

print("\nNo checkpoints were modified.")
print("No training was performed.")
print("No prediction files were modified.")
print("Only publication figures and summary CSV files were created.")

