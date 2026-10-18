import os
import json
import pandas as pd

ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"

LOG_ROOT = os.path.join(
    ROOT,
    "results/publication_results/logs"
)

FINAL_ROOT = os.path.join(
    ROOT,
    "results/publication_results/final_test"
)

EFF_ROOT = os.path.join(
    ROOT,
    "results/publication_results/energy_efficiency"
)

OUT_ROOT = os.path.join(
    ROOT,
    "results/publication_results/publication_tables"
)

os.makedirs(OUT_ROOT, exist_ok=True)

MODELS = ["A0", "A1", "A2", "A3"]


# ============================================================
# VALIDATION RESULTS
# ============================================================

validation = {}

for model in MODELS:

    path = os.path.join(
        LOG_ROOT,
        model,
        "training_history.json"
    )

    with open(path) as f:
        history = json.load(f)

    best = max(
        history,
        key=lambda x: x["patient_val_dice"]
    )

    validation[model] = {
        "best_val_epoch": best["epoch"],
        "best_val_dice": best["patient_val_dice"],
        "stopping_epoch": history[-1]["epoch"]
    }


# ============================================================
# FINAL TEST RESULTS
# ============================================================

ci_path = os.path.join(
    FINAL_ROOT,
    "bootstrap_95CI.csv"
)

ci_df = pd.read_csv(ci_path)

test_metrics = {}

for model in MODELS:

    test_metrics[model] = {}

    for metric in [
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_mm"
    ]:

        row = ci_df[
            (ci_df["model"] == model) &
            (ci_df["metric"] == metric)
        ]

        if len(row) == 0:
            raise ValueError(
                f"Missing {model}/{metric}"
            )

        row = row.iloc[0]

        test_metrics[model][metric] = {
            "mean": row["mean"],
            "lower": row["ci95_lower"],
            "upper": row["ci95_upper"],
            "n_valid": row["n_valid"]
        }


# ============================================================
# EFFICIENCY
# ============================================================

eff_path = os.path.join(
    EFF_ROOT,
    "energy_efficiency_summary.csv"
)

eff_df = pd.read_csv(eff_path)


# ============================================================
# BUILD MASTER TABLE
# ============================================================

rows = []

for model in MODELS:

    e = eff_df[
        eff_df["model"] == model
    ]

    if len(e) == 0:
        raise ValueError(
            f"Missing efficiency data for {model}"
        )

    e = e.iloc[0]

    row = {
        "model": model,

        "best_val_epoch":
            validation[model]["best_val_epoch"],

        "best_val_dice":
            validation[model]["best_val_dice"],

        "stopping_epoch":
            validation[model]["stopping_epoch"],

        "test_dice":
            test_metrics[model]["dice"]["mean"],

        "dice_ci_lower":
            test_metrics[model]["dice"]["lower"],

        "dice_ci_upper":
            test_metrics[model]["dice"]["upper"],

        "test_iou":
            test_metrics[model]["iou"]["mean"],

        "test_precision":
            test_metrics[model]["precision"]["mean"],

        "test_recall":
            test_metrics[model]["recall"]["mean"],

        "test_hd95_mm":
            test_metrics[model]["hd95_mm"]["mean"],

        "parameters":
            e["parameters"],

        "estimated_macs":
            e["estimated_macs"],

        "sop_proxy":
            e["sop_proxy"],

        "latency_mean_ms":
            e["latency_mean_ms"],

        "latency_median_ms":
            e["latency_median_ms"],

        "checkpoint_mb":
            e["checkpoint_mb"],

        "mean_firing_rate":
            e["mean_firing_rate"]
    }

    rows.append(row)


master = pd.DataFrame(rows)

out_path = os.path.join(
    OUT_ROOT,
    "MASTER_Publication_Results.csv"
)

master.to_csv(
    out_path,
    index=False,
    float_format="%.6f"
)

print("=" * 80)
print("MASTER PUBLICATION RESULTS")
print("=" * 80)

print(master.to_string(index=False))

print("\nSaved:")
print(out_path)

print("=" * 80)
