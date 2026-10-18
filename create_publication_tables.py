import pandas as pd
import json
from pathlib import Path

ROOT = Path("/home/krishna_19622/dataset/stroke_3d_snn1_xai")
SRC = ROOT / "results/publication_results"
OUT = SRC / "publication_tables"

OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# TABLE 1: FINAL PATIENT-LEVEL PERFORMANCE
# ------------------------------------------------------------
ci = pd.read_csv(SRC / "final_test/bootstrap_95CI.csv")

dice = ci[ci.metric == "dice"].copy()

table1 = dice[
    ["model", "n_valid", "mean", "ci95_lower", "ci95_upper"]
].copy()

table1.columns = [
    "Model", "N", "Dice", "95% CI Lower", "95% CI Upper"
]

for c in ["Dice", "95% CI Lower", "95% CI Upper"]:
    table1[c] = table1[c].round(4)

table1.to_csv(OUT / "Table_1_Dice_95CI.csv", index=False)


# ------------------------------------------------------------
# TABLE 2: COMPLETE PATIENT-LEVEL METRICS
# ------------------------------------------------------------
metrics = ci.pivot(
    index="model",
    columns="metric",
    values="mean"
).reset_index()

metrics.columns.name = None

rename = {
    "model": "Model",
    "dice": "Dice",
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
    "hd95_mm": "HD95 (mm)"
}

metrics = metrics.rename(columns=rename)

for c in metrics.columns:
    if c != "Model":
        metrics[c] = metrics[c].round(4)

metrics.to_csv(OUT / "Table_2_Patient_Level_Metrics.csv", index=False)


# ------------------------------------------------------------
# TABLE 3: 95% CONFIDENCE INTERVALS
# ------------------------------------------------------------
ci_table = ci.copy()

ci_table = ci_table.rename(columns={
    "model": "Model",
    "metric": "Metric",
    "n_valid": "N",
    "mean": "Mean",
    "ci95_lower": "95% CI Lower",
    "ci95_upper": "95% CI Upper"
})

for c in ["Mean", "95% CI Lower", "95% CI Upper"]:
    ci_table[c] = ci_table[c].round(4)

ci_table.to_csv(OUT / "Table_3_All_95CI.csv", index=False)


# ------------------------------------------------------------
# TABLE 4: ABLATION STATISTICS
# ------------------------------------------------------------
stats = pd.read_csv(SRC / "final_test/statistical_analysis.csv")

stats_table = stats.rename(columns={
    "comparison": "Comparison",
    "metric": "Metric",
    "n": "N",
    "median_A": "Median A",
    "median_B": "Median B",
    "median_difference_B_minus_A": "Median Δ",
    "mean_difference_B_minus_A": "Mean Δ",
    "wilcoxon_statistic": "Wilcoxon W",
    "p_value": "p-value",
    "p_holm": "Holm-adjusted p",
    "significant_holm": "Significant"
})

for c in ["Median A", "Median B", "Median Δ", "Mean Δ",
          "Wilcoxon W", "p-value", "Holm-adjusted p"]:
    if c in stats_table.columns:
        stats_table[c] = stats_table[c].round(6)

stats_table.to_csv(
    OUT / "Table_4_Ablation_Statistics.csv",
    index=False
)


# ------------------------------------------------------------
# TABLE 5: EFFECT SIZES
# ------------------------------------------------------------
effects = pd.read_csv(SRC / "final_test/effect_sizes.csv")

effects_table = effects.rename(columns={
    "comparison": "Comparison",
    "model_A": "Model A",
    "model_B": "Model B",
    "metric": "Metric",
    "n_paired": "Paired N",
    "mean_difference_B_minus_A": "Mean Δ",
    "rank_biserial_effect": "Rank-biserial Effect",
    "p_holm": "Holm-adjusted p",
    "significant_holm": "Significant"
})

keep = [
    "Comparison",
    "Model A",
    "Model B",
    "Metric",
    "Paired N",
    "Mean Δ",
    "Rank-biserial Effect",
    "Holm-adjusted p",
    "Significant"
]

effects_table = effects_table[keep]

for c in ["Mean Δ", "Rank-biserial Effect", "Holm-adjusted p"]:
    effects_table[c] = effects_table[c].round(6)

effects_table.to_csv(
    OUT / "Table_5_Effect_Sizes.csv",
    index=False
)


# ------------------------------------------------------------
# TABLE 6: COMPUTATIONAL / ENERGY EFFICIENCY
# ------------------------------------------------------------
energy = pd.read_csv(
    SRC / "energy_efficiency/energy_efficiency_summary.csv"
)

energy_table = energy.rename(columns={
    "model": "Model",
    "description": "Architecture",
    "parameters": "Parameters",
    "trainable_parameters": "Trainable Parameters",
    "checkpoint_mb": "Checkpoint (MB)",
    "estimated_macs": "Estimated MACs",
    "mean_firing_rate": "Mean Firing Rate",
    "total_spikes": "Total Spikes",
    "sop_proxy": "SOP Proxy",
    "relative_macs_vs_A0": "Relative MACs vs A0",
    "relative_sop_proxy_vs_A0": "Relative SOP vs A0",
    "latency_mean_ms": "Mean Latency (ms)",
    "latency_median_ms": "Median Latency (ms)",
    "gpu_peak_allocated_mb": "Peak GPU Allocated (MB)",
    "gpu_peak_reserved_mb": "Peak GPU Reserved (MB)",
    "relative_latency_vs_A0": "Relative Latency vs A0"
})

keep = [
    "Model",
    "Architecture",
    "Parameters",
    "Checkpoint (MB)",
    "Estimated MACs",
    "Mean Firing Rate",
    "Total Spikes",
    "SOP Proxy",
    "Relative MACs vs A0",
    "Relative SOP vs A0",
    "Mean Latency (ms)",
    "Relative Latency vs A0"
]

energy_table = energy_table[keep]

energy_table.to_csv(
    OUT / "Table_6_Computational_Efficiency.csv",
    index=False
)


print("=" * 70)
print("PUBLICATION TABLE GENERATION COMPLETE")
print("=" * 70)

for f in sorted(OUT.glob("*.csv")):
    print(f"  {f}")
