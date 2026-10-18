import numpy as np, pandas as pd, os
from scipy.stats import wilcoxon, rankdata

SRC = "results/publication_results/final_test/patient_level_results.csv"
OUT = "results/publication_results/sensitivity_no_empty"
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(SRC)
d = d[d.gt_voxels > 0]
print("patients kept per model:", d.groupby("model").subject.nunique().to_dict())

metrics = ["dice", "iou", "precision", "recall", "hd95_mm"]
pairs = [("A0", "A1"), ("A1", "A2"), ("A2", "A3"), ("A0", "A3")]
rng = np.random.default_rng(42)
rows = []
for a, b in pairs:
    for m in metrics:
        w = d.pivot(index="subject", columns="model", values=m)[[a, b]].dropna()
        diff = (w[b] - w[a]).to_numpy()
        nz = diff[diff != 0]
        W, p = wilcoxon(nz, alternative="two-sided", method="approx")
        r = rankdata(np.abs(nz))
        wp, wn = r[nz > 0].sum(), r[nz < 0].sum()
        boots = rng.choice(diff, (10000, len(diff)), replace=True).mean(axis=1)
        rows.append(dict(comparison=f"{a}_vs_{b}", metric=m, n_paired=len(diff),
                         n_nonzero=len(nz), mean_diff_B_minus_A=diff.mean(),
                         ci95_lower=np.percentile(boots, 2.5),
                         ci95_upper=np.percentile(boots, 97.5),
                         wilcoxon_statistic=W, p_value=p,
                         rank_biserial=(wp - wn) / (wp + wn)))
t = pd.DataFrame(rows)
order = np.argsort(t.p_value.to_numpy())
m_ = len(t)
adj = np.empty(m_)
run = 0.0
for i, idx in enumerate(order):
    run = max(run, (m_ - i) * t.p_value.iloc[idx])
    adj[idx] = min(1.0, run)
t["p_holm"] = adj
t["significant_holm"] = t.p_holm <= 0.05
t.to_csv(os.path.join(OUT, "effect_sizes_no_empty.csv"), index=False)
pd.set_option("display.width", 200)
print(t[["comparison", "metric", "n_paired", "mean_diff_B_minus_A", "p_value",
         "p_holm", "rank_biserial", "significant_holm"]].round(4).to_string(index=False))
print("\nmean per model (no empty):")
print(d.groupby("model")[metrics].mean().round(4).to_string())
