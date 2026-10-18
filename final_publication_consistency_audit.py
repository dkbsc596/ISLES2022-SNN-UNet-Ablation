import os
import json
import pandas as pd
import numpy as np

ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"

PUB = os.path.join(ROOT, "results/publication_results")
FINAL = os.path.join(PUB, "final_test")
TABLES = os.path.join(PUB, "publication_tables")
EFF = os.path.join(PUB, "energy_efficiency")
FIG = os.path.join(PUB, "publication_figures")
LOGS = os.path.join(PUB, "logs")

MODELS = ["A0", "A1", "A2", "A3"]

LABELS = {
    "A0": "3D U-Net",
    "A1": "3D SNN-U-Net",
    "A2": "3D Attention SNN-U-Net",
    "A3": "3D Attention-Residual SNN-U-Net",
}

EXPECTED = {
    "A0": {
        "val_dice": 0.765857,
        "best_epoch": 50,
        "stop_epoch": 50,
        "test_dice": 0.702990,
        "iou": 0.594231,
        "hd95": 12.606891,
    },
    "A1": {
        "val_dice": 0.767176,
        "best_epoch": 28,
        "stop_epoch": 43,
        "test_dice": 0.736311,
        "iou": 0.620126,
        "hd95": 14.025572,
    },
    "A2": {
        "val_dice": 0.746701,
        "best_epoch": 48,
        "stop_epoch": 50,
        "test_dice": 0.667856,
        "iou": 0.559644,
        "hd95": 14.161381,
    },
    "A3": {
        "val_dice": 0.752607,
        "best_epoch": 14,
        "stop_epoch": 29,
        "test_dice": 0.740625,
        "iou": 0.628876,
        "hd95": 12.274104,
    },
}

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL

    if condition:
        print(f"PASS: {name}")
        PASS += 1
    else:
        print(f"FAIL: {name}")
        if detail:
            print("      " + detail)
        FAIL += 1


print("=" * 78)
print("FINAL PUBLICATION CONSISTENCY AUDIT")
print("=" * 78)


# ============================================================
# 1. PATIENT-LEVEL RESULTS
# ============================================================

patient_path = os.path.join(
    FINAL,
    "patient_level_results.csv"
)

patients = pd.read_csv(patient_path)

print("\n[1] PATIENT-LEVEL RESULTS")
print("Rows:", len(patients))

check(
    "Patient result count = 152",
    len(patients) == 152
)

for model in MODELS:
    sub = patients[patients["model"] == model]

    check(
        f"{model} has 38 patients",
        len(sub) == 38,
        f"found {len(sub)}"
    )

    dice = sub["dice"].mean()
    iou = sub["iou"].mean()
    hd95 = sub["hd95_mm"].mean()

    check(
        f"{model} Dice matches final result",
        np.isclose(dice, EXPECTED[model]["test_dice"], atol=1e-6),
        f"found {dice:.6f}"
    )

    check(
        f"{model} IoU matches final result",
        np.isclose(iou, EXPECTED[model]["iou"], atol=1e-6),
        f"found {iou:.6f}"
    )

    check(
        f"{model} HD95 matches final result",
        np.isclose(hd95, EXPECTED[model]["hd95"], atol=1e-6),
        f"found {hd95:.6f}"
    )


# ============================================================
# 2. VALIDATION HISTORIES
# ============================================================

print("\n[2] TRAINING HISTORIES")

for model in MODELS:

    path = os.path.join(
        LOGS,
        model,
        "training_history.json"
    )

    check(
        f"{model} history exists",
        os.path.exists(path)
    )

    if not os.path.exists(path):
        continue

    with open(path) as f:
        hist = json.load(f)

    epochs = [x["epoch"] for x in hist]

    best_dice = max(
        x["patient_val_dice"]
        for x in hist
    )

    best_epoch = max(
        hist,
        key=lambda x: x["patient_val_dice"]
    )["epoch"]

    stop_epoch = epochs[-1]

    check(
        f"{model} best validation Dice",
        np.isclose(
            best_dice,
            EXPECTED[model]["val_dice"],
            atol=1e-6
        ),
        f"found {best_dice:.6f}"
    )

    check(
        f"{model} best epoch",
        best_epoch == EXPECTED[model]["best_epoch"],
        f"found {best_epoch}"
    )

    check(
        f"{model} stopping epoch",
        stop_epoch == EXPECTED[model]["stop_epoch"],
        f"found {stop_epoch}"
    )


# ============================================================
# 3. MASTER PUBLICATION TABLE
# ============================================================

print("\n[3] MASTER PUBLICATION TABLE")

master_path = os.path.join(
    TABLES,
    "MASTER_Publication_Results.csv"
)

master = pd.read_csv(master_path)

check(
    "Master table has 4 models",
    len(master) == 4,
    f"found {len(master)}"
)

for model in MODELS:

    row = master[master["model"] == model]

    check(
        f"{model} exists in master table",
        len(row) == 1
    )

    if len(row) != 1:
        continue

    r = row.iloc[0]

    checks = [
        ("best_val_dice", EXPECTED[model]["val_dice"]),
        ("best_val_epoch", EXPECTED[model]["best_epoch"]),
        ("stopping_epoch", EXPECTED[model]["stop_epoch"]),
        ("test_dice", EXPECTED[model]["test_dice"]),
        ("test_iou", EXPECTED[model]["iou"]),
        ("test_hd95_mm", EXPECTED[model]["hd95"]),
    ]

    for col, expected in checks:

        check(
            f"{model} master {col}",
            np.isclose(
                float(r[col]),
                expected,
                atol=1e-6
            ),
            f"found {r[col]}"
        )


# ============================================================
# 4. EFFICIENCY TABLE
# ============================================================

print("\n[4] COMPUTATIONAL / ENERGY EFFICIENCY")

eff_path = os.path.join(
    EFF,
    "energy_efficiency_summary.csv"
)

eff = pd.read_csv(eff_path)

check(
    "Efficiency table has 4 models",
    len(eff) == 4,
    f"found {len(eff)}"
)

for model in MODELS:

    row = eff[eff["model"] == model]

    check(
        f"{model} efficiency record exists",
        len(row) == 1
    )


# ============================================================
# 5. PUBLICATION FIGURES
# ============================================================

print("\n[5] PUBLICATION FIGURES")

required_figures = [
    "Figure_1_Training_Convergence",
    "Figure_2_Dice_95CI",
    "Figure_3_MultiMetric_Comparison",
    "Figure_4_HD95_Comparison",
    "Figure_5_Accuracy_vs_Latency",
    "Figure_6_Qualitative_Model_Comparison",
    "Figure_7_Patient_Level_Dice_Distribution",
    "Figure_8_Patient_Level_HD95_Distribution",
    "Figure_9_Dice_vs_SOP_Proxy",
]

for base in required_figures:

    png = os.path.join(
        FIG,
        base + ".png"
    )

    pdf = os.path.join(
        FIG,
        base + ".pdf"
    )

    check(
        f"{base} PNG exists",
        os.path.exists(png)
    )

    check(
        f"{base} PDF exists",
        os.path.exists(pdf)
    )


# ============================================================
# 6. FIGURE 7 DATA
# ============================================================

print("\n[6] FIGURE 7 DATA")

f7 = pd.read_csv(
    os.path.join(
        FIG,
        "Figure_7_Patient_Level_Dice_Summary.csv"
    )
)

check(
    "Figure 7 contains 4 models",
    len(f7) == 4
)

for model in MODELS:

    r = f7[f7["model"] == model]

    check(
        f"Figure 7 {model} mean Dice",
        len(r) == 1 and np.isclose(
            r.iloc[0]["mean_dice"],
            EXPECTED[model]["test_dice"],
            atol=1e-6
        )
    )


# ============================================================
# 7. FIGURE 8 DATA
# ============================================================

print("\n[7] FIGURE 8 DATA")

f8 = pd.read_csv(
    os.path.join(
        FIG,
        "Figure_8_Patient_Level_HD95_Summary.csv"
    )
)

check(
    "Figure 8 contains 4 models",
    len(f8) == 4
)

for model in MODELS:

    r = f8[f8["model"] == model]

    check(
        f"Figure 8 {model} mean HD95",
        len(r) == 1 and np.isclose(
            r.iloc[0]["mean_hd95_mm"],
            EXPECTED[model]["hd95"],
            atol=1e-6
        )
    )


# ============================================================
# 8. FIGURE 9 DATA
# ============================================================

print("\n[8] FIGURE 9 DATA")

f9 = pd.read_csv(
    os.path.join(
        FIG,
        "Figure_9_Dice_vs_SOP_Proxy_Data.csv"
    )
)

check(
    "Figure 9 contains 4 models",
    len(f9) == 4
)

for model in MODELS:

    r = f9[f9["model"] == model]

    check(
        f"Figure 9 {model} Dice",
        len(r) == 1 and np.isclose(
            r.iloc[0]["test_dice"],
            EXPECTED[model]["test_dice"],
            atol=1e-6
        )
    )


# ============================================================
# 9. MODEL RANKING
# ============================================================

print("\n[9] FINAL MODEL RANKING")

ranking = (
    patients.groupby("model")["dice"]
    .mean()
    .sort_values(ascending=False)
)

print(ranking)

check(
    "A3 has highest test Dice",
    ranking.index[0] == "A3"
)

hd_ranking = (
    patients.groupby("model")["hd95_mm"]
    .mean()
    .sort_values(ascending=True)
)

print("\nHD95 ranking:")
print(hd_ranking)

check(
    "A3 has lowest mean HD95",
    hd_ranking.index[0] == "A3"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 78)
print("AUDIT SUMMARY")
print("=" * 78)

print(f"PASS: {PASS}")
print(f"FAIL: {FAIL}")

if FAIL == 0:
    print("\nFINAL STATUS: PASS")
    print("Publication figures, tables, histories, and final-test metrics")
    print("are numerically consistent.")
else:
    print("\nFINAL STATUS: REVIEW REQUIRED")
    print("One or more consistency checks failed.")

print("\nNo training performed.")
print("No checkpoint modified.")
print("No prediction modified.")
