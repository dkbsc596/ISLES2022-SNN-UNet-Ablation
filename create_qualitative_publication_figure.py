import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FINAL PUBLICATION QUALITATIVE COMPARISON
# ============================================================
#
# Internal IDs:
#   A0 -> 3D U-Net
#   A1 -> 3D SNN-U-Net
#   A2 -> 3D Attention SNN-U-Net
#   A3 -> 3D Attention-Residual SNN-U-Net
#
# IMPORTANT:
#   - Uses held-out test predictions only
#   - Does NOT modify checkpoints
#   - Does NOT retrain models
#   - Does NOT use old block11j predictions
# ============================================================

PROJECT_ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"

PRED_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "block45_final_test"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "publication_results",
    "publication_figures"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Publication names
# ------------------------------------------------------------

MODELS = {
    "A0": "3D U-Net",
    "A1": "3D SNN-U-Net",
    "A2": "3D Attention SNN-U-Net",
    "A3": "3D Attention-Residual SNN-U-Net",
}

MODEL_IDS = ["A0", "A1", "A2", "A3"]

# ------------------------------------------------------------
# Locate original ISLES-2022 dataset
# ------------------------------------------------------------

DATASET_CANDIDATES = [
    "/home/krishna_19622/isles2022_extracted/ISLES-2022",
    "/home/krishna_19622/dataset/ISLES-2022",
    "/home/krishna_19622/dataset/isles2022/ISLES-2022",
]

DATASET_ROOT = None

for candidate in DATASET_CANDIDATES:
    if os.path.isdir(candidate):
        DATASET_ROOT = candidate
        break

if DATASET_ROOT is None:
    raise FileNotFoundError(
        "\nCould not find ISLES-2022 dataset.\n"
        "Checked:\n" +
        "\n".join(DATASET_CANDIDATES)
    )

print("=" * 90)
print("FINAL PUBLICATION QUALITATIVE FIGURE")
print("=" * 90)
print("Dataset root :", DATASET_ROOT)
print("Prediction root :", PRED_ROOT)
print("Output directory :", OUT_DIR)

# ------------------------------------------------------------
# Find DWI for a subject
# ------------------------------------------------------------

def find_dwi(subject):

    matches = []

    subject_dir = os.path.join(DATASET_ROOT, subject)

    if os.path.isdir(subject_dir):
        matches.extend(
            glob.glob(
                os.path.join(
                    subject_dir,
                    "**",
                    "*dwi*.nii.gz"
                ),
                recursive=True
            )
        )

        matches.extend(
            glob.glob(
                os.path.join(
                    subject_dir,
                    "**",
                    "*dwi*.nii"
                ),
                recursive=True
            )
        )

    if not matches:
        # fallback: search globally
        matches.extend(
            glob.glob(
                os.path.join(
                    DATASET_ROOT,
                    "**",
                    subject,
                    "**",
                    "*dwi*.nii.gz"
                ),
                recursive=True
            )
        )

        matches.extend(
            glob.glob(
                os.path.join(
                    DATASET_ROOT,
                    "**",
                    subject,
                    "**",
                    "*dwi*.nii"
                ),
                recursive=True
            )
        )

    matches = sorted(set(matches))

    if not matches:
        return None

    # Prefer files explicitly containing dwi
    matches = [
        x for x in matches
        if "dwi" in os.path.basename(x).lower()
    ]

    return matches[0] if matches else None


# ------------------------------------------------------------
# Load NIfTI
# ------------------------------------------------------------

try:
    import nibabel as nib
except ImportError:
    raise ImportError(
        "nibabel is required.\n"
        "Install with:\n"
        "pip3 install nibabel"
    )


def load_dwi(subject):

    path = find_dwi(subject)

    if path is None:
        return None, None

    img = nib.load(path)
    data = img.get_fdata()

    return data.astype(np.float32), path


# ------------------------------------------------------------
# Collect common test patients
# ------------------------------------------------------------

patient_sets = {}

for model_id in MODEL_IDS:

    directory = os.path.join(
        PRED_ROOT,
        model_id,
        "patient_volumes"
    )

    files = sorted(
        glob.glob(
            os.path.join(directory, "*.npz")
        )
    )

    patient_sets[model_id] = {
        os.path.basename(f).replace(".npz", ""): f
        for f in files
    }

common_subjects = set(
    patient_sets["A0"].keys()
)

for model_id in MODEL_IDS[1:]:
    common_subjects &= set(
        patient_sets[model_id].keys()
    )

common_subjects = sorted(common_subjects)

print()
print("Common held-out patients :", len(common_subjects))

if len(common_subjects) != 38:
    print(
        "WARNING: Expected 38 common test patients."
    )

# ------------------------------------------------------------
# Load patient information
# ------------------------------------------------------------

patients = []

for subject in common_subjects:

    data = {}

    valid = True

    for model_id in MODEL_IDS:

        f = patient_sets[model_id][subject]

        z = np.load(f)

        data[model_id] = {
            "prediction": z["prediction"].astype(np.uint8),
            "probability": z["probability"].astype(np.float32),
            "mask": z["mask"].astype(np.uint8),
        }

    gt = data["A0"]["mask"]

    lesion_volume = int(gt.sum())

    if lesion_volume == 0:
        continue

    # Check that GT is identical across model files
    for model_id in MODEL_IDS[1:]:

        if not np.array_equal(
            gt,
            data[model_id]["mask"]
        ):
            print(
                "WARNING: GT differs for",
                subject,
                "between models."
            )
            valid = False

    if not valid:
        continue

    # Patient-level Dice for each model
    dices = {}

    for model_id in MODEL_IDS:

        pred = data[model_id]["prediction"]

        intersection = np.logical_and(
            pred > 0,
            gt > 0
        ).sum()

        denom = (
            (pred > 0).sum()
            + (gt > 0).sum()
        )

        dice = (
            2.0 * intersection / denom
            if denom > 0
            else 1.0
        )

        dices[model_id] = dice

    mean_dice = float(
        np.mean(list(dices.values()))
    )

    patients.append({
        "subject": subject,
        "lesion_volume": lesion_volume,
        "dices": dices,
        "mean_dice": mean_dice,
        "data": data,
    })


if len(patients) < 4:
    raise RuntimeError(
        "Fewer than four valid lesion-containing patients."
    )

# ------------------------------------------------------------
# Select representative cases
# ------------------------------------------------------------

patients_sorted = sorted(
    patients,
    key=lambda x: x["lesion_volume"]
)

n = len(patients_sorted)

# Small lesion
small_patient = patients_sorted[
    max(0, int(round(0.10 * (n - 1))))
]

# Medium lesion
medium_patient = patients_sorted[
    int(round(0.40 * (n - 1)))
]

# Large lesion
large_patient = patients_sorted[
    int(round(0.85 * (n - 1)))
]

# Challenging case = lowest mean Dice
challenging_patient = min(
    patients,
    key=lambda x: x["mean_dice"]
)

selected = []

for p in [
    small_patient,
    medium_patient,
    large_patient,
    challenging_patient,
]:

    if p["subject"] not in [
        x["subject"] for x in selected
    ]:
        selected.append(p)

# If duplicates occurred, fill from lesion-volume distribution
for p in patients_sorted:

    if len(selected) >= 4:
        break

    if p["subject"] not in [
        x["subject"] for x in selected
    ]:
        selected.append(p)

selected = selected[:4]

print()
print("=" * 90)
print("SELECTED REPRESENTATIVE PATIENTS")
print("=" * 90)

for i, p in enumerate(selected, 1):

    print(
        f"{i}. {p['subject']} | "
        f"GT voxels={p['lesion_volume']} | "
        f"mean Dice={p['mean_dice']:.4f}"
    )

    for model_id in MODEL_IDS:
        print(
            f"   {MODELS[model_id]:40s} "
            f"Dice={p['dices'][model_id]:.4f}"
        )

# ------------------------------------------------------------
# Select lesion-focused slice
# ------------------------------------------------------------

def choose_slice(gt, predictions):

    # Ground truth lesion count per axial slice
    gt_counts = gt.sum(axis=(0, 1))

    lesion_slices = np.where(
        gt_counts > 0
    )[0]

    if len(lesion_slices) == 0:
        return gt.shape[2] // 2

    # Score slices using GT + prediction activity.
    # This favors representative lesion slices while
    # remaining comparable across all four models.
    scores = {}

    for z in lesion_slices:

        gt_area = int(gt[:, :, z].sum())

        pred_areas = []

        for pred in predictions:
            pred_areas.append(
                int(pred[:, :, z].sum())
            )

        # Favor substantial GT lesion and some model activity.
        score = (
            2.0 * gt_area
            + 0.5 * max(pred_areas)
            + 0.25 * np.mean(pred_areas)
        )

        scores[z] = score

    return max(
        scores,
        key=scores.get
    )


# ------------------------------------------------------------
# Normalize DWI
# ------------------------------------------------------------

def normalize_dwi(slice_data):

    x = np.asarray(
        slice_data,
        dtype=np.float32
    )

    finite = np.isfinite(x)

    if not finite.any():
        return np.zeros_like(x)

    values = x[finite]

    low = np.percentile(values, 1)
    high = np.percentile(values, 99)

    if high <= low:
        return np.zeros_like(x)

    x = np.clip(
        x,
        low,
        high
    )

    x = (x - low) / (high - low)

    return x


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=4,
    ncols=6,
    figsize=(18, 13),
    constrained_layout=True
)

column_titles = [
    "DWI",
    "Ground Truth",
    "3D U-Net",
    "3D SNN-U-Net",
    "3D Attention SNN-U-Net",
    "3D Attention-Residual SNN-U-Net",
]

# ------------------------------------------------------------
# Output CSV for reproducibility
# ------------------------------------------------------------

csv_path = os.path.join(
    OUT_DIR,
    "Figure_6_Selected_Cases.csv"
)

with open(csv_path, "w") as f:

    f.write(
        "row,subject,slice,"
        "gt_voxels,mean_dice,"
        "dice_3D_UNet,"
        "dice_3D_SNN_UNet,"
        "dice_3D_Attention_SNN_UNet,"
        "dice_3D_Attention_Residual_SNN_UNet\n"
    )

    # --------------------------------------------------------
    # Plot rows
    # --------------------------------------------------------

    for row, patient in enumerate(selected):

        subject = patient["subject"]
        gt = patient["data"]["A0"]["mask"]

        predictions = [
            patient["data"][m]["prediction"]
            for m in MODEL_IDS
        ]

        # Verify all geometries
        for pred in predictions:

            if pred.shape != gt.shape:
                raise RuntimeError(
                    f"Shape mismatch for {subject}: "
                    f"GT={gt.shape}, prediction={pred.shape}"
                )

        # Load DWI
        dwi, dwi_path = load_dwi(subject)

        if dwi is None:
            raise FileNotFoundError(
                f"\nDWI not found for {subject}.\n"
                "Please verify the ISLES-2022 dataset path."
            )

        print()
        print(
            "DWI:",
            subject,
            "shape=",
            dwi.shape,
            "path=",
            dwi_path
        )

        if dwi.shape != gt.shape:

            raise RuntimeError(
                f"\nDWI geometry mismatch for {subject}.\n"
                f"DWI shape = {dwi.shape}\n"
                f"Prediction/GT shape = {gt.shape}\n\n"
                "Do not continue with the figure until "
                "the DWI geometry is matched."
            )

        z = choose_slice(
            gt,
            predictions
        )

        dwi_slice = normalize_dwi(
            dwi[:, :, z]
        )

        gt_slice = gt[:, :, z]

        # ----------------------------------------------------
        # DWI
        # ----------------------------------------------------

        ax = axes[row, 0]

        ax.imshow(
            np.rot90(dwi_slice),
            cmap="gray"
        )

        ax.set_title(
            column_titles[0],
            fontsize=12,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ax = axes[row, 1]

        ax.imshow(
            np.rot90(dwi_slice),
            cmap="gray"
        )

        ax.imshow(
            np.ma.masked_where(
                np.rot90(gt_slice) == 0,
                np.rot90(gt_slice)
            ),
            cmap="Greens",
            alpha=0.70,
            vmin=0,
            vmax=1
        )

        ax.set_title(
            column_titles[1],
            fontsize=12,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        for col, model_id in enumerate(
            MODEL_IDS,
            start=2
        ):

            ax = axes[row, col]

            pred = patient["data"][
                model_id
            ]["prediction"]

            pred_slice = pred[:, :, z]

            ax.imshow(
                np.rot90(dwi_slice),
                cmap="gray"
            )

            ax.imshow(
                np.ma.masked_where(
                    np.rot90(pred_slice) == 0,
                    np.rot90(pred_slice)
                ),
                cmap="Reds",
                alpha=0.70,
                vmin=0,
                vmax=1
            )

            ax.set_title(
                MODELS[model_id],
                fontsize=11,
                fontweight="bold"
            )

        # ----------------------------------------------------
        # Row label
        # ----------------------------------------------------

        axes[row, 0].set_ylabel(
            f"{subject}\nSlice {z}",
            fontsize=11,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # Save CSV information
        # ----------------------------------------------------

        f.write(
            f"{row+1},"
            f"{subject},"
            f"{z},"
            f"{patient['lesion_volume']},"
            f"{patient['mean_dice']:.6f},"
            f"{patient['dices']['A0']:.6f},"
            f"{patient['dices']['A1']:.6f},"
            f"{patient['dices']['A2']:.6f},"
            f"{patient['dices']['A3']:.6f}\n"
        )

# ------------------------------------------------------------
# Clean axes
# ------------------------------------------------------------

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

# ------------------------------------------------------------
# Main title
# ------------------------------------------------------------

fig.suptitle(
    "Qualitative Comparison of 3D Stroke Lesion Segmentation",
    fontsize=18,
    fontweight="bold"
)

# ------------------------------------------------------------
# Save publication figure
# ------------------------------------------------------------

output_png = os.path.join(
    OUT_DIR,
    "Figure_6_Qualitative_A0_A3_Comparison.png"
)

output_pdf = os.path.join(
    OUT_DIR,
    "Figure_6_Qualitative_A0_A3_Comparison.pdf"
)

fig.savefig(
    output_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    output_pdf,
    bbox_inches="tight"
)

plt.close(fig)

print()
print("=" * 90)
print("QUALITATIVE FIGURE CREATED")
print("=" * 90)
print("PNG :", output_png)
print("PDF :", output_pdf)
print("CSV :", csv_path)
print()
print("Publication model names used:")
for model_id in MODEL_IDS:
    print(
        f"  {model_id} -> {MODELS[model_id]}"
    )

print("=" * 90)
