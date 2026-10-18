import os
import sys
import json
import csv
import math
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from medpy.metric.binary import hd95

ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"
sys.path.insert(0, ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SPLIT = os.path.join(ROOT, "datasets/splits/test.txt")
PATCH_DIR = os.path.join(ROOT, "datasets/patches/test")
GT_DIR = os.path.join(ROOT, "datasets/ISLES2022_preprocessed")

OUT_DIR = os.path.join(
    ROOT,
    "results/block50_epoch/final_internal_heldout"
)

os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.5
PATCH_SIZE = (128, 128, 64)


# ============================================================
# A0 — EXACT TRAINED ARCHITECTURE
# ============================================================

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                bias=False,
                padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                bias=False,
                padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv3D(3, 16)
        self.enc2 = DoubleConv3D(16, 32)
        self.enc3 = DoubleConv3D(32, 64)

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        self.bottleneck = DoubleConv3D(64, 128)

        self.up3 = nn.ConvTranspose3d(
            128, 64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv3D(128, 64)

        self.up2 = nn.ConvTranspose3d(
            64, 32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv3D(64, 32)

        self.up1 = nn.ConvTranspose3d(
            32, 16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv3D(32, 16)

        self.out_conv = nn.Conv3d(
            16, 1,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# ============================================================
# SNN MODELS
# ============================================================

from models.controlled_snn_unet import (
    A1SNNUNet3D,
    A2SNNUNet3D,
    A3SNNUNet3D
)


# ============================================================
# TEST SUBJECTS
# ============================================================

with open(TEST_SPLIT, "r") as f:
    TEST_SUBJECTS = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("=" * 75)
print("BLOCK50 INTERNAL HELD-OUT TEST EVALUATION")
print("=" * 75)
print("Device:", DEVICE)
print("Test subjects:", len(TEST_SUBJECTS))
print("Threshold:", THRESHOLD)
print()

if len(TEST_SUBJECTS) != 38:
    raise RuntimeError(
        f"Expected 38 held-out test subjects, found {len(TEST_SUBJECTS)}"
    )


# ============================================================
# PATCH RECONSTRUCTION
# ============================================================

def load_test_patches(subject):

    files = []

    for fname in sorted(os.listdir(PATCH_DIR)):

        if not fname.endswith(".npz"):
            continue

        path = os.path.join(PATCH_DIR, fname)

        with np.load(path, allow_pickle=True) as data:

            if "subject" not in data:
                continue

            patch_subject = str(data["subject"])

            if patch_subject == subject:
                files.append(path)

    if len(files) == 0:
        raise RuntimeError(
            f"No test patches found for {subject}"
        )

    return files


def reconstruct_prediction(subject, patch_predictions):

    gt_path = os.path.join(
        GT_DIR,
        subject + ".npz"
    )

    if not os.path.exists(gt_path):
        raise RuntimeError(
            f"Ground-truth file missing: {gt_path}"
        )

    with np.load(gt_path, allow_pickle=True) as gt_data:

        gt = np.asarray(
            gt_data["mask"]
        ).astype(np.float32)

        original_shape = tuple(gt.shape)

        spacing = np.asarray(
            gt_data["spacing"],
            dtype=np.float64
        )

    prediction_sum = np.zeros(
        original_shape,
        dtype=np.float64
    )

    prediction_count = np.zeros(
        original_shape,
        dtype=np.float64
    )

    for patch_path, pred in patch_predictions:

        with np.load(
            patch_path,
            allow_pickle=True
        ) as data:

            start = np.asarray(
                data["start"]
            ).astype(int)

            pad_before = np.asarray(
                data["pad_before"]
            ).astype(int)

        start = tuple(start.tolist())
        pad_before = tuple(pad_before.tolist())

        pred = np.asarray(pred).squeeze().astype(np.float64)

        if pred.shape != PATCH_SIZE:
            raise RuntimeError(
                f"Wrong prediction patch shape for {patch_path}: "
                f"{pred.shape}"
            )

        valid_shape = tuple(
            min(
                PATCH_SIZE[d] - pad_before[d],
                original_shape[d] - start[d]
            )
            for d in range(3)
        )

        src = tuple(
            slice(
                pad_before[d],
                pad_before[d] + valid_shape[d]
            )
            for d in range(3)
        )

        dst = tuple(
            slice(
                start[d],
                start[d] + valid_shape[d]
            )
            for d in range(3)
        )

        prediction_sum[dst] += pred[src]
        prediction_count[dst] += 1.0

    if np.any(prediction_count == 0):
        uncovered = int(
            np.sum(prediction_count == 0)
        )

        raise RuntimeError(
            f"{subject}: {uncovered} uncovered voxels"
        )

    prediction = (
        prediction_sum /
        prediction_count
    )

    return prediction.astype(np.float32), gt, spacing


# ============================================================
# METRICS
# ============================================================

def safe_div(a, b):

    if b == 0:
        return np.nan

    return float(a / b)


def segmentation_metrics(pred_prob, gt, spacing):

    pred = pred_prob >= THRESHOLD
    gt_bin = gt > 0.5

    pred_count = int(pred.sum())
    gt_count = int(gt_bin.sum())

    intersection = int(
        np.logical_and(pred, gt_bin).sum()
    )

    union = int(
        np.logical_or(pred, gt_bin).sum()
    )

    tp = intersection

    fp = int(
        np.logical_and(pred, ~gt_bin).sum()
    )

    fn = int(
        np.logical_and(~pred, gt_bin).sum()
    )

    # Dice
    if pred_count + gt_count == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection /
            (pred_count + gt_count)
        )

    # IoU
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    precision = safe_div(
        tp,
        tp + fp
    )

    recall = safe_div(
        tp,
        tp + fn
    )

    # Physical voxel volume
    voxel_volume_mm3 = float(
        np.prod(spacing)
    )

    gt_volume_ml = (
        gt_count *
        voxel_volume_mm3 /
        1000.0
    )

    pred_volume_ml = (
        pred_count *
        voxel_volume_mm3 /
        1000.0
    )

    absolute_volume_error_ml = abs(
        pred_volume_ml -
        gt_volume_ml
    )

    if gt_volume_ml == 0:
        relative_volume_error_percent = np.nan
    else:
        relative_volume_error_percent = (
            absolute_volume_error_ml /
            gt_volume_ml *
            100.0
        )

    # Connected lesion count: 26-connectivity
    structure = np.ones(
        (3, 3, 3),
        dtype=np.uint8
    )

    _, gt_lesion_count = ndimage.label(
        gt_bin,
        structure=structure
    )

    _, pred_lesion_count = ndimage.label(
        pred,
        structure=structure
    )

    # HD95
    #
    # HD95 is undefined when either segmentation is empty.
    # We record NaN rather than inventing a value.
    if pred_count == 0 or gt_count == 0:

        hd95_mm = np.nan

    else:

        try:

            hd95_mm = float(
                hd95(
                    pred,
                    gt_bin,
                    voxelspacing=tuple(
                        spacing.tolist()
                    )
                )
            )

        except Exception:

            hd95_mm = np.nan

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision)
            if not np.isnan(precision)
            else np.nan,
        "recall": float(recall)
            if not np.isnan(recall)
            else np.nan,
        "hd95_mm": hd95_mm,
        "gt_volume_ml": float(gt_volume_ml),
        "pred_volume_ml": float(pred_volume_ml),
        "absolute_volume_error_ml":
            float(absolute_volume_error_ml),
        "relative_volume_error_percent":
            float(relative_volume_error_percent)
            if not np.isnan(relative_volume_error_percent)
            else np.nan,
        "gt_lesion_count": int(gt_lesion_count),
        "pred_lesion_count": int(pred_lesion_count),
        "gt_voxels": gt_count,
        "pred_voxels": pred_count
    }


# ============================================================
# MODEL LOADING
# ============================================================

MODELS = {

    "A0": {
        "description": "3D CNN U-Net",
        "checkpoint":
            "checkpoints/block50_epoch/A0/A0_best.pth",
        "constructor": UNet3D,
        "snn": False
    },

    "A1": {
        "description": "3D SNN U-Net",
        "checkpoint":
            "checkpoints/block50_epoch/A1/block50_epoch/A1/A1_best.pth",
        "constructor": A1SNNUNet3D,
        "snn": True
    },

    "A2": {
        "description":
            "3D SNN U-Net + Attention",
        "checkpoint":
            "checkpoints/block50_epoch/A2/A2_best.pth",
        "constructor": A2SNNUNet3D,
        "snn": True
    },

    "A3": {
        "description":
            "3D SNN U-Net + Attention + Residual",
        "checkpoint":
            "checkpoints/block50_epoch/A3/A3_best.pth",
        "constructor": A3SNNUNet3D,
        "snn": True
    }
}


def load_model(info):

    model = info["constructor"]()

    checkpoint_path = os.path.join(
        ROOT,
        info["checkpoint"]
    )

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"Checkpoint missing: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


# ============================================================
# SINGLE PATCH INFERENCE
# ============================================================

def predict_patch(model, image, is_snn):

    x = torch.from_numpy(
        image.astype(np.float32)
    )

    if x.ndim == 4:
        x = x.unsqueeze(0)

    x = x.to(DEVICE)

    if tuple(x.shape) != (
        1, 3, 128, 128, 64
    ):
        raise RuntimeError(
            f"Unexpected input shape: {tuple(x.shape)}"
        )

    if is_snn:

        from spikingjelly.activation_based import functional

        functional.reset_net(model)

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda",
                enabled=True
            ):

                output = model(x)

        else:

            output = model(x)

    if is_snn:

        from spikingjelly.activation_based import functional

        functional.reset_net(model)

    if tuple(output.shape) != (
        1, 1, 128, 128, 64
    ):
        raise RuntimeError(
            f"Unexpected output shape: "
            f"{tuple(output.shape)}"
        )

    probability = torch.sigmoid(
        output
    )

    return (
        probability
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# AGGREGATION
# ============================================================

def finite_values(values):

    return np.asarray(
        [
            float(v)
            for v in values
            if np.isfinite(v)
        ],
        dtype=np.float64
    )


def summarize(values):

    arr = finite_values(values)

    if len(arr) == 0:

        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None
        }

    q1, q3 = np.percentile(
        arr,
        [25, 75]
    )

    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1))
            if len(arr) > 1
            else 0.0,
        "median": float(np.median(arr)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1)
    }


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_name, info):

    print()
    print("=" * 75)
    print(
        f"{model_name}: {info['description']}"
    )
    print("=" * 75)

    model, checkpoint = load_model(info)

    print(
        "Checkpoint best epoch:",
        checkpoint.get("best_epoch")
    )

    print(
        "Validation Dice:",
        checkpoint.get("best_val_dice")
    )

    rows = []

    for index, subject in enumerate(
        TEST_SUBJECTS,
        start=1
    ):

        print(
            f"[{index:02d}/38] {subject}",
            flush=True
        )

        patch_files = load_test_patches(
            subject
        )

        patch_predictions = []

        for patch_path in patch_files:

            with np.load(
                patch_path,
                allow_pickle=True
            ) as data:

                image = np.asarray(
                    data["image"]
                ).astype(np.float32)

            if tuple(image.shape) != (3,) + PATCH_SIZE:

                raise RuntimeError(
                    f"{subject}: "
                    f"wrong image shape "
                    f"{image.shape}"
                )

            prediction = predict_patch(
                model,
                image,
                info["snn"]
            )

            patch_predictions.append(
                (
                    patch_path,
                    prediction
                )
            )

        prediction, gt, spacing = (
            reconstruct_prediction(
                subject,
                patch_predictions
            )
        )

        metrics = segmentation_metrics(
            prediction,
            gt,
            spacing
        )

        row = {
            "model": model_name,
            "subject": subject,
            "num_patches": len(patch_files),
            "spacing_x_mm": float(spacing[0]),
            "spacing_y_mm": float(spacing[1]),
            "spacing_z_mm": float(spacing[2]),
            **metrics
        }

        rows.append(row)

        print(
            f"  Dice={metrics['dice']:.4f} "
            f"IoU={metrics['iou']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"HD95={metrics['hd95_mm']:.2f}"
            if np.isfinite(metrics["hd95_mm"])
            else
            f"  Dice={metrics['dice']:.4f} "
            f"IoU={metrics['iou']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"HD95=NaN"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_mm",
        "gt_volume_ml",
        "pred_volume_ml",
        "absolute_volume_error_ml",
        "relative_volume_error_percent",
        "gt_lesion_count",
        "pred_lesion_count"
    ]

    summary = {}

    for metric in metric_names:

        summary[metric] = summarize(
            [
                row[metric]
                for row in rows
            ]
        )

    print()
    print("SUMMARY:", model_name)

    for metric in metric_names:

        s = summary[metric]

        print(
            f"{metric:35s} "
            f"mean={s['mean']} "
            f"std={s['std']} "
            f"median={s['median']} "
            f"IQR={s['iqr']}"
        )

    return rows, summary


# ============================================================
# MAIN
# ============================================================

all_rows = []
all_summaries = {}

for model_name, info in MODELS.items():

    rows, summary = evaluate_model(
        model_name,
        info
    )

    all_rows.extend(rows)
    all_summaries[model_name] = summary


# ============================================================
# SAVE PATIENT-LEVEL CSV
# ============================================================

csv_path = os.path.join(
    OUT_DIR,
    "patient_level_results.csv"
)

fieldnames = list(
    all_rows[0].keys()
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_rows)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

json_path = os.path.join(
    OUT_DIR,
    "summary_results.json"
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        {
            "experiment":
                "Block50 Internal Held-Out Test",
            "num_test_subjects":
                len(TEST_SUBJECTS),
            "threshold":
                THRESHOLD,
            "test_subjects":
                TEST_SUBJECTS,
            "models":
                all_summaries
        },
        f,
        indent=2
    )


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print()
print("=" * 100)
print("FINAL INTERNAL HELD-OUT TEST RESULTS")
print("=" * 100)

print(
    f"{'Model':<6}"
    f"{'Dice':>12}"
    f"{'IoU':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'HD95(mm)':>14}"
)

print("-" * 100)

for model_name in MODELS:

    s = all_summaries[model_name]

    def fmt(metric):

        value = s[metric]["mean"]

        if value is None:
            return "N/A"

        return f"{value:.4f}"

    print(
        f"{model_name:<6}"
        f"{fmt('dice'):>12}"
        f"{fmt('iou'):>12}"
        f"{fmt('precision'):>12}"
        f"{fmt('recall'):>12}"
        f"{fmt('hd95_mm'):>14}"
    )


print()
print("Patient-level CSV:")
print(csv_path)

print()
print("Summary JSON:")
print(json_path)

print()
print("=" * 75)
print("FINAL EVALUATION COMPLETE")
print("=" * 75)
print("38 internal held-out patients evaluated for each architecture.")
print("No training performed.")
print("No checkpoint modified.")
print("No validation/test tuning performed.")
print("=" * 75)
