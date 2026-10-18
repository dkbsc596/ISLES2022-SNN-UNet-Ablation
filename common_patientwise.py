import os
import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset


PATCH_SIZE = (128, 128, 64)
THRESHOLD = 0.5


def timestamp():
    return datetime.now().astimezone().isoformat()


class PatientWisePatchDataset(Dataset):

    def __init__(self, directory):
        self.directory = directory

        self.files = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".npz")
        ])

        if len(self.files) == 0:
            raise RuntimeError(
                f"No NPZ patches found in {directory}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        with np.load(path, allow_pickle=True) as data:

            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.float32)

            subject = str(data["subject"])

            start = np.asarray(
                data["start"],
                dtype=np.int64
            )

            original_shape = np.asarray(
                data["original_shape"],
                dtype=np.int64
            )

            pad_before = np.asarray(
                data["pad_before"],
                dtype=np.int64
            )

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if tuple(image.shape) != (3,) + PATCH_SIZE:
            raise RuntimeError(
                f"Invalid image shape in {path}: "
                f"{tuple(image.shape)}"
            )

        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        if tuple(mask.shape) != (1,) + PATCH_SIZE:
            raise RuntimeError(
                f"Invalid mask shape in {path}: "
                f"{tuple(mask.shape)}"
            )

        return (
            image,
            mask,
            subject,
            start,
            original_shape,
            pad_before
        )


def reconstruct_patient_predictions(
    predictions,
    metadata
):

    groups = {}

    for prediction, meta in zip(
        predictions,
        metadata
    ):

        subject = meta["subject"]

        if subject not in groups:
            groups[subject] = []

        groups[subject].append(
            (prediction, meta)
        )

    patient_probabilities = {}

    for subject, items in groups.items():

        original_shape = tuple(
            int(x)
            for x in items[0][1]["original_shape"]
        )

        probability_sum = np.zeros(
            original_shape,
            dtype=np.float32
        )

        probability_count = np.zeros(
            original_shape,
            dtype=np.float32
        )

        for prediction, meta in items:

            start = np.asarray(
                meta["start"],
                dtype=np.int64
            )

            pad_before = np.asarray(
                meta["pad_before"],
                dtype=np.int64
            )

            valid_shape = np.minimum(
                np.asarray(PATCH_SIZE)
                - pad_before,
                np.asarray(original_shape)
                - start
            )

            valid_shape = np.maximum(
                valid_shape,
                0
            )

            src_slices = tuple(
                slice(
                    int(pad_before[d]),
                    int(
                        pad_before[d]
                        + valid_shape[d]
                    )
                )
                for d in range(3)
            )

            dst_slices = tuple(
                slice(
                    int(start[d]),
                    int(
                        start[d]
                        + valid_shape[d]
                    )
                )
                for d in range(3)
            )

            probability_sum[dst_slices] += (
                prediction[src_slices]
            )

            probability_count[dst_slices] += 1.0

        if np.any(probability_count == 0):

            uncovered = int(
                np.sum(
                    probability_count == 0
                )
            )

            raise RuntimeError(
                f"{subject}: {uncovered} "
                f"uncovered voxels"
            )

        patient_probabilities[subject] = (
            probability_sum /
            probability_count
        )

    return patient_probabilities


def load_patient_ground_truth(
    preprocessed_dir,
    subject
):

    path = os.path.join(
        preprocessed_dir,
        f"{subject}.npz"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing patient file: {path}"
        )

    with np.load(
        path,
        allow_pickle=True
    ) as data:

        mask = data["mask"].astype(
            np.float32
        )

    return mask


def dice_score(
    prediction,
    ground_truth
):

    prediction = (
        prediction >= THRESHOLD
    )

    ground_truth = (
        ground_truth > 0.5
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    denominator = (
        prediction.sum()
        + ground_truth.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        2.0 * intersection /
        denominator
    )


def save_json(path, data):

    def convert(value):

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, dict):
            return {
                str(k): convert(v)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [
                convert(v)
                for v in value
            ]

        return value

    with open(path, "w") as f:

        json.dump(
            convert(data),
            f,
            indent=2
        )


def patientwise_validation(
    model,
    val_loader,
    preprocessed_dir,
    device,
    loss_function,
    is_snn=False
):

    model.eval()

    predictions = []
    metadata = []

    val_loss_sum = 0.0
    val_batch_count = 0

    functional = None

    if is_snn:

        from spikingjelly.activation_based import functional

    with torch.no_grad():

        for batch in val_loader:

            (
                images,
                masks,
                subjects,
                starts,
                original_shapes,
                pad_befores
            ) = batch

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            if is_snn:
                functional.reset_net(model)

            outputs = model(images)

            loss = loss_function(
                outputs,
                masks
            )

            val_loss_sum += (
                loss.item()
            )

            val_batch_count += 1

            probabilities = torch.sigmoid(
                outputs
            )

            probabilities = (
                probabilities[:, 0]
                .detach()
                .cpu()
                .numpy()
            )

            for i in range(
                len(subjects)
            ):

                predictions.append(
                    probabilities[i]
                )

                metadata.append({
                    "subject":
                        str(subjects[i]),

                    "start":
                        np.asarray(
                            starts[i],
                            dtype=np.int64
                        ),

                    "original_shape":
                        np.asarray(
                            original_shapes[i],
                            dtype=np.int64
                        ),

                    "pad_before":
                        np.asarray(
                            pad_befores[i],
                            dtype=np.int64
                        )
                })

    patient_probabilities = (
        reconstruct_patient_predictions(
            predictions,
            metadata
        )
    )

    patient_dice = {}

    for subject, probability in (
        patient_probabilities.items()
    ):

        ground_truth = (
            load_patient_ground_truth(
                preprocessed_dir,
                subject
            )
        )

        if probability.shape != ground_truth.shape:

            raise RuntimeError(
                f"{subject}: prediction shape "
                f"{probability.shape} != "
                f"ground truth shape "
                f"{ground_truth.shape}"
            )

        patient_dice[subject] = dice_score(
            probability,
            ground_truth
        )

    if len(patient_dice) != 38:

        raise RuntimeError(
            f"Expected 38 validation patients, "
            f"found {len(patient_dice)}"
        )

    mean_dice = float(
        np.mean(
            list(patient_dice.values())
        )
    )

    val_loss = float(
        val_loss_sum /
        max(val_batch_count, 1)
    )

    return {
        "val_loss": val_loss,
        "val_dice": mean_dice,
        "patient_dice": patient_dice,
        "patient_count":
            len(patient_dice),
        "patch_count":
            len(predictions)
    }
