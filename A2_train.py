import os
import random
from datetime import datetime
import time
from tqdm import tqdm
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.utils.data import Dataset, DataLoader

from models.controlled_snn_unet import A2SNNUNet3D

from block50_epoch.common_patientwise import (
    PatientWisePatchDataset,
    patientwise_validation,
)

# ============================================================
# BLOCK 50 - A2: 3D SNN U-NET + ATTENTION 50-EPOCH PATIENT-WISE TRAINING
# ============================================================
# A2 = 3D SNN U-Net + Attention
# SNN / LIF enabled
# Attention enabled
# No Residual blocks
#
# DATA SPLIT
# ------------------------------------------------------------
# Training Set              : 174 subjects
# Validation Set            : 38 subjects
# Internal Held-Out Test Set: 38 subjects
# Total Public Cohort       : 250 subjects
#
# IMPORTANT:
# Internal Held-Out Test Set is NOT used during training,
# validation, model selection, or checkpoint selection.
# ============================================================

PROJECT_ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"

TRAIN_PATCH_DIR = os.path.join(
    PROJECT_ROOT, "datasets", "patches", "train"
)

VAL_PATCH_DIR = os.path.join(
    PROJECT_ROOT, "datasets", "patches", "val"
)

TRAIN_SPLIT = os.path.join(
    PROJECT_ROOT, "datasets", "splits", "train.txt"
)

VAL_SPLIT = os.path.join(
    PROJECT_ROOT, "datasets", "splits", "validation.txt"
)

TEST_SPLIT = os.path.join(
    PROJECT_ROOT, "datasets", "splits", "test.txt"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT, "checkpoints", "block50_epoch", "A2"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT, "results", "block50_epoch", "A2"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BLOCK 50 - A2: 3D SNN U-NET + ATTENTION 50-EPOCH PATIENT-WISE TRAINING")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")


# ============================================================
# 3. READ PATIENT SPLITS
# ============================================================

def read_subjects(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        subjects = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return subjects


train_subjects = read_subjects(TRAIN_SPLIT)
val_subjects = read_subjects(VAL_SPLIT)
test_subjects = read_subjects(TEST_SPLIT)

train_set = set(train_subjects)
val_set = set(val_subjects)
test_set = set(test_subjects)

print("\n" + "=" * 70)
print("PATIENT-LEVEL SPLIT")
print("=" * 70)

print(f"Training subjects              : {len(train_set)}")
print(f"Validation subjects            : {len(val_set)}")
print(f"Internal Held-Out Test subjects: {len(test_set)}")
print(f"Total unique subjects          : "
      f"{len(train_set | val_set | test_set)}")


# ============================================================
# 4. SPLIT SAFETY CHECK
# ============================================================

assert len(train_set) == 174, \
    f"Expected 174 training subjects, found {len(train_set)}"

assert len(val_set) == 38, \
    f"Expected 38 validation subjects, found {len(val_set)}"

assert len(test_set) == 38, \
    f"Expected 38 internal held-out test subjects, found {len(test_set)}"

assert len(train_set & val_set) == 0, \
    "TRAIN/VAL patient leakage detected"

assert len(train_set & test_set) == 0, \
    "TRAIN/INTERNAL-TEST patient leakage detected"

assert len(val_set & test_set) == 0, \
    "VAL/INTERNAL-TEST patient leakage detected"

assert len(train_set | val_set | test_set) == 250, \
    "Expected 250 unique public-cohort subjects"

print("\nSplit integrity: PASS")
print("174 + 38 + 38 = 250")
print("Internal Held-Out Test Set: LOCKED")


# ============================================================
# 5. DATASET
# ============================================================

class ISLES3DPatchDataset(Dataset):

    def __init__(self, patch_dir, allowed_subjects):

        self.patch_dir = patch_dir
        self.allowed_subjects = set(allowed_subjects)

        all_files = sorted([
            f for f in os.listdir(patch_dir)
            if f.endswith(".npz")
        ])

        self.files = []

        for fname in all_files:

            path = os.path.join(
                patch_dir,
                fname
            )

            with np.load(
                path,
                allow_pickle=True
            ) as data:

                subject = data["subject"]

                if isinstance(subject, np.ndarray):
                    subject = subject.item()

                subject = str(subject)

            if subject in self.allowed_subjects:
                self.files.append(fname)

        if len(self.files) == 0:
            raise RuntimeError(
                f"No valid patches found in {patch_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        path = os.path.join(
            self.patch_dir,
            fname
        )

        with np.load(
            path,
            allow_pickle=True
        ) as data:

            image = data["image"].astype(
                np.float32
            )

            mask = data["mask"].astype(
                np.float32
            )

            subject = data["subject"]

            if isinstance(subject, np.ndarray):
                subject = subject.item()

            subject = str(subject)

        # image:
        # (3,128,128,64)
        image = torch.from_numpy(image)

        # mask:
        # (128,128,64)
        # Convert to:
        # (1,128,128,64)
        mask = torch.from_numpy(
            mask[None, ...]
        )

        return image, mask, subject


# ============================================================
# 6. LOAD TRAINING / VALIDATION DATA
# ============================================================

train_dataset = ISLES3DPatchDataset(
    TRAIN_PATCH_DIR,
    train_set
)

val_dataset = ISLES3DPatchDataset(
    VAL_PATCH_DIR,
    val_set
)

print("\n" + "=" * 70)
print("PATCH DATA")
print("=" * 70)

print(f"Training patches   : {len(train_dataset)}")
print(f"Validation patches : {len(val_dataset)}")

print("Internal Held-Out Test patches: NOT LOADED")


# ============================================================
# 7. DATA LOADERS
# ============================================================

BATCH_SIZE = 1
NUM_WORKERS = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 8. A2 - 3D SNN U-NET + ATTENTION
# ============================================================

class DoubleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv3D(
            3, 16
        )

        self.enc2 = DoubleConv3D(
            16, 32
        )

        self.enc3 = DoubleConv3D(
            32, 64
        )

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # Bottleneck
        self.bottleneck = DoubleConv3D(
            64, 128
        )

        # Decoder
        self.up3 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv3D(
            128, 64
        )

        self.up2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv3D(
            64, 32
        )

        self.up1 = nn.ConvTranspose3d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv3D(
            32, 16
        )

        self.out_conv = nn.Conv3d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e3)
        )

        # Decoder
        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        out = self.out_conv(d1)

        return out


# ============================================================
# 9. CREATE MODEL
# ============================================================

model = A2SNNUNet3D().to(DEVICE)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n" + "=" * 70)
print("A2 MODEL")
print("=" * 70)

print("Architecture : 3D SNN U-Net")
print("SNN / LIF    : YES")
print("Attention    : YES")
print("Residual     : NO")
print(f"Parameters   : {trainable_params:,}")


# ============================================================
# 10. LOSS FUNCTIONS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(
            probs.size(0), -1
        )

        targets = targets.contiguous().view(
            targets.size(0), -1
        )

        intersection = (
            probs * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probs.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):

        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        return (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )


criterion = CombinedLoss(
    bce_weight=0.5,
    dice_weight=0.5
)


# ============================================================
# 11. OPTIMIZER
# ============================================================

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 12. VALIDATION DICE
# ============================================================

@torch.no_grad()
def calculate_dice(logits, targets):

    probs = torch.sigmoid(logits)

    preds = (
        probs >= 0.5
    ).float()

    intersection = (
        preds * targets
    ).sum()

    denominator = (
        preds.sum()
        + targets.sum()
    )

    if denominator.item() == 0:
        return 1.0

    dice = (
        2.0 * intersection
        /
        denominator
    )

    return dice.item()


@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    count = 0

    for images, masks, subjects in val_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            masks
        )

        dice = calculate_dice(
            logits,
            masks
        )

        total_loss += loss.item()
        total_dice += dice
        count += 1

    return (
        total_loss / count,
        total_dice / count
    )


# ============================================================
# 13. TRAINING
# ============================================================

EPOCHS = 50
best_val_dice = -1.0
best_epoch = -1

history = []

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "A2_best.pth"
)

print("\n" + "=" * 70)
print("TRAINING START")
print("=" * 70)

print(f"Epochs       : {EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay : {WEIGHT_DECAY}")
print(f"Seed         : {SEED}")



# ============================================================
# BLOCK 50 - 50 EPOCH PATIENT-WISE TRAINING CONTROLLER
# ============================================================

MAX_EPOCHS = 50
EPOCHS = MAX_EPOCHS
PATIENCE = 15

BLOCK50_IS_SNN = True

PREPROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "ISLES2022_preprocessed",
)

BLOCK50_CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "block50_epoch",
    "A2",
)

BLOCK50_RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "block50_epoch",
    "A2",
)

os.makedirs(
    BLOCK50_CHECKPOINT_DIR,
    exist_ok=True,
)

os.makedirs(
    BLOCK50_RESULT_DIR,
    exist_ok=True,
)

BLOCK50_BEST_CHECKPOINT = os.path.join(
    BLOCK50_CHECKPOINT_DIR,
    "A2_best.pth",
)

BLOCK50_LATEST_CHECKPOINT = os.path.join(
    BLOCK50_CHECKPOINT_DIR,
    "A2_latest.pth",
)

BLOCK50_HISTORY_FILE = os.path.join(
    BLOCK50_RESULT_DIR,
    "training_history.json",
)

BLOCK50_CONFIG_FILE = os.path.join(
    BLOCK50_RESULT_DIR,
    "training_config.json",
)

BLOCK50_SUMMARY_FILE = os.path.join(
    BLOCK50_RESULT_DIR,
    "training_summary.json",
)

BLOCK50_LOG_FILE = os.path.join(
    BLOCK50_RESULT_DIR,
    "training_log.txt",
)


def block50_timestamp():

    return datetime.now().isoformat(
        timespec="seconds"
    )


def block50_log(message):

    line = (
        f"[{block50_timestamp()}] "
        f"{message}"
    )

    print(line)

    with open(
        BLOCK50_LOG_FILE,
        "a",
    ) as f:

        f.write(line + "\n")


# ============================================================
# PATIENT-WISE VALIDATION DATASET
# ============================================================

patient_val_dataset = PatientWisePatchDataset(
    VAL_PATCH_DIR
)

patient_val_loader = DataLoader(
    patient_val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# CONFIGURATION
# ============================================================

with open(
    BLOCK50_CONFIG_FILE,
    "w",
) as f:

    json.dump(
        {
            "experiment":
                "A2_Block50_50Epoch_PatientWise",

            "model":
                "A2",

            "max_epochs":
                MAX_EPOCHS,

            "early_stopping_patience":
                PATIENCE,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "seed":
                SEED,

            "training_subjects":
                174,

            "validation_subjects":
                38,

            "validation_patches":
                len(patient_val_dataset),

            "test_subjects":
                38,

            "patient_wise_validation":
                True,

            "monitor":
                "mean_patient_level_validation_Dice",

            "test_used_for_training":
                False,

            "test_used_for_early_stopping":
                False,

            "snn":
                BLOCK50_IS_SNN,

            "single_step_lif":
                BLOCK50_IS_SNN,

            "created_timestamp":
                block50_timestamp(),
        },
        f,
        indent=2,
    )


# ============================================================
# RESUME STATE
# ============================================================

block50_start_epoch = 1

block50_best_val_dice = -1.0

block50_best_epoch = -1

block50_best_epoch_timestamp = None

block50_patience_counter = 0

block50_history = []

block50_resumed = False

block50_training_start_timestamp = (
    block50_timestamp()
)

block50_process_start_time = time.time()


# ============================================================
# RESUME FROM BLOCK 50 LATEST CHECKPOINT ONLY
# ============================================================

if os.path.exists(
    BLOCK50_LATEST_CHECKPOINT
):

    block50_log(
        "BLOCK 50 LATEST CHECKPOINT FOUND - RESUMING"
    )

    block50_checkpoint = torch.load(
        BLOCK50_LATEST_CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        block50_checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        block50_checkpoint[
            "optimizer_state_dict"
        ]
    )

    block50_start_epoch = (
        int(
            block50_checkpoint[
                "epoch"
            ]
        ) + 1
    )

    block50_best_val_dice = float(
        block50_checkpoint.get(
            "best_val_dice",
            -1.0,
        )
    )

    block50_best_epoch = int(
        block50_checkpoint.get(
            "best_epoch",
            -1,
        )
    )

    block50_best_epoch_timestamp = (
        block50_checkpoint.get(
            "best_epoch_timestamp",
            None,
        )
    )

    block50_patience_counter = int(
        block50_checkpoint.get(
            "patience_counter",
            0,
        )
    )

    block50_history = (
        block50_checkpoint.get(
            "history",
            [],
        )
    )

    block50_training_start_timestamp = (
        block50_checkpoint.get(
            "training_start_timestamp",
            block50_training_start_timestamp,
        )
    )

    block50_resumed = True

    block50_log(
        f"Resume epoch: {block50_start_epoch}"
    )

    block50_log(
        f"Previous best patient Dice: "
        f"{block50_best_val_dice:.6f}"
    )

else:

    block50_log(
        "No Block 50 latest checkpoint found - fresh training"
    )


block50_log(
    f"Training start: "
    f"{block50_training_start_timestamp}"
)

block50_log(
    f"Maximum epochs: {MAX_EPOCHS}"
)

block50_log(
    f"Early stopping patience: {PATIENCE}"
)

block50_log(
    "Validation monitor: "
    "mean patient-level Dice"
)

block50_log(
    "Internal held-out test set: NOT USED"
)


# ============================================================
# 100-EPOCH TRAINING LOOP
# ============================================================

for epoch in range(
    block50_start_epoch,
    MAX_EPOCHS + 1,
):

    epoch_start_time = time.time()

    epoch_start_timestamp = (
        block50_timestamp()
    )

    model.train()

    if BLOCK50_IS_SNN:
        functional.reset_net(model)

    running_loss = 0.0

    batch_count = 0

    train_bar = tqdm(
        train_loader,
        desc=(
            f"A2 "
            f"Epoch {epoch}/{MAX_EPOCHS} [TRAIN]"
        ),
    )

    for batch in train_bar:

        images = batch[0]

        masks = batch[1]

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True,
        )

        if masks.ndim == 4:

            masks = masks.unsqueeze(1)

        optimizer.zero_grad(
            set_to_none=True
        )

        if BLOCK50_IS_SNN:
            functional.reset_net(model)

        outputs = model(images)

        loss = criterion(
            outputs,
            masks,
        )

        loss.backward()

        optimizer.step()

        running_loss += float(
            loss.item()
        )

        batch_count += 1

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = (
        running_loss /
        max(batch_count, 1)
    )


    # ========================================================
    # PATIENT-WISE VALIDATION
    # ========================================================

    validation_start_timestamp = (
        block50_timestamp()
    )

    model.eval()

    if BLOCK50_IS_SNN:
        functional.reset_net(model)

    with torch.no_grad():

        patient_validation = patientwise_validation(
            model=model,
            val_loader=patient_val_loader,
            preprocessed_dir=PREPROCESSED_DIR,
            device=DEVICE,
            loss_function=criterion,
            is_snn=BLOCK50_IS_SNN,
        )

        patient_val_loss = float(
            patient_validation["val_loss"]
        )

        patient_val_dice = float(
            patient_validation["val_dice"]
        )

        patient_metrics = patient_validation


    validation_end_timestamp = (
        block50_timestamp()
    )


    # ========================================================
    # EARLY STOPPING STATE
    # ========================================================

    improved = (
        float(patient_val_dice)
        > float(block50_best_val_dice)
    )

    if improved:

        block50_best_val_dice = float(
            patient_val_dice
        )

        block50_best_epoch = epoch

        block50_best_epoch_timestamp = (
            validation_end_timestamp
        )

        block50_patience_counter = 0

    else:

        block50_patience_counter += 1


    epoch_end_timestamp = (
        block50_timestamp()
    )

    epoch_duration = (
        time.time() -
        epoch_start_time
    )


    # ========================================================
    # HISTORY
    # ========================================================

    epoch_record = {

        "epoch":
            int(epoch),

        "epoch_start_timestamp":
            epoch_start_timestamp,

        "validation_start_timestamp":
            validation_start_timestamp,

        "validation_end_timestamp":
            validation_end_timestamp,

        "epoch_end_timestamp":
            epoch_end_timestamp,

        "epoch_duration_seconds":
            float(epoch_duration),

        "train_loss":
            float(train_loss),

        "patient_val_loss":
            float(patient_val_loss),

        "patient_val_dice":
            float(patient_val_dice),

        "best_val_dice":
            float(block50_best_val_dice),

        "best_epoch":
            int(block50_best_epoch),

        "best_epoch_timestamp":
            block50_best_epoch_timestamp,

        "patience_counter":
            int(block50_patience_counter),

        "improved":
            bool(improved),
    }

    block50_history.append(
        epoch_record
    )


    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    if improved:

        torch.save(
            {
                "epoch":
                    int(epoch),

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_dice":
                    float(
                        block50_best_val_dice
                    ),

                "best_epoch":
                    int(
                        block50_best_epoch
                    ),

                "best_epoch_timestamp":
                    block50_best_epoch_timestamp,

                "patience_counter":
                    0,

                "history":
                    block50_history,

                "training_start_timestamp":
                    block50_training_start_timestamp,

                "saved_timestamp":
                    block50_timestamp(),

                "model":
                    "A2",

                "single_step_lif":
                    BLOCK50_IS_SNN,

                "test_used":
                    False,
            },
            BLOCK50_BEST_CHECKPOINT,
        )


    # ========================================================
    # LATEST CHECKPOINT
    # ========================================================

    torch.save(
        {
            "epoch":
                int(epoch),

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_val_dice":
                float(
                    block50_best_val_dice
                ),

            "best_epoch":
                int(
                    block50_best_epoch
                ),

            "best_epoch_timestamp":
                block50_best_epoch_timestamp,

            "patience_counter":
                int(
                    block50_patience_counter
                ),

            "history":
                block50_history,

            "training_start_timestamp":
                block50_training_start_timestamp,

            "saved_timestamp":
                block50_timestamp(),

            "model":
                "A2",

            "single_step_lif":
                BLOCK50_IS_SNN,

            "test_used":
                False,
        },
        BLOCK50_LATEST_CHECKPOINT,
    )


    # ========================================================
    # HISTORY FILE
    # ========================================================

    with open(
        BLOCK50_HISTORY_FILE,
        "w",
    ) as f:

        json.dump(
            block50_history,
            f,
            indent=2,
        )


    block50_log(
        f"Epoch {epoch}/{MAX_EPOCHS} | "
        f"train_loss={train_loss:.6f} | "
        f"patient_val_loss={patient_val_loss:.6f} | "
        f"patient_val_dice={patient_val_dice:.6f} | "
        f"best={block50_best_val_dice:.6f} | "
        f"patience={block50_patience_counter} | "
        f"duration={epoch_duration:.1f}s"
    )


    # ========================================================
    # EARLY STOP
    # ========================================================

    if block50_patience_counter >= PATIENCE:

        block50_early_stop_timestamp = (
            block50_timestamp()
        )

        block50_log(
            f"EARLY STOPPING at epoch {epoch}"
        )

        break


# ============================================================
# FINAL SUMMARY
# ============================================================

block50_training_end_timestamp = (
    block50_timestamp()
)

block50_total_seconds = (
    time.time() -
    block50_process_start_time
)

block50_summary = {

    "experiment":
        "A2_Block50_50Epoch_PatientWise",

    "model":
        "A2",

    "training_start_timestamp":
        block50_training_start_timestamp,

    "training_end_timestamp":
        block50_training_end_timestamp,

    "best_epoch":
        int(block50_best_epoch),

    "best_epoch_timestamp":
        block50_best_epoch_timestamp,

    "early_stop_timestamp":
        locals().get(
            "block50_early_stop_timestamp",
            None,
        ),

    "epochs_completed":
        len(block50_history),

    "max_epochs":
        MAX_EPOCHS,

    "patience":
        PATIENCE,

    "best_val_dice":
        float(block50_best_val_dice),

    "total_training_duration_seconds":
        float(block50_total_seconds),

    "resumed":
        bool(block50_resumed),

    "training_subjects":
        174,

    "validation_subjects":
        38,

    "validation_patches":
        len(patient_val_dataset),

    "test_subjects":
        38,

    "test_used_for_training":
        False,

    "test_used_for_early_stopping":
        False,

    "single_step_lif":
        BLOCK50_IS_SNN,

    "best_checkpoint":
        BLOCK50_BEST_CHECKPOINT,

    "latest_checkpoint":
        BLOCK50_LATEST_CHECKPOINT,
}

with open(
    BLOCK50_SUMMARY_FILE,
    "w",
) as f:

    json.dump(
        block50_summary,
        f,
        indent=2,
    )


block50_log(
    f"Training completed | "
    f"best_epoch={block50_best_epoch} | "
    f"best_patient_dice={block50_best_val_dice:.6f}"
)

block50_log(
    f"Training end: "
    f"{block50_training_end_timestamp}"
)


print()
print("=" * 70)
print("BLOCK 60 - TRAINING COMPLETE")
print("=" * 70)
print("Model             :", "A2")
print("Best epoch        :", block50_best_epoch)
print(
    "Best patient Dice :",
    f"{block50_best_val_dice:.6f}"
)
print(
    "Epochs completed  :",
    len(block50_history)
)
print("Max epochs        :", MAX_EPOCHS)
print("Patience          :", PATIENCE)
print("Resumed           :", block50_resumed)
print("Test used         : NO")
print(
    "Best checkpoint   :",
    BLOCK50_BEST_CHECKPOINT
)
print(
    "Latest checkpoint :",
    BLOCK50_LATEST_CHECKPOINT
)
print("=" * 70)
