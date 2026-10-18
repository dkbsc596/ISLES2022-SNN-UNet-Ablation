import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
#!/usr/bin/env python3

import os
import json
import time
import csv
import random
import numpy as np
import torch
from models.controlled_snn_unet import A1SNNUNet3D, A2SNNUNet3D, A3SNNUNet3D
import torch.nn as nn

# ============================================================
# EXACT A0 ARCHITECTURE
# Copied from final_internal_heldout_evaluation.py
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
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv3D(128, 64)

        self.up2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv3D(64, 32)

        self.up1 = nn.ConvTranspose3d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv3D(32, 16)

        self.out_conv = nn.Conv3d(
            16,
            1,
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

        return self.out_conv(d1)



# ============================================================
# ENERGY / COMPUTATIONAL EFFICIENCY ANALYSIS
# Block50 Epoch
#
# IMPORTANT:
# - READ-ONLY analysis
# - No training
# - No checkpoint modification
# - No dataset modification
# - Uses a fixed synthetic 3D input
# - Reports computational energy-efficiency proxies
# ============================================================

ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"
os.chdir(ROOT)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "block50_epoch",
    "energy_efficiency"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SHAPE = (1, 3, 128, 128, 64)

SEED = 42

WARMUP_RUNS = 5
TIMED_RUNS = 20


MODELS = {
    "A0": {
        "description": "3D CNN U-Net",
        "checkpoint": "checkpoints/block50_epoch/A0/A0_best.pth",
        "constructor": UNet3D,
        "snn": False,
    },

    "A1": {
        "description": "3D SNN U-Net",
        "checkpoint": "checkpoints/block50_epoch/A1/block50_epoch/A1/A1_best.pth",
        "constructor": A1SNNUNet3D,
        "snn": True,
    },

    "A2": {
        "description": "3D SNN U-Net + Attention",
        "checkpoint": "checkpoints/block50_epoch/A2/A2_best.pth",
        "constructor": A2SNNUNet3D,
        "snn": True,
    },

    "A3": {
        "description": "3D SNN U-Net + Attention + Residual",
        "checkpoint": "checkpoints/block50_epoch/A3/A3_best.pth",
        "constructor": A3SNNUNet3D,
        "snn": True,
    },
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("ENERGY / COMPUTATIONAL EFFICIENCY ANALYSIS")
print("=" * 70)
print(f"Project root : {ROOT}")
print(f"Device       : {DEVICE}")
print(f"Input shape  : {INPUT_SHAPE}")
print(f"Seed         : {SEED}")
print(f"Warmup runs  : {WARMUP_RUNS}")
print(f"Timed runs   : {TIMED_RUNS}")
print("=" * 70)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(info):

    checkpoint_path = os.path.join(
        ROOT,
        info["checkpoint"]
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = info["constructor"]()

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

    return model, checkpoint_path


# ============================================================
# PARAMETER COUNT
# ============================================================

def parameter_statistics(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# CHECKPOINT SIZE
# ============================================================

def checkpoint_size_mb(path):

    size_bytes = os.path.getsize(path)

    return (
        size_bytes,
        size_bytes / (1024 ** 2)
    )


# ============================================================
# INFERENCE LATENCY
# ============================================================

@torch.no_grad()
def measure_latency(model, x):

    # Warm-up
    for _ in range(WARMUP_RUNS):
        _ = model(x)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []

    for _ in range(TIMED_RUNS):

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(x)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        times.append(
            (end - start) * 1000.0
        )

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


# ============================================================
# GPU MEMORY
# ============================================================

@torch.no_grad()
def measure_gpu_memory(model, x):

    if DEVICE.type != "cuda":
        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _ = model(x)

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 2)
    )

    return {
        "allocated_mb": float(allocated),
        "reserved_mb": float(reserved),
        "peak_allocated_mb": float(peak_allocated),
        "peak_reserved_mb": float(peak_reserved),
    }


# ============================================================
# SPIKE RECORDING
# ============================================================

class SpikeRecorder:

    def __init__(self):

        self.handles = []
        self.total_spikes = 0
        self.total_elements = 0
        self.layer_stats = {}

    def hook(self, name):

        def forward_hook(module, inputs, output):

            try:

                tensor = output

                if isinstance(output, tuple):
                    tensor = output[0]

                if not torch.is_tensor(tensor):
                    return

                # SpikingJelly LIF outputs are generally
                # non-negative spike tensors.

                spikes = (tensor > 0).sum().item()
                elements = tensor.numel()

                self.total_spikes += spikes
                self.total_elements += elements

                if name not in self.layer_stats:

                    self.layer_stats[name] = {
                        "spikes": 0,
                        "elements": 0,
                        "calls": 0,
                    }

                self.layer_stats[name]["spikes"] += spikes
                self.layer_stats[name]["elements"] += elements
                self.layer_stats[name]["calls"] += 1

            except Exception:
                pass

        return forward_hook

    def register(self, model):

        for name, module in model.named_modules():

            # SpikingJelly LIF nodes expose these attributes.

            if (
                hasattr(module, "v_threshold")
                and hasattr(module, "tau")
                and hasattr(module, "step_mode")
            ):

                handle = module.register_forward_hook(
                    self.hook(name)
                )

                self.handles.append(handle)

    def remove(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []

    def summary(self):

        if self.total_elements == 0:

            return {
                "total_spikes": 0,
                "total_elements": 0,
                "mean_firing_rate": 0.0,
                "layer_stats": {},
            }

        layer_stats = {}

        for name, stats in self.layer_stats.items():

            rate = (
                stats["spikes"]
                / stats["elements"]
                if stats["elements"] > 0
                else 0.0
            )

            layer_stats[name] = {
                "spikes": int(stats["spikes"]),
                "elements": int(stats["elements"]),
                "calls": int(stats["calls"]),
                "firing_rate": float(rate),
            }

        return {
            "total_spikes": int(self.total_spikes),
            "total_elements": int(self.total_elements),
            "mean_firing_rate": float(
                self.total_spikes /
                self.total_elements
            ),
            "layer_stats": layer_stats,
        }


# ============================================================
# CONVOLUTION MAC ESTIMATION
# ============================================================

class MACCounter:

    def __init__(self):

        self.total_macs = 0
        self.layer_macs = {}

        self.handles = []

    def register(self, model):

        for name, module in model.named_modules():

            if isinstance(
                module,
                (nn.Conv3d, nn.ConvTranspose3d)
            ):

                handle = module.register_forward_hook(
                    self.make_hook(name, module)
                )

                self.handles.append(handle)

    def make_hook(self, name, module):

        def hook(module, inputs, output):

            if not torch.is_tensor(output):
                return

            batch = output.shape[0]
            out_channels = output.shape[1]

            spatial_output = int(
                np.prod(
                    output.shape[2:]
                )
            )

            kernel_ops = int(
                np.prod(
                    module.kernel_size
                )
            )

            in_channels = module.in_channels
            groups = module.groups

            ops_per_output = (
                kernel_ops
                * (in_channels // groups)
            )

            macs = (
                batch
                * out_channels
                * spatial_output
                * ops_per_output
            )

            self.total_macs += macs

            self.layer_macs[name] = (
                self.layer_macs.get(name, 0)
                + macs
            )

        return hook

    def remove(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []


# ============================================================
# ANALYSIS OF ONE MODEL
# ============================================================

@torch.no_grad()
def analyze_model(model_name, info, x):

    print()
    print("-" * 70)
    print(f"MODEL: {model_name}")
    print(info["description"])
    print("-" * 70)

    model, checkpoint_path = load_model(info)

    total_params, trainable_params = (
        parameter_statistics(model)
    )

    checkpoint_bytes, checkpoint_mb = (
        checkpoint_size_mb(checkpoint_path)
    )

    print(f"Parameters       : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print(f"Checkpoint size  : {checkpoint_mb:.3f} MB")

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency = measure_latency(
        model,
        x
    )

    print(
        f"Latency          : "
        f"{latency['mean_ms']:.3f} ms"
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    memory = measure_gpu_memory(
        model,
        x
    )

    if memory["peak_allocated_mb"] is not None:

        print(
            f"Peak GPU memory  : "
            f"{memory['peak_allocated_mb']:.2f} MB"
        )

    # --------------------------------------------------------
    # MAC counting
    # --------------------------------------------------------

    mac_counter = MACCounter()
    mac_counter.register(model)

    _ = model(x)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    total_macs = mac_counter.total_macs

    mac_counter.remove()

    print(
        f"Estimated MACs   : "
        f"{total_macs:,}"
    )

    # --------------------------------------------------------
    # Spike recording
    # --------------------------------------------------------

    spike_data = {
        "total_spikes": 0,
        "total_elements": 0,
        "mean_firing_rate": 0.0,
        "layer_stats": {},
    }

    if info["snn"]:

        recorder = SpikeRecorder()
        recorder.register(model)

        _ = model(x)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        spike_data = recorder.summary()

        recorder.remove()

        print(
            f"Total spikes     : "
            f"{spike_data['total_spikes']:,}"
        )

        print(
            f"Mean firing rate : "
            f"{spike_data['mean_firing_rate']:.6f}"
        )

    else:

        print(
            "Spike activity   : "
            "Not applicable (CNN)"
        )

    # --------------------------------------------------------
    # SNN computational proxy
    # --------------------------------------------------------

    if info["snn"]:

        firing_rate = spike_data[
            "mean_firing_rate"
        ]

        sop_proxy = (
            total_macs
            * firing_rate
        )

    else:

        sop_proxy = float(total_macs)

    print(
        f"SOP/MAC proxy    : "
        f"{sop_proxy:,.0f}"
    )

    result = {

        "model": model_name,

        "description": info["description"],

        "parameters": int(total_params),

        "trainable_parameters": int(
            trainable_params
        ),

        "checkpoint_bytes": int(
            checkpoint_bytes
        ),

        "checkpoint_mb": float(
            checkpoint_mb
        ),

        "latency_mean_ms": latency[
            "mean_ms"
        ],

        "latency_std_ms": latency[
            "std_ms"
        ],

        "latency_median_ms": latency[
            "median_ms"
        ],

        "latency_min_ms": latency[
            "min_ms"
        ],

        "latency_max_ms": latency[
            "max_ms"
        ],

        "gpu_allocated_mb": memory[
            "allocated_mb"
        ],

        "gpu_reserved_mb": memory[
            "reserved_mb"
        ],

        "gpu_peak_allocated_mb": memory[
            "peak_allocated_mb"
        ],

        "gpu_peak_reserved_mb": memory[
            "peak_reserved_mb"
        ],

        "estimated_macs": int(
            total_macs
        ),

        "total_spikes": int(
            spike_data["total_spikes"]
        ),

        "spike_elements": int(
            spike_data["total_elements"]
        ),

        "mean_firing_rate": float(
            spike_data["mean_firing_rate"]
        ),

        "sop_proxy": float(
            sop_proxy
        ),

        "is_snn": bool(
            info["snn"]
        ),

        "layer_spike_stats": spike_data[
            "layer_stats"
        ],

        "layer_mac_stats": {
            name: int(value)
            for name, value
            in mac_counter.layer_macs.items()
        },
    }

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# CREATE FIXED INPUT
# ============================================================

print()
print("Creating fixed synthetic input...")

torch.manual_seed(SEED)

x = torch.randn(
    INPUT_SHAPE,
    dtype=torch.float32,
    device=DEVICE
)

print(
    f"Input created: "
    f"{tuple(x.shape)}"
)


# ============================================================
# RUN ALL FOUR MODELS
# ============================================================

results = {}

for model_name, info in MODELS.items():

    results[model_name] = analyze_model(
        model_name,
        info,
        x
    )


# ============================================================
# RELATIVE PROXIES
# ============================================================

cnn_macs = results["A0"]["estimated_macs"]

for model_name in results:

    result = results[model_name]

    result["relative_macs_vs_A0"] = (
        result["estimated_macs"]
        / cnn_macs
        if cnn_macs > 0
        else None
    )

    result["relative_sop_proxy_vs_A0"] = (
        result["sop_proxy"]
        / results["A0"]["sop_proxy"]
        if results["A0"]["sop_proxy"] > 0
        else None
    )

    result["relative_latency_vs_A0"] = (
        result["latency_mean_ms"]
        / results["A0"]["latency_mean_ms"]
        if results["A0"]["latency_mean_ms"] > 0
        else None
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "energy_efficiency_summary.csv"
)

fieldnames = [
    "model",
    "description",
    "parameters",
    "trainable_parameters",
    "checkpoint_bytes",
    "checkpoint_mb",
    "estimated_macs",
    "mean_firing_rate",
    "total_spikes",
    "sop_proxy",
    "relative_macs_vs_A0",
    "relative_sop_proxy_vs_A0",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_median_ms",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "relative_latency_vs_A0",
]

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

    for model_name in [
        "A0",
        "A1",
        "A2",
        "A3"
    ]:

        row = results[model_name]

        writer.writerow({
            key: row.get(key)
            for key in fieldnames
        })


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "energy_efficiency_detailed.json"
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        {
            "experiment": (
                "Block50 computational "
                "energy-efficiency analysis"
            ),

            "device": str(DEVICE),

            "input_shape": INPUT_SHAPE,

            "seed": SEED,

            "warmup_runs": WARMUP_RUNS,

            "timed_runs": TIMED_RUNS,

            "results": results,

            "interpretation_note": (
                "SOP is a computational proxy "
                "based on estimated convolution "
                "MACs multiplied by measured mean "
                "LIF firing rate. It is not a direct "
                "measurement of electrical energy "
                "consumption."
            ),
        },
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("FINAL ENERGY / COMPUTATIONAL EFFICIENCY SUMMARY")
print("=" * 100)

print(
    f"{'Model':<8}"
    f"{'Params':>14}"
    f"{'MACs':>18}"
    f"{'FireRate':>12}"
    f"{'SOP Proxy':>18}"
    f"{'Latency(ms)':>15}"
)

print("-" * 100)

for model_name in [
    "A0",
    "A1",
    "A2",
    "A3"
]:

    r = results[model_name]

    print(
        f"{model_name:<8}"
        f"{r['parameters']:>14,}"
        f"{r['estimated_macs']:>18,}"
        f"{r['mean_firing_rate']:>12.6f}"
        f"{r['sop_proxy']:>18,.0f}"
        f"{r['latency_mean_ms']:>15.3f}"
    )

print("=" * 100)

print()
print("Relative computational proxies vs A0")
print("-" * 70)

for model_name in [
    "A0",
    "A1",
    "A2",
    "A3"
]:

    r = results[model_name]

    print(
        f"{model_name}: "
        f"MAC ratio={r['relative_macs_vs_A0']:.4f}, "
        f"SOP-proxy ratio="
        f"{r['relative_sop_proxy_vs_A0']:.4f}, "
        f"latency ratio="
        f"{r['relative_latency_vs_A0']:.4f}"
    )

print()
print("Output files:")
print(csv_path)
print(json_path)

print()
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)
