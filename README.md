# HYBRID-CSNN: Controlled 3D SNN U-Net Ablation for Ischemic Stroke Lesion Segmentation (ISLES 2022)

Code for a controlled ablation (A0-A3) of a 3D U-Net and spiking-neural-network (SNN) U-Net variants for ischemic stroke lesion segmentation on ISLES 2022 (DWI, ADC, FLAIR). All results are reported at the patient level on a locked held-out test set.

## Model configurations

| ID | Model | SNN | Attention | Residual |
|----|-------|-----|-----------|----------|
| A0 | 3D CNN U-Net | No | No | No |
| A1 | 3D SNN U-Net | Yes | No | No |
| A2 | 3D SNN U-Net + attention gates | Yes | Yes | No |
| A3 | 3D SNN U-Net + attention + residual spiking blocks | Yes | Yes | Yes |

SNN models use LIF neurons (tau = 2.0) with T = 4 time steps (SpikingJelly).

## Experimental protocol

- Dataset: ISLES 2022 (250 subjects), patient-level split 174 / 38 / 38 (train / validation / held-out test).
- Input: 3 channels (DWI, ADC, FLAIR); 3D patches of 128 x 128 x 64.
- Training: lesion-centered patch sampling (about 75% lesion-centered, 25% random), batch size 1, up to 50 epochs, early-stopping patience 15, seed 42.
- Validation / test: sliding-window inference, patches reconstructed to full patient volumes, threshold 0.5.
- Metrics (per patient): Dice, IoU, precision, recall, HD95 (mm).
- Statistics: bootstrap 95% CI, paired Wilcoxon signed-rank tests, Holm correction, rank-biserial effect size.

## Data

Download ISLES 2022 from Zenodo: https://doi.org/10.5281/zenodo.7960856

The dataset is not included in this repository.

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Repository structure

| File | Purpose |
|------|---------|
| `A0_train.py` ... `A3_train.py` | Training scripts for A0-A3 |
| `common_patientwise.py` | Patch dataset, patient-level reconstruction, patient-wise validation |
| `final_checkpoint_smoke_test.py` | Loads all four checkpoints and checks output shape |
| `final_internal_heldout_evaluation.py` | Final held-out test evaluation |
| `energy_efficiency_analysis.py` | Parameters, MACs, firing rate, SOP proxy, latency |
| `reproduce_patient_statistics.py` | Bootstrap CI, Wilcoxon, Holm, effect sizes |
| `sensitivity_exclude_empty.py` | Sensitivity analysis excluding patients with empty ground truth |
| `create_master_publication_results.py` | Master results table |
| `create_publication_tables.py` | Publication tables 1-6 |
| `create_publication_figures.py` | Figures 1-5 |
| `create_additional_publication_figures.py` | Figures 7-9 |
| `create_qualitative_publication_figure.py` | Qualitative comparison figure |
| `create_methodology_diagrams.py` | Methodology diagrams |
| `final_publication_consistency_audit.py` | Checks tables, figures and logs agree numerically |
| `models/controlled_snn_unet.py` | A1 / A2 / A3 SNN U-Net definitions |

## Suggested run order

```bash
# 1. Train
python A0_train.py
python A1_train.py
python A2_train.py
python A3_train.py

# 2. Sanity check checkpoints, then evaluate on the held-out test set
python final_checkpoint_smoke_test.py
python final_internal_heldout_evaluation.py

# 3. Efficiency and statistics
python energy_efficiency_analysis.py
python reproduce_patient_statistics.py
python sensitivity_exclude_empty.py

# 4. Tables and figures
python create_master_publication_results.py
python create_publication_tables.py
python create_publication_figures.py
python create_additional_publication_figures.py
python create_qualitative_publication_figure.py
python create_methodology_diagrams.py

# 5. Consistency audit
python final_publication_consistency_audit.py
```

## Configuration

Several scripts define the project path at the top (`ROOT` or `PROJECT_ROOT`). Set it to the location of your own copy before running.

## Reproducibility notes

- Fixed seed (42) for Python, NumPy and PyTorch.
- Test subjects are fixed before final evaluation and are not used for training, hyperparameter selection or early stopping.
- Trained checkpoints and patient-level prediction files are not included in this repository because of size.

## Citation

If you use this code, please cite the associated paper (details to be added on publication).

## License

Add a license (for example MIT) before making the repository public.
