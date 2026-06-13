# Sleep-EDF Adaptation of EEGPT and LaBraM models

This is a fork of [BINE022/EEGPT](https://github.com/BINE022/EEGPT).

Six files were added to the `downstream/` folder to adapt two EEG foundation models
(**EEGPT** and **LaBraM**) to Sleep-EDF via unsupervised masked autoencoding, evaluate
them with a linear probe, run multi-seed significance tests, and analyse intrinsic
representation quality.

---

## What was added

```
downstream/
├── pretrain_EEGPT_SleepEDF.py      ← unsupervised adaptation of EEGPT
├── pretrain_LaBraM_SleepEDF.py     ← unsupervised adaptation of LaBraM
├── eval_EEGPT_SleepEDF.py          ← linear probe / finetune evaluation of EEGPT
├── eval_LaBraM_SleepEDF.py         ← linear probe / finetune evaluation of LaBraM
├── run_significance_test.py        ← multi-seed significance test (both models)
├── analyze_representations.py      ← unsupervised representation quality analysis
└── Modules/
    └── channel_aware_masking.py    ← band-stop masking utility (used by all scripts)
```

Everything else is from the original EEGPT codebase (architectures, LaBraM
registration, dataset utilities) and is used unchanged.

---

## Setup

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Set .env Variables

Create a `.env` file in the `downstream/` folder with the paths below.
All weights and datasets are already on the server — no downloads needed.

```env
TRAIN_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/TrainFold
VAL_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/ValidFold
TEST_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/TestFold
CHECKPOINT_PATH=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
LABRAM_CKPT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/downstream/Modules/LaBraM/labram-base.pth
NUM_WORKERS=8
```

---

## Running

All scripts are run from the **`downstream/`** folder.

```bash
cd downstream
```

### Step 1 — Unsupervised adaptation

```bash
python pretrain_EEGPT_SleepEDF.py  --strategy theta
python pretrain_LaBraM_SleepEDF.py --strategy theta
```

Saves `outputs/eegpt_theta_seed42_encoder.pt` and `outputs/labram_theta_seed42_backbone.pt`.
The best checkpoint by `train/loss` is used. The epoch `.ckpt` is kept but only
the encoder/backbone weights are exported to the `.pt` file.

### Step 2 — Supervised evaluation

```bash
python eval_EEGPT_SleepEDF.py  --encoder_path  outputs/eegpt_theta_seed42_encoder.pt
python eval_LaBraM_SleepEDF.py --backbone_path outputs/labram_theta_seed42_backbone.pt
```

Trains a linear head on TrainFold, selects the best epoch by `val/acc`, and tests
on TestFold. Final numbers (Accuracy / Macro-F1 / Cohen's κ) are printed and saved
as a JSON file alongside the weights for use by the significance test.

---

## Masking strategies

Pass any strategy name to `--strategy`:

| Value | Band removed |
|---|---|
| `theta` | 4 – 8 Hz |
| `delta` | 1 – 4 Hz |
| `alpha` | 8 – 12 Hz |
| `beta` | 13 – 30 Hz |
| `beta_upper` | 20 – 30 Hz |
| `random` | random 4 Hz window, resampled each sample |
| `none` | no masking — reconstruction baseline |
| `theta_bw1` … `theta_bw12` | bandwidth grid search from 4 Hz |

---

## Significance testing

`run_significance_test.py` One full pretrain + eval
per (strategy, seed) pair. For each run it trains a fresh encoder if the weights
don't already exist, runs the linear probe evaluation if the JSON results don't
already exist, then collects all results and runs pairwise **paired t-test** +
**Wilcoxon signed-rank test**.

```bash
# Compare theta vs random vs none, 5 seeds, EEGPT
python run_significance_test.py \
    --model eegpt \
    --strategies theta random none \
    --n_seeds 5

# Same for LaBraM
python run_significance_test.py \
    --model labram \
    --strategies theta random none \
    --n_seeds 5
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--model` | required | `eegpt` or `labram` |
| `--strategies` | required | one or more strategy names |
| `--n_seeds` | 5 | number of seeds (seeds: start, start+1, …) |
| `--start_seed` | 42 | first seed value |
| `--pretrain_epochs` | 30 | epochs per pretrain run (use 50 for final results) |
| `--eval_mode` | `linear_probe` | `linear_probe` or `finetune` |
| `--output_dir` | `outputs` | where weights and eval JSONs are saved |
| `--results_dir` | `significance_results` | where the summary JSON is saved |
| `--skip_training` | off | skip pretrain; only (re-)evaluate existing weights |
| `--retrain_strategies` | none | force retrain specific strategies even if weights exist |

---

## Representation analysis

`analyze_representations.py` evaluates intrinsic representation quality without
labels. It is a purely unsupervised check of whether different masking strategies
create meaningfully different embedding spaces.

**One encoder per strategy** is sufficient for a qualitative comparison.
If you have already run `run_significance_test.py`, pass all seed values to
average metrics across seeds for a more robust estimate (mean ± std).

```bash
# Single encoder per strategy (default seed 42)
python analyze_representations.py --model eegpt  --strategies theta random none delta alpha
python analyze_representations.py --model labram --strategies theta random none delta alpha

# Reuse encoders already trained by run_significance_test.py (5 seeds)
python analyze_representations.py --model eegpt  --strategies theta random none \
    --seeds 42 43 44 45 46
```

Five metrics are reported for each strategy:

| Metric | Direction | What it measures |
|---|---|---|
| Silhouette | ↑ | Cluster compactness vs separation |
| kNN accuracy | ↑ | Non-parametric class separability |
| W/B ratio | ↓ | Within-class / between-class variance |
| Collapse | ↑ | Mean std of embedding dims (0 = collapsed) |
| Separation | ↑ | Mean centroid-to-centroid distance |

Results are saved to `outputs/{model}_representation_quality.json`.

---

## Hardware

All experiments run on a single NVIDIA GPU with mixed precision (`float16`).

---

## Original paper

> *EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals*
> [`BINE022/EEGPT`](https://github.com/BINE022/EEGPT)