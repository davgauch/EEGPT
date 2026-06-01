# Sleep-EDF Adaptation

This is a fork of [BINE022/EEGPT](https://github.com/BINE022/EEGPT).

Four scripts were added to the `downstream/` folder to adapt two EEG foundation models
(**EEGPT** and **LaBraM**) to the Sleep-EDF dataset using unsupervised masked autoencoding,
then evaluate them with a linear probe

## What was added

```
downstream/
├── pretrain_EEGPT_SleepEDF.py      ← unsupervised adaptation of EEGPT
├── pretrain_LaBraM_SleepEDF.py     ← unsupervised adaptation of LaBraM
├── eval_EEGPT_SleepEDF.py          ← linear probe / finetune evaluation of EEGPT
├── eval_LaBraM_SleepEDF.py         ← linear probe / finetune evaluation of LaBraM
└── Modules/
    └── channel_aware_masking.py    ← band-stop masking utility (used by all four scripts)
```

Everything else in the repo is from the original EEGPT codebase and is used as-is
(model architectures, LaBraM backbone registration, dataset utilities).

---

## Setup

### 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### 2 — Set `.env` Variables

Create a `.env` file at the root of the project:

```env
# Dataset paths
DATA_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/TrainFold
TRAIN_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/TrainFold
VAL_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/ValidFold
TEST_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/TestFold

# Model checkpoints
LABRAM_CKPT=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/downstream/Modules/LaBraM/labram-base.pth
CHECKPOINT_PATH=/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
```

## Reproducing results

All four scripts are run from the **`downstream/`** folder.

```bash
cd downstream
```

### Step 1 — Unsupervised adaptation (pretraining)

Run once per model per masking strategy.
The script saves a `backbone.pt` / `encoder.pt` to `downstream/outputs/`.

```bash
python pretrain_EEGPT_SleepEDF.py
python pretrain_LaBraM_SleepEDF.py
```

### Step 2 — Supervised evaluation (linear probe)

Point `ENCODER_PATH` / `BACKBONE_PATH` to the `.pt` saved above, then run:

```bash
python eval_EEGPT_SleepEDF.py
python eval_LaBraM_SleepEDF.py
```

The script trains a linear head on TrainFold, selects the best epoch by `val/acc`,
and prints Accuracy / Macro-F1 / Cohen's κ on TestFold.

---

## Changing the masking strategy

### pretrain scripts

Open the file and change the constant on the indicated line:

| File | Line | Variable | Default |
|---|---|---|---|
| `pretrain_EEGPT_SleepEDF.py` | 29 | `MASK_STRATEGY` | `"theta"` |
| `pretrain_LaBraM_SleepEDF.py` | 33 | `MASK_STRATEGY` | `"theta"` |

Available strategies:

| Value | Band removed |
|---|---|
| `"theta"` | 4 – 8 Hz |
| `"delta"` | 1 – 5 Hz |
| `"alpha"` | 8 – 12 Hz |
| `"beta"` | 13 – 30 Hz |
| `"beta_upper"` | 20 – 30 Hz |
| `"random"` | random 4 Hz window each sample |
| `"none"` | no masking (reconstruction baseline) |
| `"theta_bw1"` … `"theta_bw12"` | bandwidth grid search from 4 Hz |

### eval scripts

After pretraining, update the path constant to point at the `.pt` you want to evaluate:

| File | Line | Variable | Example |
|---|---|---|---|
| `eval_EEGPT_SleepEDF.py` | 29 | `ENCODER_PATH` | `"outputs/eegpt_theta_sleepedf_encoder.pt"` |
| `eval_LaBraM_SleepEDF.py` | 33 | `BACKBONE_PATH` | `"outputs/labram_theta_sleepedf_backbone.pt"` |

---

## Full run example (theta strategy, both models)

```bash
cd downstream

# 1. Adapt
python pretrain_EEGPT_SleepEDF.py    # saves outputs/eegpt_theta_sleepedf_encoder.pt
python pretrain_LaBraM_SleepEDF.py   # saves outputs/labram_theta_sleepedf_backbone.pt

# 2. Evaluate  (ENCODER_PATH / BACKBONE_PATH already default to theta)
python eval_EEGPT_SleepEDF.py
python eval_LaBraM_SleepEDF.py
```

Final test numbers are printed at the end of each eval run:
```
[TEST] Acc=0.XXXX | Macro-F1=0.XXXX | Kappa=0.XXXX
```

---

## Original EEGPT paper

> *EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals*
> [`BINE022/EEGPT`](https://github.com/BINE022/EEGPT)
