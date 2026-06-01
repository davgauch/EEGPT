"""
pretrain_LaBraM_SleepEDF.py
Unsupervised domain adaptation of pretrained LaBraM on Sleep-EDF via masked autoencoding.
Reconstruction target: per-channel per-patch FFT amplitude (z-score normalised).
"""

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Modules.channel_aware_masking import apply_bandstop_mask
import Modules.LaBraM.modeling_finetune          # registers 'labram_base_patch200_200'
from timm.models import create_model

# ── reproducibility ──────────────────────────────────────────────
def set_seed(seed: int = 7):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(7)

torch.set_float32_matmul_precision("medium")

# ── config ───────────────────────────────────────────────────────
TRAIN_ROOT   = os.getenv("TRAIN_ROOT")
LABRAM_CKPT  = os.getenv("LABRAM_CKPT")
OUTPUT_DIR   = "outputs"

MASK_STRATEGY = "theta"   # theta | delta | alpha | beta | random | none

SFREQ         = 200
SEG_SECONDS   = 15
PATCH_LEN     = 200
N_PATCHES     = SEG_SECONDS               # 15 patches/channel  (15 × 200 = 3 000 samples)
N_CHANNELS    = 2
N_TOK         = N_CHANNELS * N_PATCHES    # 30 total patch tokens
FFT_BINS      = PATCH_LEN // 2 + 1       # 101 frequency bins

BATCH_SIZE    = 16
MAX_EPOCHS    = 50
LR            = 1e-4
WARMUP_EPOCHS = 5
NUM_WORKERS   = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────
def to_labram_input(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) → (B, C, N_patches, patch_len)  — zero-mean then reshaped."""
    B, C, T = x.shape
    x = x - x.mean(dim=-1, keepdim=True)
    x = F.interpolate(x, SFREQ * SEG_SECONDS, mode="nearest")
    return x.reshape(B, C, N_PATCHES, PATCH_LEN)


def fft_amplitude_target(x_patches: torch.Tensor) -> torch.Tensor:
    """
    x_patches : (B, C, N_patches, patch_len)
    returns   : (B, C*N_patches, FFT_BINS)  — z-score normalised FFT amplitude.
    Z-scoring equalises loss contribution across all bins (EEG has 1/f spectrum).
    """
    B, C, N, P = x_patches.shape
    x_flat = x_patches.reshape(B, C * N, P)
    amp    = torch.fft.rfft(x_flat, dim=-1).abs()
    mu     = amp.mean(dim=-1, keepdim=True)
    std    = amp.std(dim=-1, keepdim=True) + 1e-6
    return (amp - mu) / std


# ── dataset ──────────────────────────────────────────────────────
class MaskedSleepEDF(torch.utils.data.Dataset):
    """
    Returns (x_masked, x_original, patch_mask).
    patch_mask: (C*N_patches,) = (30,) bool — True where spectral energy was removed.
    """
    def __init__(self, root: str, strategy: str, sfreq: int = SFREQ):
        self.strategy = strategy
        self.sfreq    = sfreq
        self.files    = [
            os.path.join(root, lbl, f)
            for lbl in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, lbl))
            for f in sorted(os.listdir(os.path.join(root, lbl)))
            if f.endswith(".pt")
        ]
        print(f"[Dataset] {len(self.files)} epochs in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_orig = torch.load(self.files[idx], weights_only=False).float()   # (C, T)

        if self.strategy != "none":
            x_masked = torch.tensor(
                apply_bandstop_mask(x_orig.numpy(), self.strategy, self.sfreq),
                dtype=torch.float32)
        else:
            x_masked = x_orig.clone()

        # Per-token mask: True = this channel-patch lost significant spectral energy
        def _to_patches(x):
            xr = F.interpolate(x.unsqueeze(0), SFREQ * SEG_SECONDS,
                               mode="nearest").squeeze(0)           # (C, 3000)
            return xr.reshape(xr.shape[0], N_PATCHES, PATCH_LEN)   # (C, 15, 200)

        p_orig, p_masked = _to_patches(x_orig), _to_patches(x_masked)
        amp_diff     = (torch.fft.rfft(p_orig, dim=-1).abs() -
                        torch.fft.rfft(p_masked, dim=-1).abs())     # (C, 15, FFT_BINS)
        patch_energy = amp_diff.abs().mean(dim=-1).reshape(-1)      # (30,)
        patch_mask   = patch_energy > patch_energy.median()          # (30,) bool

        return x_masked, x_orig, patch_mask


# ── model ────────────────────────────────────────────────────────
class PatchDecoder(nn.Module):
    """(B, N, D) → (B, N, FFT_BINS)"""
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_dim),
        )
    def forward(self, z): return self.net(z)


class LaBraMPretrain(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # ── backbone ──────────────────────────────────────────────
        self.backbone = create_model(
            "labram_base_patch200_200",
            qkv_bias=False, rel_pos_bias=True, num_classes=0,
            drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
            drop_block_rate=None, use_mean_pooling=False,
            init_scale=0.001, use_rel_pos_bias=True,
            use_abs_pos_emb=True, init_values=0.1,
        )

        # Load pretrained weights; skip shape mismatches (patch embed, spatial embed)
        ckpt      = torch.load(LABRAM_CKPT, weights_only=False)
        state     = {k[len("student."):]: v
                     for k, v in ckpt["model"].items()
                     if k.startswith("student.")}
        model_state = self.backbone.state_dict()
        state_fit   = {k: v for k, v in state.items()
                       if k in model_state and v.shape == model_state[k].shape}
        missing, _  = self.backbone.load_state_dict(state_fit, strict=False)
        print(f"[Backbone] Loaded {len(state_fit)} tensors | missing {len(missing)}")

        # ── hook: capture all tokens from last transformer block ──
        # use_mean_pooling=False keeps CLS+patch tokens alive, but the backbone's
        # final head still discards the sequence.  The hook grabs it beforehand.
        self._hook_output = None
        self.backbone.blocks[-1].register_forward_hook(
            lambda m, i, o: setattr(self, "_hook_output", o))

        with torch.no_grad():
            dummy = torch.zeros(2, N_CHANNELS, N_PATCHES, PATCH_LEN)
            self.backbone(dummy, input_chans=list(range(N_CHANNELS + 1)))
            n_all     = self._hook_output.shape[1]   # CLS + patch tokens
            embed_dim = self._hook_output.shape[2]
            assert n_all - 1 == N_TOK, \
                f"Expected {N_TOK} patch tokens, got {n_all - 1}"
        print(f"[Hook] Tokens: {n_all} (1 CLS + {N_TOK} patch) | dim={embed_dim}")

        self.decoder = PatchDecoder(embed_dim, FFT_BINS)
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = to_labram_input(x)                                    # (B, C, 15, 200)
        self.backbone(x_in, input_chans=list(range(x.shape[1] + 1)))
        patch_toks = self._hook_output[:, 1:, :]                     # drop CLS → (B, 30, D)
        return self.decoder(patch_toks)                              # (B, 30, FFT_BINS)

    def training_step(self, batch, _):
        x_masked, x_orig, patch_mask = batch
        pred   = self(x_masked)                                      # (B, 30, FFT_BINS)
        target = fft_amplitude_target(to_labram_input(x_orig))       # (B, 30, FFT_BINS)
        pm     = patch_mask.float()                                  # (B, 30)

        per_patch     = self.loss_fn(pred, target).mean(dim=-1)      # (B, 30)
        masked_loss   = (per_patch * pm).sum()       / (pm.sum()       + 1e-6)
        unmasked_loss = (per_patch * (1 - pm)).sum() / ((1 - pm).sum() + 1e-6)
        loss = masked_loss + 0.1 * unmasked_loss

        self.log("train/loss",          loss,         prog_bar=True)
        self.log("train/masked_loss",   masked_loss)
        self.log("train/unmasked_loss", unmasked_loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=0.05)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=WARMUP_EPOCHS / MAX_EPOCHS)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


# ── run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = torch.utils.data.DataLoader(
        MaskedSleepEDF(TRAIN_ROOT, MASK_STRATEGY),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)

    trainer = pl.Trainer(
        accelerator="cuda", devices=1, max_epochs=MAX_EPOCHS,
        precision="16-mixed", log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(dirpath=OUTPUT_DIR,
                            filename=f"labram_{MASK_STRATEGY}_{{epoch:02d}}_{{train/loss:.4f}}",
                            monitor="train/loss", save_top_k=3),
            LearningRateMonitor(logging_interval="step")])

    trainer.fit(LaBraMPretrain(), loader)

    out = os.path.join(OUTPUT_DIR, f"labram_{MASK_STRATEGY}_sleepedf_backbone.pt")
    torch.save(trainer.model.backbone.state_dict(), out)
    print(f"[Done] Backbone saved → {out}")
