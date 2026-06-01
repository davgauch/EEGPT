"""
pretrain_EEGPT_SleepEDF.py
Unsupervised domain adaptation of pretrained EEGPT on Sleep-EDF via masked autoencoding.
Reconstruction target: raw waveform per patch (time-domain MSE).
"""

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Modules.channel_aware_masking import apply_bandstop_mask
from Modules.models.EEGPT_mcae import EEGTransformer

# ── reproducibility ──────────────────────────────────────────────
def set_seed(seed: int = 7):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(7)

# ── config ───────────────────────────────────────────────────────
TRAIN_ROOT       = os.getenv("TRAIN_ROOT")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
OUTPUT_DIR      = "outputs"

MASK_STRATEGY = "theta"   # theta | delta | alpha | beta | random | none

SFREQ, N_CHANNELS, SEG_SECONDS = 100, 2, 30
PATCH_SIZE = 50           # 3000 / 50 = 60 patches/channel

BATCH_SIZE, MAX_EPOCHS, LR, WARMUP_EPOCHS = 32, 50, 1e-4, 5
NUM_WORKERS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────
def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, T) → (B, C*N_patches, patch_size)"""
    B, C, T = x.shape
    n = T // patch_size
    return x.reshape(B, C, n, patch_size).reshape(B, C * n, patch_size)


# ── dataset ──────────────────────────────────────────────────────
class MaskedSleepEDF(torch.utils.data.Dataset):
    """
    Returns (x_masked, x_original, patch_mask).
    patch_mask: (N_patches,) bool — True where time-domain energy was removed.
    """
    def __init__(self, root: str, strategy: str, sfreq: int = SFREQ,
                 patch_size: int = PATCH_SIZE):
        self.strategy   = strategy
        self.sfreq      = sfreq
        self.patch_size = patch_size
        self.files      = [
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

        # Channel-averaged energy difference per patch → binary mask
        diff = patchify(
            (x_orig - x_masked).mean(dim=0, keepdim=True).unsqueeze(0),
            self.patch_size).squeeze(0)                  # (N_patches, patch_size)
        patch_energy = diff.pow(2).mean(dim=-1)          # (N_patches,)
        patch_mask   = patch_energy > patch_energy.median()

        return x_masked, x_orig, patch_mask


# ── model ────────────────────────────────────────────────────────
class PatchDecoder(nn.Module):
    """(B, N, D) → (B, N, patch_size)"""
    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size),
        )
    def forward(self, z): return self.net(z)


class EEGPTPretrain(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.patch_size = PATCH_SIZE

        # ── encoder ───────────────────────────────────────────────
        self.encoder = EEGTransformer(
            img_size=[N_CHANNELS, SFREQ * SEG_SECONDS],
            patch_size=PATCH_SIZE, embed_dim=512, depth=8, num_heads=8)

        # Load target_encoder weights; skip shape mismatches
        ckpt  = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state = {k[len("target_encoder."):]: v
                 for k, v in ckpt["state_dict"].items()
                 if k.startswith("target_encoder.")}
        model_state = self.encoder.state_dict()
        state_fit   = {k: v for k, v in state.items()
                       if k in model_state and v.shape == model_state[k].shape}
        self.encoder.load_state_dict(state_fit, strict=False)
        print(f"[Encoder] Loaded {len(state_fit)} tensors")

        with torch.no_grad():
            dummy = torch.zeros(1, N_CHANNELS, SFREQ * SEG_SECONDS)
            n_tok = self.encoder(dummy).shape[1]
        n_patches        = N_CHANNELS * (SFREQ * SEG_SECONDS // PATCH_SIZE)
        self._use_tokens = min(n_tok, n_patches)

        self.decoder = PatchDecoder(512, PATCH_SIZE)
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if z.dim() == 4:
            z = z.squeeze(2)
        return self.decoder(z[:, :self._use_tokens, :])

    def training_step(self, batch, _):
        x_masked, x_orig, patch_mask = batch
        pred   = self(x_masked)                                   # (B, N, patch_size)
        target = patchify(x_orig.mean(dim=1, keepdim=True), self.patch_size)
        target = target[:, :self._use_tokens, :]
        pm     = patch_mask[:, :self._use_tokens].float()

        per_patch     = self.loss_fn(pred, target).mean(dim=-1)   # (B, N)
        masked_loss   = (per_patch * pm).sum()       / (pm.sum()       + 1e-6)
        unmasked_loss = (per_patch * (1 - pm)).sum() / ((1 - pm).sum() + 1e-6)
        loss = masked_loss + 0.1 * unmasked_loss

        self.log("train/loss", loss, prog_bar=True)
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
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    trainer = pl.Trainer(
        accelerator="cuda", devices=1, max_epochs=MAX_EPOCHS,
        precision="16-mixed", log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(dirpath=OUTPUT_DIR,
                            filename=f"eegpt_{MASK_STRATEGY}_{{epoch:02d}}_{{train/loss:.4f}}",
                            monitor="train/loss", save_top_k=3),
            LearningRateMonitor(logging_interval="step")])

    trainer.fit(EEGPTPretrain(), loader)

    out = os.path.join(OUTPUT_DIR, f"eegpt_{MASK_STRATEGY}_sleepedf_encoder.pt")
    torch.save(trainer.model.encoder.state_dict(), out)
    print(f"[Done] Encoder saved → {out}")
