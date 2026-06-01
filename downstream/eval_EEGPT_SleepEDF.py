"""
eval_EEGPT_SleepEDF.py
Supervised evaluation of the EEGPT encoder.

Two modes (set EVAL_MODE below):
  "linear_probe" : encoder FROZEN — measures representation quality in isolation.
  "finetune"     : entire encoder updated — higher ceiling, confounds representation quality.

"""

import os, sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Modules.models.EEGPT_mcae import EEGTransformer

# ── config ───────────────────────────────────────────────────────
TRAIN_ROOT   = os.getenv("TRAIN_ROOT")
VAL_ROOT     = os.getenv("VAL_ROOT")
TEST_ROOT    = os.getenv("TEST_ROOT")

ENCODER_PATH = "outputs/eegpt_theta_sleepedf_encoder.pt"   # ← swap per strategy

EVAL_MODE    = "linear_probe"   # "linear_probe" | "finetune"
N_CLASSES    = 5
N_CHANNELS   = 2
SFREQ        = 100
SEG_SECONDS  = 30
PATCH_SIZE   = 50
EMBED_DIM    = 512

BATCH_SIZE   = 64
MAX_EPOCHS   = 30
LR           = 1e-3 if EVAL_MODE == "linear_probe" else 1e-4
NUM_WORKERS  = 0


# ── dataset ──────────────────────────────────────────────────────
LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}

class SleepEDFLabelled(Dataset):
    def __init__(self, root: str):
        self.samples = []
        for label_name, label_idx in LABEL_MAP.items():
            d = os.path.join(root, label_name)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(d, f), label_idx))
        print(f"[Dataset] {root}: {len(self.samples)} labelled epochs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = torch.load(path, weights_only=False).float()
        return x, torch.tensor(label, dtype=torch.long)


# ── classifier ───────────────────────────────────────────────────
class EEGPTClassifier(pl.LightningModule):
    def __init__(self, eval_mode: str = EVAL_MODE):
        super().__init__()
        self.save_hyperparameters()
        self.eval_mode    = eval_mode
        self.val_outputs  = []
        self.test_outputs = []

        # ── encoder ───────────────────────────────────────────────
        self.encoder = EEGTransformer(
            img_size=[N_CHANNELS, SFREQ * SEG_SECONDS],
            patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, depth=8, num_heads=8)
        state = torch.load(ENCODER_PATH, map_location="cpu", weights_only=False)
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        print(f"[Encoder] {ENCODER_PATH} | missing={len(missing)}, unexpected={len(unexpected)}")

        if eval_mode == "linear_probe":
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[Eval] linear_probe: encoder is FROZEN")
        else:
            print("[Eval] finetune: encoder will be updated")

        self.head    = nn.Linear(EMBED_DIM, N_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if z.dim() == 4:
            z = z.squeeze(2)
        return self.head(z.mean(dim=1))

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y   = batch
        logits = self(x)
        self.log("val/loss", self.loss_fn(logits, y), prog_bar=True)
        self.val_outputs.append({"y": y.cpu(), "pred": logits.argmax(1).cpu()})

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        y_all    = torch.cat([o["y"]    for o in self.val_outputs]).numpy()
        pred_all = torch.cat([o["pred"] for o in self.val_outputs]).numpy()
        self.val_outputs.clear()
        acc   = accuracy_score(y_all, pred_all)
        f1    = f1_score(y_all, pred_all, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_all, pred_all)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1",  f1,  prog_bar=True)
        self.log("val/kappa", kappa)
        print(f"\n[Val] Acc={acc:.4f} | Macro-F1={f1:.4f} | Kappa={kappa:.4f}")

    def test_step(self, batch, _):
        x, y = batch
        self.test_outputs.append({"y": y.cpu(), "pred": self(x).argmax(1).cpu()})

    def on_test_epoch_end(self):
        y_all    = torch.cat([o["y"]    for o in self.test_outputs]).numpy()
        pred_all = torch.cat([o["pred"] for o in self.test_outputs]).numpy()
        self.test_outputs.clear()
        acc   = accuracy_score(y_all, pred_all)
        f1    = f1_score(y_all, pred_all, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_all, pred_all)
        self.log("test/acc", acc); self.log("test/f1", f1); self.log("test/kappa", kappa)
        print(f"\n[TEST] Acc={acc:.4f} | Macro-F1={f1:.4f} | Kappa={kappa:.4f}")
        print(  "[TEST] ← these are the numbers to report")

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR * 0.01)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ── run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_ds = SleepEDFLabelled(TRAIN_ROOT)
    val_ds   = SleepEDFLabelled(VAL_ROOT)
    test_ds  = SleepEDFLabelled(TEST_ROOT)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)

    strategy_name = os.path.basename(ENCODER_PATH).split("_")[1]
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_DIR := "outputs", "eval_ckpts"),
        filename=f"eegpt_{strategy_name}_best",
        monitor="val/acc", mode="max", save_top_k=1)

    trainer = pl.Trainer(
        accelerator="cuda", devices=1, max_epochs=MAX_EPOCHS,
        precision="16-mixed", log_every_n_steps=10, callbacks=[ckpt_cb])

    trainer.fit(EEGPTClassifier(), make_loader(train_ds, True), make_loader(val_ds, False))

    print(f"\n[Eval] Best checkpoint: {ckpt_cb.best_model_path}")
    print(f"[Eval] Best val/acc:    {ckpt_cb.best_model_score:.4f}")
    trainer.test(EEGPTClassifier(), make_loader(test_ds, False),
                 ckpt_path=ckpt_cb.best_model_path)
