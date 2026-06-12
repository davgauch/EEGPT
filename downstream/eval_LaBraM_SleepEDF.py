"""Supervised evaluation of the LaBraM backbone adapted by pretrain_LaBraM_SleepEDF.py.

Two modes (--eval_mode):
  linear_probe : backbone FROZEN 
  finetune     : entire backbone updated
Usage:
    python eval_LaBraM_SleepEDF.py --backbone_path outputs/labram_theta_seed42_backbone.pt --seed 42
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Modules.LaBraM.modeling_finetune          # registers 'labram_base_patch200_200'
from timm.models import create_model

torch.set_float32_matmul_precision("medium")

TRAIN_ROOT = os.getenv("TRAIN_ROOT")
VAL_ROOT   = os.getenv("VAL_ROOT")
TEST_ROOT  = os.getenv("TEST_ROOT")

N_CLASSES   = 5
N_CHANNELS  = 2
SFREQ       = 200
SEG_SECONDS = 15
PATCH_LEN   = 200
N_PATCHES   = SEG_SECONDS   # 15
BATCH_SIZE  = 64
MAX_EPOCHS  = 30
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}


def to_labram_input(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) → (B, C, N_patches, patch_len)  — zero-mean then reshaped."""
    x = x - x.mean(dim=-1, keepdim=True)
    x = F.interpolate(x, SFREQ * SEG_SECONDS, mode="nearest")
    return x.reshape(x.shape[0], x.shape[1], N_PATCHES, PATCH_LEN)


class SleepEDFLabelled(Dataset):
    def __init__(self, root: str):
        self.samples = []
        for label_name, label_idx in LABEL_MAP.items():
            d = os.path.join(root, label_name)
            if not os.path.isdir(d): continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(d, f), label_idx))
        print(f"[Dataset] {root}: {len(self.samples)} labelled epochs")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return torch.load(path, weights_only=False).float(), \
               torch.tensor(label, dtype=torch.long)


class LaBraMClassifier(pl.LightningModule):
    def __init__(self, backbone_path: str, eval_mode: str = "linear_probe",
                 seed: int = 42):
        super().__init__()
        self.save_hyperparameters()   # saves backbone_path, eval_mode, seed
        self.val_outputs  = []
        self.test_outputs = []
        self._test_metrics = {}

        self.backbone = create_model(
            "labram_base_patch200_200",
            qkv_bias=False, rel_pos_bias=True, num_classes=0,
            drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
            drop_block_rate=None, use_mean_pooling=True,  
            init_scale=0.001, use_rel_pos_bias=True,
            use_abs_pos_emb=True, init_values=0.1,
        )
        state = torch.load(backbone_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        print(f"[Backbone] {backbone_path} | "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

        if eval_mode == "linear_probe":
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[Eval] linear_probe: backbone is FROZEN")
        else:
            print("[Eval] finetune: backbone will be updated")

        with torch.no_grad():
            dummy   = torch.zeros(1, N_CHANNELS, N_PATCHES, PATCH_LEN)
            out_dim = self.backbone(dummy, input_chans=list(range(N_CHANNELS + 1))).shape[-1]
        print(f"[Backbone] Embed dim: {out_dim}")

        lr = 1e-3 if eval_mode == "linear_probe" else 4e-4
        self._lr     = lr
        self.head    = nn.Linear(out_dim, N_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = to_labram_input(x)
        z    = self.backbone(x_in, input_chans=list(range(x.shape[1] + 1)))
        if z.dim() == 3: z = z.mean(dim=1)
        return self.head(z)

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
        if not self.val_outputs: return
        y_all    = torch.cat([o["y"]    for o in self.val_outputs]).numpy()
        pred_all = torch.cat([o["pred"] for o in self.val_outputs]).numpy()
        self.val_outputs.clear()
        acc   = accuracy_score(y_all, pred_all)
        f1    = f1_score(y_all, pred_all, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_all, pred_all)
        self.log("val/acc",   acc,   prog_bar=True)
        self.log("val/f1",    f1)
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

        # Save for significance test (alongside backbone file)
        self._test_metrics = {"accuracy": acc, "f1_macro": f1, "kappa": kappa,
                              "backbone_path": self.hparams.backbone_path,
                              "eval_mode": self.hparams.eval_mode,
                              "seed": self.hparams.seed}
        json_path = self.hparams.backbone_path.replace("_backbone.pt", "_eval.json")
        with open(json_path, "w") as fp:
            json.dump(self._test_metrics, fp, indent=2)
        print(f"[Saved] Results → {json_path}")

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=self._lr, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=MAX_EPOCHS, eta_min=self._lr * 0.01)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LaBraM on Sleep-EDF")
    parser.add_argument("--backbone_path", required=True,
                        help="Path to backbone .pt saved by pretrain_LaBraM_SleepEDF.py")
    parser.add_argument("--eval_mode", default="linear_probe",
                        choices=["linear_probe", "finetune"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for head init")
    args = parser.parse_args()

    assert TRAIN_ROOT, "TRAIN_ROOT is not set — create a .env file (see .env.example)"
    assert VAL_ROOT,   "VAL_ROOT is not set"
    assert TEST_ROOT,  "TEST_ROOT is not set"
    assert os.path.isfile(args.backbone_path), \
        f"Backbone not found: {args.backbone_path}  — run pretrain_LaBraM_SleepEDF.py first"

    torch.manual_seed(args.seed)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=NUM_WORKERS > 0)

    train_loader = make_loader(SleepEDFLabelled(TRAIN_ROOT), True)
    val_loader   = make_loader(SleepEDFLabelled(VAL_ROOT),   False)
    test_loader  = make_loader(SleepEDFLabelled(TEST_ROOT),  False)

    model = LaBraMClassifier(backbone_path=args.backbone_path,
                              eval_mode=args.eval_mode, seed=args.seed)

    strategy_tag = os.path.basename(args.backbone_path).split("_")[1]
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join("outputs", "eval_ckpts"),
        filename=f"labram_{strategy_tag}_seed{args.seed}_best",
        monitor="val/acc", mode="max", save_top_k=1)

    trainer = pl.Trainer(
        accelerator="cuda", devices=1, max_epochs=MAX_EPOCHS,
        precision="16-mixed", log_every_n_steps=10, callbacks=[ckpt_cb])

    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        print(f"\n[Eval] Loading best checkpoint (val/acc={ckpt_cb.best_model_score:.4f})")
        print(f"[Eval] {best_ckpt}")
    else:
        print("\n[Eval] No checkpoint saved — testing with last epoch model")
        best_ckpt = None

    trainer.test(model, test_loader, ckpt_path=best_ckpt)
