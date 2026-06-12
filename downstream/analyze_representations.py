"""
analyze_representations.py
Evaluates intrinsic representation quality of adapted encoders across masking
strategies (no labels used during analysis).

Metrics used:
  silhouette     : cluster quality in embedding space           (↑ better)
  knn_accuracy   : 5-NN accuracy — non-parametric separability (↑ better)
  within_between : within-class / between-class variance       (↓ better)
  collapse       : mean std of embedding dimensions            (↑ less collapse)
  separation     : mean centroid-to-centroid distance          (↑ better)

Usage:
    python analyze_representations.py --model eegpt  --strategies theta random none \\
        --seeds 42 43 44 45 46
"""

import argparse, json, os, sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Modules.models.EEGPT_mcae import EEGTransformer
import Modules.LaBraM.modeling_finetune          
from timm.models import create_model


TEST_ROOT = os.getenv("TEST_ROOT")

#  EEGPT constants
EEGPT_N_CHANNELS  = 2
EEGPT_SFREQ       = 100
EEGPT_SEG_SECONDS = 30
EEGPT_PATCH_SIZE  = 50
EEGPT_EMBED_DIM   = 512

# LaBraM constants
LABRAM_N_CHANNELS  = 2
LABRAM_SFREQ       = 200
LABRAM_SEG_SECONDS = 15
LABRAM_PATCH_LEN   = 200
LABRAM_N_PATCHES   = LABRAM_SEG_SECONDS   # 15

LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}


# LaBraM input preprocessing
def to_labram_input(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) → (B, C, N_patches, patch_len) — zero-mean then reshaped."""
    x = x - x.mean(dim=-1, keepdim=True)
    x = F.interpolate(x, LABRAM_SFREQ * LABRAM_SEG_SECONDS, mode="nearest")
    return x.reshape(x.shape[0], x.shape[1], LABRAM_N_PATCHES, LABRAM_PATCH_LEN)


class SleepEDFLabelled(Dataset):
    def __init__(self, root: str):
        self.samples = []
        for label_name, label_idx in LABEL_MAP.items():
            d = os.path.join(root, label_name)
            if not os.path.isdir(d): continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(d, f), label_idx))
        print(f"[Dataset] TestFold: {len(self.samples)} epochs")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return torch.load(path, weights_only=False).float(), label



def load_eegpt(strategy: str, seed: int, output_dir: str,
               device: torch.device) -> torch.nn.Module:
    path = os.path.join(output_dir, f"eegpt_{strategy}_seed{seed}_encoder.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = EEGTransformer(
        img_size=[EEGPT_N_CHANNELS, EEGPT_SFREQ * EEGPT_SEG_SECONDS],
        patch_size=EEGPT_PATCH_SIZE, embed_dim=EEGPT_EMBED_DIM,
        depth=8, num_heads=8)
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False), strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model.to(device)


def load_labram(strategy: str, seed: int, output_dir: str,
                device: torch.device) -> torch.nn.Module:
    path = os.path.join(output_dir, f"labram_{strategy}_seed{seed}_backbone.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = create_model(
        "labram_base_patch200_200",
        qkv_bias=False, rel_pos_bias=True, num_classes=0,
        drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0,
        drop_block_rate=None, use_mean_pooling=True,
        init_scale=0.001, use_rel_pos_bias=True,
        use_abs_pos_emb=True, init_values=0.1,
    )
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False), strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model.to(device)



@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, loader: DataLoader,
                       model_name: str, device: torch.device):
    all_emb, all_labels = [], []
    for x, y in tqdm(loader, desc="    extracting", leave=False):
        x = x.to(device)
        if model_name == "eegpt":
            z = model(x)
            if z.dim() == 4: z = z.squeeze(2)
            z = z.mean(dim=1)                                              # (B, 512)
        else:
            x_in = to_labram_input(x)
            z    = model(x_in, input_chans=list(range(LABRAM_N_CHANNELS + 1)))  # (B, D)
            if z.dim() == 3: z = z.mean(dim=1)
        all_emb.append(z.cpu().numpy())
        all_labels.extend(y.tolist() if hasattr(y, "tolist") else list(y))
    return np.vstack(all_emb), np.array(all_labels)



def silhouette(emb, labels) -> float:
    try:
        return float(silhouette_score(emb, labels, metric="euclidean",
                                      sample_size=min(10_000, len(labels)),
                                      random_state=42))
    except Exception:
        return float("nan")


def knn_accuracy(emb, labels, k: int = 5, test_frac: float = 0.3) -> float:
    rng      = np.random.default_rng(42)
    idx      = rng.permutation(len(labels))
    n_test   = int(len(labels) * test_frac)
    scaler   = StandardScaler()
    X_tr     = scaler.fit_transform(emb[idx[n_test:]])
    X_te     = scaler.transform(emb[idx[:n_test]])
    knn      = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_tr, labels[idx[n_test:]])
    return float(knn.score(X_te, labels[idx[:n_test]]))


def within_between_ratio(emb, labels) -> float:
    classes   = np.unique(labels)
    centroids = {c: emb[labels == c].mean(0) for c in classes}
    within    = sum(((emb[labels == c] - centroids[c]) ** 2).sum() for c in classes)
    global_c  = emb.mean(0)
    between   = sum((labels == c).sum() * ((centroids[c] - global_c) ** 2).sum()
                    for c in classes)
    return float(within / between) if between > 0 else float("inf")


def collapse_metric(emb) -> float:
    return float(np.std(emb, axis=0).mean())


def separation_score(emb, labels) -> float:
    classes   = np.unique(labels)
    centroids = {c: emb[labels == c].mean(0) for c in classes}
    return float(np.mean([
        np.mean([np.linalg.norm(centroids[c] - centroids[o])
                 for o in classes if o != c])
        for c in classes
    ]))


def compute_metrics(emb, labels) -> dict:
    return {
        "silhouette":      silhouette(emb, labels),
        "knn_accuracy":    knn_accuracy(emb, labels),
        "within_between":  within_between_ratio(emb, labels),
        "collapse":        collapse_metric(emb),
        "separation":      separation_score(emb, labels),
    }


def analyze_strategy(model_name: str, strategy: str, seeds: list,
                     output_dir: str, loader: DataLoader,
                     device: torch.device) -> dict:
    load_fn  = load_eegpt if model_name == "eegpt" else load_labram
    per_seed = defaultdict(list)

    for seed in seeds:
        print(f"  seed={seed}: ", end="", flush=True)
        try:
            model = load_fn(strategy, seed, output_dir, device)
        except FileNotFoundError as e:
            print(f"SKIP — not found: {e}")
            continue

        emb, labels = extract_embeddings(model, loader, model_name, device)
        m = compute_metrics(emb, labels)
        for k, v in m.items():
            per_seed[k].append(v)
        print(f"silhouette={m['silhouette']:.4f}  "
              f"kNN={m['knn_accuracy']:.4f}  "
              f"W/B={m['within_between']:.4f}")

    if not per_seed:
        return {}

    return {
        metric: {
            "mean":    float(np.mean(vals)),
            "std":     float(np.std(vals)),
            "values":  [float(v) for v in vals],
            "n_seeds": len(vals),
        }
        for metric, vals in per_seed.items()
    }



HIGHER_IS_BETTER = {
    "silhouette":     True,
    "knn_accuracy":   True,
    "within_between": False,
    "collapse":       True,
    "separation":     True,
}

def print_pairwise(results: dict, strategies: list):
    print(f"\n{'='*70}")
    print("  PAIRWISE COMPARISON")
    print(f"{'='*70}")
    for s_a, s_b in combinations(strategies, 2):
        if s_a not in results or s_b not in results: continue
        wins_a = wins_b = 0
        print(f"\n  {s_a.upper()} vs {s_b.upper()}")
        for metric, hib in HIGHER_IS_BETTER.items():
            if metric not in results[s_a] or metric not in results[s_b]: continue
            v_a = results[s_a][metric]["mean"]
            v_b = results[s_b][metric]["mean"]
            winner = s_a if (v_a > v_b) == hib else s_b
            if winner == s_a: wins_a += 1
            else:             wins_b += 1
            arrow = "↑" if hib else "↓"
            print(f"    {metric:<18} {arrow}   {s_a}: {v_a:.4f}   {s_b}: {v_b:.4f}   → {winner}")
        total = wins_a + wins_b
        winner_str = s_a if wins_a > wins_b else (s_b if wins_b > wins_a else "tie")
        print(f"  ➜ {winner_str} wins on {max(wins_a, wins_b)}/{total} metrics")


METRICS_TABLE = [
    ("silhouette",     "Silhouette  ↑"),
    ("knn_accuracy",   "kNN Acc     ↑"),
    ("within_between", "W/B ratio   ↓"),
    ("collapse",       "Collapse    ↑"),
    ("separation",     "Separation  ↑"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised representation quality analysis")
    parser.add_argument("--model",      required=True, choices=["eegpt", "labram"])
    parser.add_argument("--strategies", nargs="+",
                        default=["none", "random", "theta", "delta", "alpha"])
    parser.add_argument("--seeds",      nargs="+", type=int, default=[42],
                        help="Seeds to average over. Default [42] = single encoder per strategy. "
                             "Pass --seeds 42 43 44 45 46 to reuse significance-test runs.")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    assert TEST_ROOT, "TEST_ROOT is not set — create a .env file (see .env.example)"

    multi  = len(args.seeds) > 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}  |  model={args.model}  |  "
          f"seeds={args.seeds}  |  strategies={args.strategies}")

    loader = DataLoader(
        SleepEDFLabelled(TEST_ROOT),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    results = {}
    for strategy in args.strategies:
        print(f"\n[{strategy.upper()}]")
        summary = analyze_strategy(args.model, strategy, args.seeds,
                                   args.output_dir, loader, device)
        if summary:
            results[strategy] = summary

    col = 20
    print(f"\n{'='*80}")
    print(f"  REPRESENTATION QUALITY — {args.model.upper()} on Sleep-EDF")
    if multi:
        print(f"  Averaged over {len(args.seeds)} seeds (mean ± std)")
    print(f"{'='*80}")
    header = f"  {'Strategy':<14}" + "".join(f"{n:>{col}}" for _, n in METRICS_TABLE)
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    for strategy in args.strategies:
        if strategy not in results: continue
        row = f"  {strategy:<14}"
        for key, _ in METRICS_TABLE:
            if key not in results[strategy]:
                row += f"{'N/A':>{col}}"
                continue
            d = results[strategy][key]
            cell = f"{d['mean']:.4f}±{d['std']:.4f}" if multi else f"{d['mean']:.4f}"
            row += f"{cell:>{col}}"
        print(row)


    if len(results) >= 2:
        print_pairwise(results, args.strategies)

    out_path = os.path.join(args.output_dir,
                            f"{args.model}_representation_quality.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()