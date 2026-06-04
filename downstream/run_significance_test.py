"""
run_significance_test.py
Multi-seed significance testing for Sleep-EDF masking strategies.

Usage:
    python run_significance_test.py --model eegpt --strategies theta random none --n_seeds 5
    python run_significance_test.py --model labram --strategies theta delta alpha --n_seeds 3
    python run_significance_test.py --model eegpt --strategies theta random --skip_training
"""

import argparse, json, logging, os, subprocess, sys
from itertools import combinations

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── file-naming convention (must match the pretrain/eval scripts) ─
def weights_path(model: str, strategy: str, seed: int, output_dir: str) -> str:
    suffix = "encoder" if model == "eegpt" else "backbone"
    return os.path.join(output_dir, f"{model}_{strategy}_seed{seed}_{suffix}.pt")

def eval_json_path(model: str, strategy: str, seed: int, output_dir: str) -> str:
    return os.path.join(output_dir, f"{model}_{strategy}_seed{seed}_eval.json")

PRETRAIN_SCRIPT = {
    "eegpt":  "pretrain_EEGPT_SleepEDF.py",
    "labram": "pretrain_LaBraM_SleepEDF.py",
}
EVAL_SCRIPT = {
    "eegpt":  "eval_EEGPT_SleepEDF.py",
    "labram": "eval_LaBraM_SleepEDF.py",
}
WEIGHTS_ARG = {
    "eegpt":  "--encoder_path",
    "labram": "--backbone_path",
}


# ── subprocess helpers ────────────────────────────────────────────
def run_pretrain(model: str, strategy: str, seed: int, epochs: int,
                 output_dir: str) -> bool:
    cmd = [
        sys.executable, PRETRAIN_SCRIPT[model],
        "--strategy",   strategy,
        "--seed",       str(seed),
        "--epochs",     str(epochs),
        "--output_dir", output_dir,
    ]
    log.info(f"Pretrain  {model.upper()} | strategy={strategy} | seed={seed}")
    log.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"Pretraining FAILED (strategy={strategy}, seed={seed})")
        return False
    if not os.path.isfile(weights_path(model, strategy, seed, output_dir)):
        log.error(f"Weights file not created after pretrain")
        return False
    return True


def run_eval(model: str, strategy: str, seed: int, output_dir: str,
             eval_mode: str) -> bool:
    w_path = weights_path(model, strategy, seed, output_dir)
    cmd = [
        sys.executable, EVAL_SCRIPT[model],
        WEIGHTS_ARG[model], w_path,
        "--eval_mode",  eval_mode,
        "--seed",       str(seed),
    ]
    log.info(f"Eval      {model.upper()} | strategy={strategy} | seed={seed}")
    log.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"Evaluation FAILED (strategy={strategy}, seed={seed})")
        return False
    if not os.path.isfile(eval_json_path(model, strategy, seed, output_dir)):
        log.error(f"Eval JSON not created — check on_test_epoch_end in the eval script")
        return False
    return True


# ── statistics ────────────────────────────────────────────────────
def pairwise_test(scores_a: list, scores_b: list, label_a: str, label_b: str,
                  metric: str) -> dict:
    n = min(len(scores_a), len(scores_b))
    a, b = np.array(scores_a[:n]), np.array(scores_b[:n])
    t_stat, p_t = stats.ttest_rel(a, b)
    try:
        w_stat, p_w = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, p_w = float("nan"), float("nan")   # all differences are zero

    return {
        "metric":   metric,
        "pair":     f"{label_a} vs {label_b}",
        "mean_a":   float(a.mean()), "std_a": float(a.std()),
        "mean_b":   float(b.mean()), "std_b": float(b.std()),
        "diff":     float(b.mean() - a.mean()),
        "t_stat":   float(t_stat),  "p_ttest":    float(p_t),
        "w_stat":   float(w_stat),  "p_wilcoxon": float(p_w),
        "n":        n,
    }


def sig_marker(p: float) -> str:
    if p < 0.01: return " **"
    if p < 0.05: return " *"
    return ""


def print_comparison(res: dict):
    print(f"\n  {res['metric'].upper()}: "
          f"{res['pair']}")
    print(f"    {res['pair'].split(' vs ')[0]:12s}: "
          f"{res['mean_a']:.4f} ± {res['std_a']:.4f}")
    print(f"    {res['pair'].split(' vs ')[1]:12s}: "
          f"{res['mean_b']:.4f} ± {res['std_b']:.4f}  "
          f"(Δ={res['diff']:+.4f})")
    print(f"    t-test:   t={res['t_stat']:+.3f}  p={res['p_ttest']:.4f}"
          f"{sig_marker(res['p_ttest'])}")
    print(f"    Wilcoxon: W={res['w_stat']:.1f}   p={res['p_wilcoxon']:.4f}"
          f"{sig_marker(res['p_wilcoxon'])}")


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-seed significance test")
    parser.add_argument("--model",      required=True, choices=["eegpt", "labram"])
    parser.add_argument("--strategies", required=True, nargs="+",
                        help="Strategies to compare, e.g. theta random none")
    parser.add_argument("--n_seeds",    type=int, default=5,
                        help="Number of seeds to run per strategy")
    parser.add_argument("--start_seed", type=int, default=42,
                        help="First seed value (subsequent seeds: start+1, start+2, …)")
    parser.add_argument("--pretrain_epochs", type=int, default=30,
                        help="Epochs for pretraining (30 is faster; use 50 for final results)")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory for pretrained weights and eval JSONs")
    parser.add_argument("--results_dir", default="significance_results",
                        help="Directory for the final statistical summary")
    parser.add_argument("--eval_mode",  default="linear_probe",
                        choices=["linear_probe", "finetune"])
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip pretrain; only run eval on existing weights")
    parser.add_argument("--retrain_strategies", nargs="*", default=[],
                        help="Force retrain these strategies even if weights exist, e.g. --retrain_strategies theta random")
    args = parser.parse_args()

    os.makedirs(args.output_dir,  exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    seeds      = list(range(args.start_seed, args.start_seed + args.n_seeds))
    total_runs = len(args.strategies) * len(seeds)

    log.info(f"Model:      {args.model.upper()}")
    log.info(f"Strategies: {args.strategies}")
    log.info(f"Seeds:      {seeds}")
    log.info(f"Total runs: {total_runs}")

    # ── collect results ──────────────────────────────────────────
    all_results = {s: [] for s in args.strategies}
    run_idx = 0

    for seed in seeds:
        for strategy in args.strategies:
            run_idx += 1
            log.info(f"\n{'='*60}")
            log.info(f"Run {run_idx}/{total_runs}: {strategy} | seed={seed}")
            log.info(f"{'='*60}")

            w_path = weights_path(args.model, strategy, seed, args.output_dir)
            j_path = eval_json_path(args.model, strategy, seed, args.output_dir)

            # ── pretrain if needed ────────────────────────────────
            force_retrain = strategy in (args.retrain_strategies or [])
            if force_retrain and os.path.isfile(w_path):
                log.info(f"Forcing retrain for {strategy} (--retrain_strategies)")
            need_pretrain = not os.path.isfile(w_path) or force_retrain
            if not args.skip_training and need_pretrain:
                ok = run_pretrain(args.model, strategy, seed,
                                  args.pretrain_epochs, args.output_dir)
                if not ok:
                    log.error(f"Skipping eval for {strategy} seed={seed}")
                    continue
            elif not os.path.isfile(w_path):
                log.error(f"Weights not found and --skip_training is set: {w_path}")
                continue
            else:
                log.info(f"Weights already exist — skipping pretrain")

            # ── eval if needed ────────────────────────────────────
            if not os.path.isfile(j_path):
                ok = run_eval(args.model, strategy, seed, args.output_dir, args.eval_mode)
                if not ok:
                    log.error(f"Skipping {strategy} seed={seed}")
                    continue
            else:
                log.info(f"Eval JSON already exists — skipping eval")

            # ── load results ──────────────────────────────────────
            with open(j_path) as fp:
                metrics = json.load(fp)
            metrics["seed"] = seed
            all_results[strategy].append(metrics)
            log.info(f"Acc={metrics['accuracy']:.4f}  "
                     f"F1={metrics['f1_macro']:.4f}  "
                     f"Kappa={metrics['kappa']:.4f}")

    # ── summary table ─────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  RESULTS SUMMARY — {args.model.upper()} on Sleep-EDF")
    print(f"{'='*70}")
    print(f"  {'Strategy':<14} {'N':>3}  {'Acc':>10}  {'F1-macro':>10}  {'Kappa':>10}")
    print(f"  {'-'*60}")
    for strategy, results in all_results.items():
        if not results:
            print(f"  {strategy:<14}   0  (no results)")
            continue
        accs   = [r["accuracy"] for r in results]
        f1s    = [r["f1_macro"] for r in results]
        kappas = [r["kappa"]    for r in results]
        print(f"  {strategy:<14} {len(results):>3}  "
              f"{np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"{np.mean(kappas):.4f}±{np.std(kappas):.4f}")

    # ── pairwise statistical tests ────────────────────────────────
    all_comparisons = []
    print(f"\n{'='*70}")
    print(f"  PAIRWISE TESTS  (* p<0.05  ** p<0.01)")
    print(f"{'='*70}")

    for strat_a, strat_b in combinations(args.strategies, 2):
        res_a = all_results.get(strat_a, [])
        res_b = all_results.get(strat_b, [])
        if not res_a or not res_b:
            print(f"\n  {strat_a} vs {strat_b}: insufficient data — skipping")
            continue

        print(f"\n  {strat_a.upper()} vs {strat_b.upper()}")
        for metric_key, metric_label in [("accuracy", "accuracy"),
                                          ("f1_macro", "f1_macro"),
                                          ("kappa",    "kappa")]:
            scores_a = [r[metric_key] for r in res_a if metric_key in r]
            scores_b = [r[metric_key] for r in res_b if metric_key in r]
            res = pairwise_test(scores_a, scores_b, strat_a, strat_b, metric_label)
            print_comparison(res)
            all_comparisons.append(res)

    # ── save full results ─────────────────────────────────────────
    summary = {
        "model":      args.model,
        "strategies": args.strategies,
        "seeds":      seeds,
        "eval_mode":  args.eval_mode,
        "raw_results":    all_results,
        "comparisons":    all_comparisons,
    }
    out_json = os.path.join(args.results_dir,
                            f"{args.model}_significance.json")
    with open(out_json, "w") as fp:
        json.dump(summary, fp, indent=2)

    log.info(f"\nFull results saved → {out_json}")
    print("\n  Legend: * p<0.05  ** p<0.01  (both t-test and Wilcoxon must hold)")


if __name__ == "__main__":
    main()