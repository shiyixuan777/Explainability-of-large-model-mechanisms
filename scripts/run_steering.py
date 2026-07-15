from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.steering import run_steered_logits
from src.truth_direction import logit_diff, mean_difference_direction, probe_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--out", default="figures/steering_results.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default=None, help="Comma-separated domain filter, e.g. capital,science")
    parser.add_argument("--direction-method", choices=["mean_diff", "probe"], default="mean_diff")
    parser.add_argument("--alphas", nargs="+", type=float, default=[-4, -2, -1, 0, 1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true-token", default=" true")
    parser.add_argument("--false-token", default=" false")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain)
    print(f"Using {len(data)} rows from {args.data}")
    print(data["domain"].value_counts().sort_index().to_string())
    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy()
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(activations.numpy(), labels, groups=groups))
    train_activations = activations[train_idx]
    test_activations = activations[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    test_prompts = [prompts[i] for i in test_idx]

    if args.direction_method == "mean_diff":
        direction = mean_difference_direction(train_activations, train_labels)
    else:
        direction = probe_direction(train_activations, train_labels)
    direction = direction.to(model.cfg.device)

    tokens = model.to_tokens(test_prompts, prepend_bos=True)
    true_token = model.to_single_token(args.true_token)
    false_token = model.to_single_token(args.false_token)

    rows = []
    train_probe_scores = train_activations @ direction.detach().cpu()
    probe_threshold = float(
        0.5 * (train_probe_scores[train_labels == 1].mean() + train_probe_scores[train_labels == 0].mean())
    )
    for alpha in args.alphas:
        with torch.no_grad():
            logits = run_steered_logits(model, tokens, args.layer, direction, alpha)
        diffs = logit_diff(logits, true_token=true_token, false_token=false_token).detach().cpu()
        # Adding alpha * direction to the final-token residual changes projection by alpha
        # because direction is normalized.
        probe_scores = (test_activations @ direction.detach().cpu()) + alpha
        predicted_true = (diffs.numpy() > 0).astype(int)
        accuracy = float((predicted_true == test_labels).mean())
        probe_predicted_true = (probe_scores.numpy() > probe_threshold).astype(int)
        probe_accuracy = float((probe_predicted_true == test_labels).mean())
        true_mean = float(diffs[test_labels == 1].mean())
        false_mean = float(diffs[test_labels == 0].mean())
        probe_true_mean = float(probe_scores[test_labels == 1].mean())
        probe_false_mean = float(probe_scores[test_labels == 0].mean())
        rows.append(
            {
                "direction_method": args.direction_method,
                "layer": args.layer,
                "seed": args.seed,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "split": "group",
                "threshold_source": "train_midpoint",
                "alpha": alpha,
                "mean_logit_diff_true_minus_false": float(diffs.mean()),
                "mean_logit_diff_on_true_examples": true_mean,
                "mean_logit_diff_on_false_examples": false_mean,
                "std_logit_diff": float(diffs.std()),
                "accuracy_from_logit_sign": accuracy,
                "mean_probe_score": float(probe_scores.mean()),
                "mean_probe_score_on_true_examples": probe_true_mean,
                "mean_probe_score_on_false_examples": probe_false_mean,
                "probe_score_threshold": probe_threshold,
                "accuracy_from_probe_score_threshold": probe_accuracy,
            }
        )
        print(
            f"alpha={alpha:+.2f} mean_logit_diff={diffs.mean():+.4f} "
            f"true_mean={true_mean:+.4f} false_mean={false_mean:+.4f} "
            f"sign_acc={accuracy:.3f} probe_acc={probe_accuracy:.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved steering results to {out_path}")


if __name__ == "__main__":
    main()
