from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.truth_direction import logit_diff, mean_difference_direction, probe_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--direction-method", choices=["probe", "mean_diff"], default="probe")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0, 0.5, 1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="figures/oracle_steering_capital_probe_layer8.csv")
    parser.add_argument("--true-token", default=" true")
    parser.add_argument("--false-token", default=" false")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def make_oracle_direction_hook(direction: torch.Tensor, alpha: float, signs: torch.Tensor):
    def hook(resid, hook):
        device_direction = direction.to(resid.device)
        device_signs = signs.to(resid.device, dtype=resid.dtype)
        resid[:, -1, :] = resid[:, -1, :] + alpha * device_signs[:, None] * device_direction
        return resid

    return hook


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain)
    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
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

    if args.direction_method == "probe":
        direction = probe_direction(train_activations, train_labels)
    else:
        direction = mean_difference_direction(train_activations, train_labels)
    direction = direction.to(model.cfg.device)

    train_probe_scores = train_activations @ direction.detach().cpu()
    probe_threshold = float(
        0.5 * (train_probe_scores[train_labels == 1].mean() + train_probe_scores[train_labels == 0].mean())
    )
    test_base_scores = test_activations @ direction.detach().cpu()
    label_signs = torch.tensor(2 * test_labels - 1, dtype=test_base_scores.dtype)
    tokens = model.to_tokens(test_prompts, prepend_bos=True)
    true_token = model.to_single_token(args.true_token)
    false_token = model.to_single_token(args.false_token)

    rows = []
    for alpha in args.alphas:
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (
                        f"blocks.{args.layer}.hook_resid_post",
                        make_oracle_direction_hook(direction, alpha, label_signs),
                    )
                ],
            )
        diffs = logit_diff(logits, true_token=true_token, false_token=false_token).detach().cpu()
        probe_scores = test_base_scores + alpha * label_signs
        probe_predicted_true = (probe_scores.numpy() > probe_threshold).astype(int)
        logit_predicted_true = (diffs.numpy() > 0).astype(int)
        logit_correct_margin = torch.where(torch.tensor(test_labels) == 1, diffs, -diffs)
        probe_correct_margin = label_signs * (probe_scores - probe_threshold)
        rows.append(
            {
                "direction_method": args.direction_method,
                "layer": args.layer,
                "seed": args.seed,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "split": "group",
                "threshold_source": "train_midpoint",
                "steering_mode": "oracle_label_conditioned",
                "alpha": alpha,
                "accuracy_from_logit_sign": float((logit_predicted_true == test_labels).mean()),
                "accuracy_from_probe_score_threshold": float((probe_predicted_true == test_labels).mean()),
                "mean_logit_correct_margin": float(logit_correct_margin.mean()),
                "mean_probe_correct_margin": float(probe_correct_margin.mean()),
                "mean_probe_score": float(probe_scores.mean()),
                "probe_score_threshold": probe_threshold,
            }
        )
        print(
            f"alpha={alpha:+.2f} logit_acc={rows[-1]['accuracy_from_logit_sign']:.3f} "
            f"probe_acc={rows[-1]['accuracy_from_probe_score_threshold']:.3f} "
            f"probe_margin={rows[-1]['mean_probe_correct_margin']:+.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved oracle steering results to {out_path}")


if __name__ == "__main__":
    main()
