from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.dataset import load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.steering import run_steered_logits
from src.truth_direction import logit_diff, mean_difference_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--out", default="figures/steering_alpha.csv")
    parser.add_argument("--alphas", nargs="+", type=float, default=[-4, -2, -1, 0, 1, 2, 4])
    parser.add_argument("--true-token", default=" true")
    parser.add_argument("--false-token", default=" false")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data)
    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]
    direction = mean_difference_direction(activations, labels).to(model.cfg.device)

    tokens = model.to_tokens(prompts, prepend_bos=True)
    true_token = model.to_single_token(args.true_token)
    false_token = model.to_single_token(args.false_token)

    rows = []
    for alpha in args.alphas:
        with torch.no_grad():
            logits = run_steered_logits(model, tokens, args.layer, direction, alpha)
        diffs = logit_diff(logits, true_token=true_token, false_token=false_token).detach().cpu()
        rows.append(
            {
                "alpha": alpha,
                "mean_logit_diff_true_minus_false": float(diffs.mean()),
                "std_logit_diff": float(diffs.std()),
            }
        )
        print(f"alpha={alpha:+.2f} mean_logit_diff={diffs.mean():+.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved steering results to {out_path}")


if __name__ == "__main__":
    main()
