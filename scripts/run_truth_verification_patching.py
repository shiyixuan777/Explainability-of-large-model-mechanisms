from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import load_hooked_transformer, make_prompts
from src.truth_direction import logit_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital")
    parser.add_argument("--out", default="figures/truth_verification_patching_resid.csv")
    parser.add_argument("--max-pairs", type=int, default=76)
    parser.add_argument("--true-token", default=" true")
    parser.add_argument("--false-token", default=" false")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def build_true_false_pairs(data: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    rows = []
    for pair_id, group in data.groupby("pair_id", sort=True):
        true_rows = group[group["label"] == 1]
        false_rows = group[group["label"] == 0]
        if true_rows.empty or false_rows.empty:
            continue
        rows.append(
            {
                "pair_id": pair_id,
                "true_statement": str(true_rows.iloc[0]["statement"]),
                "false_statement": str(false_rows.iloc[0]["statement"]),
            }
        )
        if len(rows) >= max_pairs:
            break
    if not rows:
        raise ValueError("No paired true/false examples found.")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain)
    pairs = build_true_false_pairs(data, args.max_pairs)
    print(f"Using {len(pairs)} paired true/false examples from {args.domain}.")

    model = load_hooked_transformer(args.model)
    true_token = model.to_single_token(args.true_token)
    false_token = model.to_single_token(args.false_token)

    clean_prompts = make_prompts(pairs["true_statement"].tolist(), args.prompt_template)
    corrupt_prompts = make_prompts(pairs["false_statement"].tolist(), args.prompt_template)
    clean_tokens = model.to_tokens(clean_prompts, prepend_bos=True)
    corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=True)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits = model(corrupt_tokens)

    clean_diff = logit_diff(clean_logits, true_token=true_token, false_token=false_token).detach()
    corrupt_diff = logit_diff(corrupt_logits, true_token=true_token, false_token=false_token).detach()
    denominator = clean_diff - corrupt_diff
    valid = denominator.abs() > 1e-5
    if int(valid.sum()) == 0:
        raise ValueError("All clean-corrupt truth logit differences are nearly zero.")

    rows = []
    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"
        clean_activation = clean_cache[hook_name]

        def patch_resid_post(activation, hook, clean_activation=clean_activation):
            activation[:, -1, :] = clean_activation[:, -1, :]
            return activation

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, patch_resid_post)],
            )
        patched_diff = logit_diff(patched_logits, true_token=true_token, false_token=false_token).detach()
        recovery = (patched_diff[valid] - corrupt_diff[valid]) / (denominator[valid] + 1e-8)
        rows.append(
            {
                "component": "resid_post",
                "hook_name": hook_name,
                "layer": layer,
                "n_pairs": len(pairs),
                "n_valid_recovery_pairs": int(valid.sum()),
                "clean_true_minus_false_logit_diff": float(clean_diff.mean()),
                "corrupt_true_minus_false_logit_diff": float(corrupt_diff.mean()),
                "patched_true_minus_false_logit_diff": float(patched_diff.mean()),
                "mean_recovery": float(recovery.mean()),
                "median_recovery": float(recovery.median()),
                "mean_abs_logit_shift": float((patched_diff - corrupt_diff).abs().mean()),
            }
        )
        print(
            f"layer={layer:02d} patched_diff={patched_diff.mean():+.4f} "
            f"mean_recovery={recovery.mean():+.4f} abs_shift={(patched_diff - corrupt_diff).abs().mean():.4f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved truth verification patching results to {out_path}")


if __name__ == "__main__":
    main()
