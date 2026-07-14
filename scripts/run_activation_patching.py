from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.dataset import CAPITALS
from src.model_hooks import load_hooked_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--out", default="figures/activation_patching_capital_recall.csv")
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--offset", type=int, default=17)
    parser.add_argument("--prompt-template", default="The capital of {country} is")
    parser.add_argument(
        "--components",
        default="resid_post,attn_out,mlp_out",
        help="Comma-separated components to patch: resid_post,attn_out,mlp_out",
    )
    return parser.parse_args()


def single_token_id(model, text: str) -> int | None:
    try:
        return int(model.to_single_token(text))
    except Exception:
        return None


def build_capital_recall_pairs(model, max_pairs: int, offset: int, prompt_template: str):
    examples = []
    n = len(CAPITALS)
    for i, (clean_country, clean_capital) in enumerate(CAPITALS):
        corrupt_country, corrupt_capital = CAPITALS[(i + offset) % n]
        clean_token = single_token_id(model, f" {clean_capital}")
        corrupt_token = single_token_id(model, f" {corrupt_capital}")
        if clean_token is None or corrupt_token is None:
            continue
        examples.append(
            {
                "clean_country": clean_country,
                "clean_capital": clean_capital,
                "corrupt_country": corrupt_country,
                "corrupt_capital": corrupt_capital,
                "clean_prompt": prompt_template.format(country=clean_country),
                "corrupt_prompt": prompt_template.format(country=corrupt_country),
                "clean_token": clean_token,
                "corrupt_token": corrupt_token,
            }
        )
        if len(examples) >= max_pairs:
            break
    if not examples:
        raise ValueError("No capital pairs with single-token capital names were found.")
    return examples


def capital_logit_diff(logits: torch.Tensor, clean_tokens: list[int], corrupt_tokens: list[int]) -> torch.Tensor:
    final_logits = logits[:, -1, :]
    clean_index = torch.tensor(clean_tokens, device=final_logits.device)
    corrupt_index = torch.tensor(corrupt_tokens, device=final_logits.device)
    rows = torch.arange(final_logits.shape[0], device=final_logits.device)
    return final_logits[rows, clean_index] - final_logits[rows, corrupt_index]


COMPONENT_TO_HOOK = {
    "resid_post": "blocks.{layer}.hook_resid_post",
    "attn_out": "blocks.{layer}.hook_attn_out",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
}


def main() -> None:
    args = parse_args()
    model = load_hooked_transformer(args.model)
    examples = build_capital_recall_pairs(model, args.max_pairs, args.offset, args.prompt_template)
    components = [component.strip() for component in args.components.split(",") if component.strip()]
    unknown = sorted(set(components).difference(COMPONENT_TO_HOOK))
    if unknown:
        raise ValueError(f"Unknown components: {unknown}. Valid: {sorted(COMPONENT_TO_HOOK)}")

    clean_prompts = [example["clean_prompt"] for example in examples]
    corrupt_prompts = [example["corrupt_prompt"] for example in examples]
    clean_answer_tokens = [example["clean_token"] for example in examples]
    corrupt_answer_tokens = [example["corrupt_token"] for example in examples]

    clean_tokens = model.to_tokens(clean_prompts, prepend_bos=True)
    corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=True)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits = model(corrupt_tokens)

    clean_diff = capital_logit_diff(clean_logits, clean_answer_tokens, corrupt_answer_tokens).detach()
    corrupt_diff = capital_logit_diff(corrupt_logits, clean_answer_tokens, corrupt_answer_tokens).detach()
    denominator = clean_diff - corrupt_diff

    rows = []
    for component in components:
        for layer in range(model.cfg.n_layers):
            hook_name = COMPONENT_TO_HOOK[component].format(layer=layer)
            clean_activation = clean_cache[hook_name]

            def patch_activation(activation, hook, clean_activation=clean_activation):
                activation[:, -1, :] = clean_activation[:, -1, :]
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupt_tokens,
                    fwd_hooks=[(hook_name, patch_activation)],
                )
            patched_diff = capital_logit_diff(
                patched_logits, clean_answer_tokens, corrupt_answer_tokens
            ).detach()
            recovery = (patched_diff - corrupt_diff) / (denominator + 1e-8)
            rows.append(
                {
                    "component": component,
                    "hook_name": hook_name,
                    "layer": layer,
                    "n_pairs": len(examples),
                    "clean_logit_diff": float(clean_diff.mean()),
                    "corrupt_logit_diff": float(corrupt_diff.mean()),
                    "patched_logit_diff": float(patched_diff.mean()),
                    "mean_recovery": float(recovery.mean()),
                    "median_recovery": float(recovery.median()),
                }
            )
            print(
                f"component={component:10s} layer={layer:02d} "
                f"patched_diff={patched_diff.mean():+.3f} mean_recovery={recovery.mean():+.3f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved activation patching results to {out_path}")
    print(f"Used {len(examples)} single-token capital pairs.")


if __name__ == "__main__":
    main()
