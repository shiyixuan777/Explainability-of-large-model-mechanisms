from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.truth_direction import evaluate_layer_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital")
    parser.add_argument("--layers", nargs="+", type=int, default=[5, 8, 10, 11])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 42])
    parser.add_argument("--out", default="figures/probe_seed_sensitivity_capital.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain)
    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations_by_layer = collect_resid_post_by_layer(model, prompts)

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for layer in args.layers:
            metrics = evaluate_layer_probe(activations_by_layer[layer], labels, groups=groups, seed=seed)
            rows.append(
                {
                    "model": args.model,
                    "domain": args.domain,
                    "layer": layer,
                    "seed": seed,
                    "n_rows": len(data),
                    "split": "group",
                    **metrics,
                }
            )
            print(
                f"seed={seed:02d} layer={layer:02d} accuracy={metrics['accuracy']:.3f} "
                f"auc={metrics['auc']:.3f} sep_auc={metrics['separability_auc']:.3f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved probe seed sensitivity results to {out_path}")


if __name__ == "__main__":
    main()
