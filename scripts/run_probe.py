from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset import load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.truth_direction import evaluate_layer_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--out", default="figures/probe_layers.csv")
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
    activations_by_layer = collect_resid_post_by_layer(model, prompts)

    rows = []
    for layer, activations in activations_by_layer.items():
        metrics = evaluate_layer_probe(activations, labels)
        rows.append({"layer": layer, **metrics})
        print(f"layer={layer:02d} accuracy={metrics['accuracy']:.3f} auc={metrics['auc']:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved probe results to {out_path}")


if __name__ == "__main__":
    main()
