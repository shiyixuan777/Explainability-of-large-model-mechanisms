from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.truth_direction import evaluate_layer_probe


PROMPT_TEMPLATES = {
    "statement_is": "Statement: {statement}\nThe statement is",
    "answer": "Statement: {statement}\nAnswer true or false:",
    "question": "Question: Is the following statement true or false?\n{statement}\nAnswer:",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--out", default="figures/probe_sweep.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--domains",
        default="all,capital,continent,element_symbol,book_author,landmark_country,science,math",
    )
    parser.add_argument("--prompts", default="statement_is,answer,question")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_data = load_dataset(args.data)
    model = load_hooked_transformer(args.model)

    rows = []
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    prompts = [item.strip() for item in args.prompts.split(",") if item.strip()]

    for domain_name in domains:
        domain_filter = None if domain_name == "all" else domain_name
        data = filter_dataset(base_data, language=args.language, domain=domain_filter)
        labels = data["label"].to_numpy()
        groups = data["pair_id"].to_numpy()

        for prompt_name in prompts:
            template = PROMPT_TEMPLATES[prompt_name]
            prompt_texts = make_prompts(data["statement"].tolist(), template)
            activations_by_layer = collect_resid_post_by_layer(model, prompt_texts)

            best_row = None
            for layer, activations in activations_by_layer.items():
                metrics = evaluate_layer_probe(activations, labels, groups=groups, seed=args.seed)
                row = {
                    "domain": domain_name,
                    "prompt": prompt_name,
                    "n_rows": len(data),
                    "layer": layer,
                    **metrics,
                }
                rows.append(row)
                if best_row is None or row["separability_auc"] > best_row["separability_auc"]:
                    best_row = row

            print(
                f"domain={domain_name:16s} prompt={prompt_name:12s} "
                f"best_layer={best_row['layer']:02d} "
                f"acc={best_row['accuracy']:.3f} auc={best_row['auc']:.3f} "
                f"sep_auc={best_row['separability_auc']:.3f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved probe sweep results to {out_path}")


if __name__ == "__main__":
    main()
