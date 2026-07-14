from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital", help="Comma-separated domain filter, or 'all'")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--out", default="figures/pca_capital_layer8.csv")
    parser.add_argument(
        "--plot-trim-quantile",
        type=float,
        default=0.01,
        help="Trim this tail quantile from PC1/PC2 for the saved figure only; CSV keeps all rows.",
    )
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain = None if args.domain == "all" else args.domain
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=domain)
    if data.empty:
        raise ValueError("No rows matched the requested filters.")

    print(f"Using {len(data)} rows from {args.data}")
    print(data["domain"].value_counts().sort_index().to_string())

    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    model = load_hooked_transformer(args.model)
    activations_by_layer = collect_resid_post_by_layer(model, prompts)

    if args.layer not in activations_by_layer:
        raise ValueError(f"Layer {args.layer} is unavailable for {args.model}.")

    activations = activations_by_layer[args.layer].numpy()
    activations = StandardScaler().fit_transform(activations)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(activations)
    explained = pca.explained_variance_ratio_

    result = data[["statement", "label", "domain", "pair_id"]].copy()
    result["label_name"] = result["label"].map({0: "false", 1: "true"})
    result["pc1"] = coords[:, 0]
    result["pc2"] = coords[:, 1]
    result["layer"] = args.layer
    result["pc1_explained_variance"] = float(explained[0])
    result["pc2_explained_variance"] = float(explained[1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    plot_data = result
    if args.plot_trim_quantile > 0:
        q = args.plot_trim_quantile
        pc1_low, pc1_high = result["pc1"].quantile([q, 1 - q])
        pc2_low, pc2_high = result["pc2"].quantile([q, 1 - q])
        plot_data = result[
            result["pc1"].between(pc1_low, pc1_high)
            & result["pc2"].between(pc2_low, pc2_high)
        ]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=plot_data,
        x="pc1",
        y="pc2",
        hue="label_name",
        style="domain" if plot_data["domain"].nunique() > 1 else None,
        alpha=0.78,
        s=45,
    )
    plt.title(
        f"Layer {args.layer} Residual Activation PCA "
        f"(PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%})"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    png_path = out_path.with_suffix(".png")
    plt.savefig(png_path, dpi=200)

    print(
        f"Saved PCA coordinates to {out_path}; figure to {png_path}; "
        f"explained_variance=({explained[0]:.4f}, {explained[1]:.4f}); "
        f"plotted_rows={len(plot_data)}/{len(result)}"
    )


if __name__ == "__main__":
    main()
