from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="figures/probe_layers.csv")
    parser.add_argument("--steering", default="figures/steering_alpha.csv")
    parser.add_argument("--out-dir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    probe_path = Path(args.probe)
    if probe_path.exists():
        probe = pd.read_csv(probe_path)
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=probe, x="layer", y="accuracy", marker="o", label="accuracy")
        sns.lineplot(data=probe, x="layer", y="auc", marker="o", label="AUC")
        plt.ylim(0, 1.05)
        plt.title("Truth/False Linear Probe by Layer")
        plt.tight_layout()
        plt.savefig(out_dir / "probe_layers.png", dpi=200)

    steering_path = Path(args.steering)
    if steering_path.exists():
        steering = pd.read_csv(steering_path)
        plt.figure(figsize=(7, 4))
        sns.lineplot(
            data=steering,
            x="alpha",
            y="mean_logit_diff_true_minus_false",
            marker="o",
        )
        plt.axhline(0, color="black", linewidth=1)
        plt.title("Steering Strength vs True-False Logit Difference")
        plt.tight_layout()
        plt.savefig(out_dir / "steering_alpha.png", dpi=200)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
