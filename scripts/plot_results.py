from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def output_png_name(csv_path: Path, suffix: str = "") -> str:
    return f"{csv_path.stem}{suffix}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="figures/probe_layers.csv")
    parser.add_argument("--probe-sweep", default="figures/probe_sweep.csv")
    parser.add_argument("--steering", default="figures/steering_alpha.csv")
    parser.add_argument("--patching", default="figures/activation_patching_capital_recall.csv")
    parser.add_argument("--ablation", default="figures/ablation_capital_probe_layer8.csv")
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
        plt.savefig(out_dir / output_png_name(probe_path), dpi=200)

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
        plt.savefig(out_dir / output_png_name(steering_path), dpi=200)

        if "accuracy_from_logit_sign" in steering.columns:
            plt.figure(figsize=(7, 4))
            sns.lineplot(
                data=steering,
                x="alpha",
                y="accuracy_from_logit_sign",
                marker="o",
            )
            plt.ylim(0, 1.05)
            plt.title("Steering Strength vs Logit-Sign Accuracy")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(steering_path, "_accuracy"), dpi=200)

        probe_accuracy_column = None
        if "accuracy_from_probe_score_threshold" in steering.columns:
            probe_accuracy_column = "accuracy_from_probe_score_threshold"
        elif "accuracy_from_probe_score_sign" in steering.columns:
            probe_accuracy_column = "accuracy_from_probe_score_sign"

        if probe_accuracy_column:
            plt.figure(figsize=(7, 4))
            sns.lineplot(
                data=steering,
                x="alpha",
                y=probe_accuracy_column,
                marker="o",
            )
            plt.ylim(0, 1.05)
            plt.title("Steering Strength vs Probe-Score Accuracy")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(steering_path, "_probe_accuracy"), dpi=200)

    sweep_path = Path(args.probe_sweep)
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        best = (
            sweep.sort_values("separability_auc", ascending=False)
            .groupby(["domain", "prompt"], as_index=False)
            .first()
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=best, x="domain", y="separability_auc", hue="prompt")
        plt.axhline(0.5, color="black", linewidth=1)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.title("Best Probe Separability by Domain and Prompt")
        plt.tight_layout()
        plt.savefig(out_dir / "probe_sweep_summary.png", dpi=200)

    patching_path = Path(args.patching)
    if patching_path.exists():
        patching = pd.read_csv(patching_path)
        plt.figure(figsize=(7, 4))
        if "component" in patching.columns:
            sns.lineplot(data=patching, x="layer", y="mean_recovery", hue="component", marker="o")
        else:
            sns.lineplot(data=patching, x="layer", y="mean_recovery", marker="o")
        plt.axhline(0, color="black", linewidth=1)
        plt.axhline(1, color="black", linewidth=1, linestyle="--")
        plt.title("Activation Patching Recovery by Layer")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(patching_path), dpi=200)

    ablation_path = Path(args.ablation)
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=ablation, x="strength", y="separability_auc", marker="o", label="after ablation")
        if "fixed_direction_separability_auc" in ablation.columns:
            sns.lineplot(
                data=ablation,
                x="strength",
                y="fixed_direction_separability_auc",
                marker="o",
                label="fixed direction",
            )
        if "baseline_separability_auc" in ablation.columns:
            baseline = float(ablation["baseline_separability_auc"].iloc[0])
            plt.axhline(baseline, color="black", linewidth=1, linestyle="--", label="baseline")
        plt.ylim(0, 1.05)
        plt.title("Probe Separability After Direction Ablation")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(ablation_path), dpi=200)

        if "fixed_direction_score_gap" in ablation.columns:
            plt.figure(figsize=(7, 4))
            sns.lineplot(data=ablation, x="strength", y="fixed_direction_score_gap", marker="o")
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Fixed Probe-Direction Score Gap After Ablation")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(ablation_path, "_score_gap"), dpi=200)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
