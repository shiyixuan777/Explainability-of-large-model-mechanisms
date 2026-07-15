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
    parser.add_argument("--probe", default="figures/probe_capital_answer.csv")
    parser.add_argument("--probe-sweep", default="figures/probe_sweep.csv")
    parser.add_argument("--probe-seeds", default="figures/probe_seed_sensitivity_capital.csv")
    parser.add_argument("--readout", default="figures/output_readout_baselines.csv")
    parser.add_argument("--surface", default="figures/surface_baselines.csv")
    parser.add_argument("--domain-transfer", default="figures/domain_transfer_layer8.csv")
    parser.add_argument("--domain-cosine", default="figures/domain_direction_cosine_layer8.csv")
    parser.add_argument("--steering", default="figures/steering_capital_probe_layer8.csv")
    parser.add_argument("--oracle-steering", default="figures/oracle_steering_capital_probe_layer8.csv")
    parser.add_argument("--patching", default="figures/activation_patching_capital_recall.csv")
    parser.add_argument("--truth-patching", default="figures/truth_verification_patching_resid.csv")
    parser.add_argument("--ablation", default="figures/ablation_capital_probe_layer8.csv")
    parser.add_argument("--iterative-ablation", default="figures/iterative_ablation_capital_layer8.csv")
    parser.add_argument("--knowledge-summary", default="figures/capital_knowledge_margin_summary.csv")
    parser.add_argument("--knowledge-details", default="figures/capital_knowledge_margin_details.csv")
    parser.add_argument("--completion-steering", default="figures/completion_margin_steering_summary.csv")
    parser.add_argument("--completion-steering-decomposition", default="figures/completion_margin_steering_decomposition.csv")
    parser.add_argument("--completion-steering-paired", default="figures/completion_margin_steering_paired_bootstrap.csv")
    parser.add_argument("--unembedding-projection-summary", default="figures/unembedding_projection_baseline_summary.csv")
    parser.add_argument("--unembedding-projection-details", default="figures/unembedding_projection_baseline_details.csv")
    parser.add_argument("--completion-steering-position-all", default="figures/completion_margin_steering_summary.csv")
    parser.add_argument(
        "--completion-steering-position-prompt-final",
        default="figures/completion_margin_steering_position_prompt_final_summary.csv",
    )
    parser.add_argument(
        "--completion-steering-position-completion-internal",
        default="figures/completion_margin_steering_position_completion_internal_summary.csv",
    )
    parser.add_argument("--completion-steering-null", default="figures/completion_margin_steering_null_distribution.csv")
    parser.add_argument("--repeated-split-steering", default="figures/repeated_split_completion_steering_summary.csv")
    parser.add_argument("--ambiguous-sensitivity", default="figures/ambiguous_fact_sensitivity_summary.csv")
    parser.add_argument("--candidate-rank-steering", default="figures/candidate_rank_steering_summary.csv")
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
        plt.close()

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
        plt.close()

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
            plt.close()

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
            plt.title("Steering Strength vs Held-Out Probe-Score Accuracy")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(steering_path, "_probe_accuracy"), dpi=200)
            plt.close()

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
        plt.close()

    probe_seeds_path = Path(args.probe_seeds)
    if probe_seeds_path.exists():
        probe_seeds = pd.read_csv(probe_seeds_path)
        plt.figure(figsize=(7, 4))
        sns.pointplot(data=probe_seeds, x="layer", y="auc", errorbar="sd")
        plt.ylim(0, 1.05)
        plt.title("Capital Probe AUC Across Group-Split Seeds")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(probe_seeds_path), dpi=200)
        plt.close()

    readout_path = Path(args.readout)
    if readout_path.exists():
        readout = pd.read_csv(readout_path)
        readout = readout[readout["single_token_readout"] == True].copy()
        if not readout.empty:
            capital_readout = readout[readout["domain"] == "capital"]
            if not capital_readout.empty:
                plt.figure(figsize=(10, 4))
                sns.barplot(
                    data=capital_readout,
                    x="verbalizer",
                    y="accuracy_from_logit_sign",
                    hue="shots",
                )
                plt.axhline(0.5, color="black", linewidth=1)
                plt.ylim(0, 1.05)
                plt.xticks(rotation=20, ha="right")
                plt.title("Output Readout Accuracy on Capital Facts")
                plt.tight_layout()
                plt.savefig(out_dir / output_png_name(readout_path), dpi=200)
                plt.close()

            best_readout = (
                readout.sort_values("accuracy_from_logit_sign", ascending=False)
                .groupby(["domain", "verbalizer"], as_index=False)
                .first()
            )
            plt.figure(figsize=(10, 4))
            sns.barplot(
                data=best_readout,
                x="verbalizer",
                y="accuracy_from_logit_sign",
                hue="domain",
            )
            plt.axhline(0.5, color="black", linewidth=1)
            plt.ylim(0, 1.05)
            plt.xticks(rotation=20, ha="right")
            plt.title("Best Output Readout Accuracy by Domain")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(readout_path, "_best_by_domain"), dpi=200)
            plt.close()

    surface_path = Path(args.surface)
    if surface_path.exists():
        surface = pd.read_csv(surface_path)
        plt.figure(figsize=(7, 4))
        sns.barplot(data=surface, x="domain", y="separability_auc", hue="baseline")
        plt.axhline(0.5, color="black", linewidth=1)
        plt.ylim(0, 1.05)
        plt.title("Surface Baseline Separability")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(surface_path), dpi=200)
        plt.close()

    transfer_path = Path(args.domain_transfer)
    if transfer_path.exists():
        transfer = pd.read_csv(transfer_path)
        matrix = transfer.pivot(index="source_domain", columns="target_domain", values="auc")
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, vmin=0, vmax=1, center=0.5, cmap="vlag", annot=True, fmt=".2f")
        plt.title("Cross-Domain Probe Transfer AUC")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(transfer_path), dpi=200)
        plt.close()

        sep_matrix = transfer.pivot(index="source_domain", columns="target_domain", values="separability_auc")
        plt.figure(figsize=(8, 6))
        sns.heatmap(sep_matrix, vmin=0.5, vmax=1, cmap="viridis", annot=True, fmt=".2f")
        plt.title("Cross-Domain Probe Transfer Separability")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(transfer_path, "_separability"), dpi=200)
        plt.close()

    cosine_path = Path(args.domain_cosine)
    if cosine_path.exists():
        cosine = pd.read_csv(cosine_path)
        cosine_matrix = cosine.pivot(index="source_domain", columns="target_domain", values="cosine_similarity")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cosine_matrix, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f")
        plt.title("Domain Direction Cosine Similarity")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(cosine_path), dpi=200)
        plt.close()

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
        plt.close()

    truth_patching_path = Path(args.truth_patching)
    if truth_patching_path.exists():
        truth_patching = pd.read_csv(truth_patching_path)
        matched_truth_patching = truth_patching
        if "control" in truth_patching.columns:
            matched_truth_patching = truth_patching[truth_patching["control"] == "matched_clean"]
        plt.figure(figsize=(7, 4))
        if "component" in matched_truth_patching.columns:
            sns.lineplot(
                data=matched_truth_patching,
                x="layer",
                y="mean_recovery",
                hue="component",
                marker="o",
            )
        else:
            sns.lineplot(data=matched_truth_patching, x="layer", y="mean_recovery", marker="o")
        plt.axhline(0, color="black", linewidth=1)
        plt.axhline(1, color="black", linewidth=1, linestyle="--")
        plt.title("Truth Verification Patching Recovery")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(truth_patching_path), dpi=200)
        plt.close()

        if "mean_abs_logit_shift" in matched_truth_patching.columns:
            plt.figure(figsize=(7, 4))
            if "component" in matched_truth_patching.columns:
                sns.lineplot(
                    data=matched_truth_patching,
                    x="layer",
                    y="mean_abs_logit_shift",
                    hue="component",
                    marker="o",
                )
            else:
                sns.lineplot(data=matched_truth_patching, x="layer", y="mean_abs_logit_shift", marker="o")
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Truth Verification Patching Mean Absolute Logit Shift")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(truth_patching_path, "_logit_shift"), dpi=200)
            plt.close()

        if "control" in truth_patching.columns:
            resid_post = truth_patching[truth_patching["component"] == "resid_post"]
            if not resid_post.empty:
                plt.figure(figsize=(7, 4))
                sns.lineplot(data=resid_post, x="layer", y="mean_abs_logit_shift", hue="control", marker="o")
                plt.axhline(0, color="black", linewidth=1)
                plt.title("Matched vs Shuffled Residual Patching Shift")
                plt.tight_layout()
                plt.savefig(out_dir / output_png_name(truth_patching_path, "_control_shift"), dpi=200)
                plt.close()

    oracle_steering_path = Path(args.oracle_steering)
    if oracle_steering_path.exists():
        oracle = pd.read_csv(oracle_steering_path)
        plt.figure(figsize=(7, 4))
        sns.lineplot(
            data=oracle,
            x="alpha",
            y="accuracy_from_probe_score_threshold",
            marker="o",
            label="probe-threshold accuracy",
        )
        if "accuracy_from_logit_sign" in oracle.columns:
            sns.lineplot(
                data=oracle,
                x="alpha",
                y="accuracy_from_logit_sign",
                marker="o",
                label="logit-sign accuracy",
            )
        plt.ylim(0, 1.05)
        plt.title("Oracle Conditional Steering Accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(oracle_steering_path), dpi=200)
        plt.close()

        if "mean_probe_correct_margin" in oracle.columns:
            plt.figure(figsize=(7, 4))
            sns.lineplot(
                data=oracle,
                x="alpha",
                y="mean_probe_correct_margin",
                marker="o",
                label="probe margin",
            )
            if "mean_logit_correct_margin" in oracle.columns:
                sns.lineplot(
                    data=oracle,
                    x="alpha",
                    y="mean_logit_correct_margin",
                    marker="o",
                    label="logit margin",
                )
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Oracle Conditional Steering Margins")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(oracle_steering_path, "_margins"), dpi=200)
            plt.close()

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
        plt.close()

        if "fixed_direction_score_gap" in ablation.columns:
            plt.figure(figsize=(7, 4))
            sns.lineplot(data=ablation, x="strength", y="fixed_direction_score_gap", marker="o")
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Fixed Probe-Direction Score Gap After Ablation")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(ablation_path, "_score_gap"), dpi=200)
            plt.close()

    iterative_ablation_path = Path(args.iterative_ablation)
    if iterative_ablation_path.exists():
        iterative_ablation = pd.read_csv(iterative_ablation_path)
        plt.figure(figsize=(8, 4))
        sns.lineplot(
            data=iterative_ablation,
            x="directions_removed",
            y="separability_auc",
            hue="control",
            marker="o",
        )
        plt.axhline(0.5, color="black", linewidth=1)
        plt.ylim(0, 1.05)
        plt.title("Iterative Direction Ablation Controls")
        plt.tight_layout()
        plt.savefig(out_dir / output_png_name(iterative_ablation_path), dpi=200)
        plt.close()

    knowledge_summary_path = Path(args.knowledge_summary)
    if knowledge_summary_path.exists():
        knowledge = pd.read_csv(knowledge_summary_path)
        plot_knowledge = knowledge[
            knowledge["group"].isin(
                [
                    "heldout_rows",
                    "heldout_high_avg_token_margin",
                    "heldout_low_avg_token_margin",
                ]
            )
        ].copy()
        if not plot_knowledge.empty:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=plot_knowledge, x="group", y="separability_auc", hue="analysis")
            plt.axhline(0.5, color="black", linewidth=1)
            plt.ylim(0, 1.05)
            plt.xticks(rotation=20, ha="right")
            plt.title("Completion Metrics vs Residual Probe on Balanced Capital")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(knowledge_summary_path), dpi=200)
            plt.close()

    knowledge_details_path = Path(args.knowledge_details)
    if knowledge_details_path.exists():
        details = pd.read_csv(knowledge_details_path)
        x_col = "completion_avg_token_margin" if "completion_avg_token_margin" in details.columns else "knowledge_margin"
        if {x_col, "probe_prob_true", "label"}.issubset(details.columns):
            plot_details = details[details["split"] == "test"].copy() if "split" in details.columns else details
            plot_details["label_name"] = plot_details["label"].map({0: "false", 1: "true"})
            plt.figure(figsize=(7, 4))
            sns.scatterplot(
                data=plot_details,
                x=x_col,
                y="probe_prob_true",
                hue="label_name",
                style="knowledge_bin" if "knowledge_bin" in plot_details.columns else None,
            )
            plt.axvline(0, color="black", linewidth=1)
            plt.axhline(0.5, color="black", linewidth=1)
            plt.ylim(-0.03, 1.03)
            plt.title("Held-Out Probe Score vs Avg-Token Completion Margin")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(knowledge_details_path), dpi=200)
            plt.close()

    completion_steering_path = Path(args.completion_steering)
    if completion_steering_path.exists():
        completion_steering = pd.read_csv(completion_steering_path)
        heldout = completion_steering[completion_steering["split"] == "heldout_countries"].copy()
        if not heldout.empty:
            plt.figure(figsize=(8, 4))
            sns.lineplot(
                data=heldout,
                x="alpha",
                y="mean_delta_avg_token_margin",
                hue="direction_type",
                marker="o",
            )
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Completion-Margin Steering on Balanced Capital")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(completion_steering_path), dpi=200)
            plt.close()

            plt.figure(figsize=(8, 4))
            sns.lineplot(
                data=heldout,
                x="alpha",
                y="pairwise_avg_token_accuracy",
                hue="direction_type",
                marker="o",
            )
            plt.axhline(0.5, color="black", linewidth=1)
            plt.ylim(0, 1.05)
            plt.title("Completion-Margin Steering Pairwise Accuracy")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(completion_steering_path, "_pairwise_accuracy"), dpi=200)
            plt.close()

    position_paths = [
        Path(args.completion_steering_position_all),
        Path(args.completion_steering_position_prompt_final),
        Path(args.completion_steering_position_completion_internal),
    ]
    if all(path.exists() for path in position_paths):
        position_frames = [pd.read_csv(path) for path in position_paths]
        position_summary = pd.concat(position_frames, ignore_index=True)
        heldout_learned = position_summary[
            (position_summary["split"] == "heldout_countries")
            & (position_summary["direction_type"] == "learned_probe")
            & (position_summary["alpha"].isin([-4.0, 0.0, 4.0]))
        ].copy()
        if not heldout_learned.empty:
            plt.figure(figsize=(8, 4))
            sns.lineplot(
                data=heldout_learned,
                x="alpha",
                y="mean_delta_avg_token_margin",
                hue="position_mode",
                marker="o",
            )
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Completion Steering by Injection Position")
            plt.tight_layout()
            plt.savefig(out_dir / "completion_margin_steering_position_comparison.png", dpi=200)
            plt.close()

    null_path = Path(args.completion_steering_null)
    if null_path.exists():
        null_data = pd.read_csv(null_path)
        if not null_data.empty:
            plt.figure(figsize=(8, 4))
            controls = null_data[null_data["control_type"] != "learned_probe"].copy()
            learned = null_data[null_data["control_type"] == "learned_probe"].copy()
            sns.histplot(
                data=controls,
                x="mean_delta_avg_token_margin",
                hue="control_type",
                bins=16,
                element="step",
                stat="count",
                common_norm=False,
            )
            if not learned.empty:
                plt.axvline(
                    float(learned["mean_delta_avg_token_margin"].iloc[0]),
                    color="black",
                    linewidth=2,
                    linestyle="--",
                    label="learned probe",
                )
                plt.legend()
            plt.axvline(0, color="black", linewidth=1)
            plt.title("Completion Steering Null Distribution")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(null_path), dpi=200)
            plt.close()

    decomposition_path = Path(args.completion_steering_decomposition)
    if decomposition_path.exists():
        decomposition = pd.read_csv(decomposition_path)
        heldout_alpha4 = decomposition[
            (decomposition["split"] == "heldout_countries") & (decomposition["alpha"] == 4.0)
        ].copy()
        if not heldout_alpha4.empty:
            plot_frame = heldout_alpha4.melt(
                id_vars=["direction_type"],
                value_vars=[
                    "mean_delta_correct_avg_token_logprob",
                    "mean_delta_false_avg_token_logprob",
                    "mean_delta_avg_token_margin",
                ],
                var_name="component",
                value_name="mean_delta",
            )
            plt.figure(figsize=(9, 4))
            sns.barplot(data=plot_frame, x="direction_type", y="mean_delta", hue="component")
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Completion Steering Decomposition at Alpha +4")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(decomposition_path), dpi=200)
            plt.close()

    paired_path = Path(args.completion_steering_paired)
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        heldout = paired[paired["split"] == "heldout_countries"].copy()
        if not heldout.empty:
            heldout["error_low"] = heldout["estimate"] - heldout["ci_low"]
            heldout["error_high"] = heldout["ci_high"] - heldout["estimate"]
            plt.figure(figsize=(9, 4))
            ax = sns.barplot(data=heldout, x="metric", y="estimate", hue="comparison")
            for patch, (_, row) in zip(ax.patches, heldout.iterrows()):
                x = patch.get_x() + patch.get_width() / 2
                ax.errorbar(
                    x=x,
                    y=row["estimate"],
                    yerr=[[row["error_low"]], [row["error_high"]]],
                    color="black",
                    capsize=3,
                    linewidth=1,
                )
            plt.axhline(0, color="black", linewidth=1)
            plt.xticks(rotation=15, ha="right")
            plt.title("Paired Block Bootstrap: Learned Minus Control")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(paired_path), dpi=200)
            plt.close()

    projection_details_path = Path(args.unembedding_projection_details)
    if projection_details_path.exists():
        projection_details = pd.read_csv(projection_details_path)
        heldout_alpha4 = projection_details[
            (projection_details["split"] == "test") & (projection_details["alpha"] == 4.0)
        ].copy()
        if not heldout_alpha4.empty:
            plt.figure(figsize=(7, 5))
            sns.scatterplot(
                data=heldout_alpha4,
                x="predicted_delta_avg_token_margin",
                y="observed_delta_avg_token_margin",
                hue="direction_type",
            )
            plt.axhline(0, color="black", linewidth=1)
            plt.axvline(0, color="black", linewidth=1)
            plt.title("Static Unembedding Projection vs Observed Shift")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(projection_details_path), dpi=200)
            plt.close()

    projection_summary_path = Path(args.unembedding_projection_summary)
    if projection_summary_path.exists():
        projection_summary = pd.read_csv(projection_summary_path)
        heldout_alpha4 = projection_summary[
            (projection_summary["split"] == "test") & (projection_summary["alpha"] == 4.0)
        ].copy()
        if not heldout_alpha4.empty:
            plot_frame = heldout_alpha4.melt(
                id_vars=["direction_type"],
                value_vars=[
                    "mean_abs_observed_delta_avg_token_margin",
                    "mean_abs_predicted_delta_avg_token_margin",
                    "mean_abs_residual",
                ],
                var_name="quantity",
                value_name="value",
            )
            plt.figure(figsize=(9, 4))
            sns.barplot(data=plot_frame, x="direction_type", y="value", hue="quantity")
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Unembedding Projection Baseline Scale")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(projection_summary_path), dpi=200)
            plt.close()

    repeated_path = Path(args.repeated_split_steering)
    if repeated_path.exists():
        repeated = pd.read_csv(repeated_path)
        split_rows = repeated[repeated["seed"].astype(str) != "aggregate"].copy()
        if not split_rows.empty:
            plot_frame = split_rows.melt(
                id_vars=["seed"],
                value_vars=["learned_minus_random_mean", "learned_minus_permutation_mean"],
                var_name="comparison",
                value_name="learned_minus_control",
            )
            plt.figure(figsize=(9, 4))
            sns.barplot(data=plot_frame, x="seed", y="learned_minus_control", hue="comparison")
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Repeated Split Completion Steering")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(repeated_path), dpi=200)
            plt.close()

    ambiguous_path = Path(args.ambiguous_sensitivity)
    if ambiguous_path.exists():
        ambiguous = pd.read_csv(ambiguous_path)
        metrics = ambiguous[ambiguous["analysis"].isin(["residual_probe", "completion_total", "completion_avg_token"])].copy()
        steering = ambiguous[ambiguous["analysis"] == "prompt_final_steering"].copy()
        if not metrics.empty:
            plt.figure(figsize=(7, 4))
            sns.barplot(data=metrics, x="analysis", y="auc")
            plt.axhline(0.5, color="black", linewidth=1)
            plt.ylim(0, 1.05)
            plt.xticks(rotation=15, ha="right")
            plt.title("Ambiguous-Fact Sensitivity: Held-Out AUC")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(ambiguous_path, "_auc"), dpi=200)
            plt.close()
        if not steering.empty:
            plt.figure(figsize=(7, 4))
            sns.barplot(data=steering, x="direction", y="mean_delta_avg_token_margin")
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Ambiguous-Fact Sensitivity: Steering Delta")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(ambiguous_path, "_steering"), dpi=200)
            plt.close()

    rank_path = Path(args.candidate_rank_steering)
    if rank_path.exists():
        rank = pd.read_csv(rank_path)
        if not rank.empty:
            row = rank.iloc[0]
            plot_frame = pd.DataFrame(
                [
                    {
                        "metric": "mean correct rank",
                        "condition": "baseline",
                        "value": row["mean_baseline_correct_rank"],
                    },
                    {
                        "metric": "mean correct rank",
                        "condition": "steered",
                        "value": row["mean_steered_correct_rank"],
                    },
                    {
                        "metric": "top1 accuracy",
                        "condition": "baseline",
                        "value": row["baseline_top1_accuracy"],
                    },
                    {
                        "metric": "top1 accuracy",
                        "condition": "steered",
                        "value": row["steered_top1_accuracy"],
                    },
                ]
            )
            plt.figure(figsize=(7, 4))
            sns.barplot(data=plot_frame, x="metric", y="value", hue="condition")
            plt.title("Candidate-Set Rank Steering")
            plt.tight_layout()
            plt.savefig(out_dir / output_png_name(rank_path), dpi=200)
            plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
