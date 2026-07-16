from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--out", default="reports/results_summary.md")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No data available._\n"
    headers = list(rows[0].keys())
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(table)


def fmt(value: float | int | str, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def fmt_label(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()


def git_metadata() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unavailable"

    try:
        subprocess.check_call(
            ["git", "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty = "no"
    except Exception:
        dirty = "yes"
    return commit, dirty


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    out_path = Path(args.out)
    commit, dirty = git_metadata()

    lines: list[str] = [
        "# Results Summary",
        "",
        "This file is generated from CSV artifacts by `python -m scripts.summarize_results`.",
        "Use it as a compact table index for the report results.",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Git commit at generation: {commit}",
        f"Working tree dirty at generation: {dirty}",
        "Source directory: project root",
        "Script: `scripts/summarize_results.py`",
        "",
        "`direction_agnostic_auc = max(AUC, 1 - AUC)`. It diagnoses whether scores have a strong label-ranking relation regardless of sign; it is not a claim that the train-time label direction generalizes as a classifier.",
        "",
        "`learned_percentile = 1.0` means no sampled null direction exceeded the learned effect in the sampled set; it is not a population percentile estimate. `mean_rank_delta > 0` means the correct candidate moved toward rank 1. Repeated-split flip counts are evaluation occurrences across overlapping splits, not necessarily unique countries.",
        "",
    ]

    core_rows: list[dict[str, object]] = []
    original_probe_for_core = read_csv(figures_dir / "probe_capital_answer.csv")
    surface_for_core = read_csv(figures_dir / "surface_baselines.csv")
    if original_probe_for_core is not None and surface_for_core is not None:
        layer8 = original_probe_for_core[original_probe_for_core["layer"] == 8]
        bow = surface_for_core[
            (surface_for_core["domain"] == "capital") & (surface_for_core["baseline"] == "bag_of_words")
        ]
        if not layer8.empty and not bow.empty:
            core_rows.append(
                {
                    "claim": "Original lexical confound",
                    "key_result": (
                        f"layer 8 residual AUC {fmt(layer8.iloc[0]['auc'])}; "
                        f"BOW direction-agnostic AUC {fmt(bow.iloc[0]['separability_auc'])}"
                    ),
                }
            )
    balanced_probe_for_core = read_csv(figures_dir / "probe_capital_balanced.csv")
    if balanced_probe_for_core is not None:
        layer6 = balanced_probe_for_core[balanced_probe_for_core["layer"] == 6]
        if not layer6.empty:
            core_rows.append({"claim": "Balanced readout", "key_result": f"layer 6 AUC {fmt(layer6.iloc[0]['auc'])}"})
    prompt_final_for_core = read_csv(figures_dir / "completion_margin_steering_position_prompt_final_summary.csv")
    if prompt_final_for_core is not None:
        row = prompt_final_for_core[
            (prompt_final_for_core["split"] == "heldout_countries")
            & (prompt_final_for_core["direction_type"] == "learned_probe")
            & (prompt_final_for_core["alpha"] == 4.0)
        ]
        if not row.empty:
            core_rows.append(
                {
                    "claim": "Score intervention",
                    "key_result": f"prompt-final delta {fmt(row.iloc[0]['mean_delta_avg_token_margin'])}",
                }
            )
    repeated_for_core = read_csv(figures_dir / "repeated_split_completion_steering_summary.csv")
    if repeated_for_core is not None:
        aggregate = repeated_for_core[repeated_for_core["seed"].astype(str) == "aggregate"]
        split_rows_for_core = repeated_for_core[repeated_for_core["seed"].astype(str) != "aggregate"]
        if not aggregate.empty:
            row = aggregate.iloc[0]
            positive_splits = int((split_rows_for_core["learned_delta"] > 0).sum()) if not split_rows_for_core.empty else 0
            split_count = len(split_rows_for_core)
            core_rows.append(
                {
                    "claim": "Repeated split stability",
                    "key_result": f"{positive_splits}/{split_count} positive; mean {fmt(row.learned_delta)}",
                }
            )
            core_rows.append(
                {
                    "claim": "Choice effect",
                    "key_result": (
                        f"pairwise change {fmt(row.pairwise_accuracy_change)}; "
                        f"wrong->correct events {int(row.wrong_to_correct_flips)}"
                    ),
                }
            )
    rank_for_core = read_csv(figures_dir / "candidate_rank_steering_summary.csv")
    if rank_for_core is not None and not rank_for_core.empty:
        row = rank_for_core.iloc[0]
        core_rows.append(
            {
                "claim": "Candidate-set top-1",
                "key_result": f"{fmt(row.baseline_top1_accuracy)} -> {fmt(row.steered_top1_accuracy)}",
            }
        )
    balanced_ablation_for_core = read_csv(figures_dir / "ablation_capital_balanced_layer6.csv")
    if balanced_ablation_for_core is not None:
        strength1 = balanced_ablation_for_core[balanced_ablation_for_core["strength"] == 1.0]
        if not strength1.empty:
            core_rows.append(
                {
                    "claim": "Mechanism boundary",
                    "key_result": f"single-direction ablation retrained AUC {fmt(strength1.iloc[0]['auc'])}",
                }
            )
    if core_rows:
        lines += ["## Core Result Index", "", markdown_table(core_rows), ""]

    surface = read_csv(figures_dir / "surface_baselines.csv")
    if surface is not None:
        rows = [
            {
                "domain": row.domain,
                "baseline": row.baseline,
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in surface.itertuples(index=False)
        ]
        lines += ["## Original Surface Baselines", "", markdown_table(rows), ""]

    sweep = read_csv(figures_dir / "probe_sweep.csv")
    if sweep is not None:
        best = (
            sweep.sort_values("separability_auc", ascending=False)
            .groupby(["domain", "prompt"], as_index=False)
            .first()
            .sort_values("separability_auc", ascending=False)
            .head(10)
        )
        rows = [
            {
                "domain": row.domain,
                "prompt": row.prompt,
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in best.itertuples(index=False)
        ]
        lines += ["## Probe Sweep: Top Settings", "", markdown_table(rows), ""]

    probe = read_csv(figures_dir / "probe_capital_answer.csv")
    if probe is not None:
        top_auc = probe.sort_values("auc", ascending=False).head(5)
        top_acc = probe.sort_values("accuracy", ascending=False).head(5)
        rows_auc = [
            {
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in top_auc.itertuples(index=False)
        ]
        rows_acc = [
            {
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in top_acc.itertuples(index=False)
        ]
        lines += [
            "## Focused Capital Probe",
            "",
            "Top layers by AUC:",
            "",
            markdown_table(rows_auc),
            "",
            "Top layers by accuracy:",
            "",
            markdown_table(rows_acc),
            "",
        ]

    probe_seeds = read_csv(figures_dir / "probe_seed_sensitivity_capital.csv")
    if probe_seeds is not None:
        summary = (
            probe_seeds.groupby("layer", as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                mean_auc=("auc", "mean"),
                std_auc=("auc", "std"),
                min_auc=("auc", "min"),
                max_auc=("auc", "max"),
            )
            .sort_values("mean_auc", ascending=False)
        )
        rows = [
            {
                "layer": int(row.layer),
                "mean_accuracy": fmt(row.mean_accuracy),
                "std_accuracy": fmt(row.std_accuracy),
                "mean_auc": fmt(row.mean_auc),
                "std_auc": fmt(row.std_auc),
                "min_auc": fmt(row.min_auc),
                "max_auc": fmt(row.max_auc),
            }
            for row in summary.itertuples(index=False)
        ]
        lines += ["## Probe Seed Sensitivity", "", markdown_table(rows), ""]

    balanced_probe = read_csv(figures_dir / "probe_capital_balanced.csv")
    balanced_surface = read_csv(figures_dir / "surface_baselines_capital_balanced.csv")
    balanced_seeds = read_csv(figures_dir / "probe_seed_sensitivity_capital_balanced.csv")
    if balanced_probe is not None:
        top_balanced = balanced_probe.sort_values("auc", ascending=False).head(6)
        rows = [
            {
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in top_balanced.itertuples(index=False)
        ]
        lines += ["## Lexically Balanced Capital Probe", "", markdown_table(rows), ""]

    if balanced_surface is not None:
        rows = [
            {
                "domain": row.domain,
                "baseline": row.baseline,
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in balanced_surface.itertuples(index=False)
        ]
        lines += ["## Lexically Balanced Surface Baselines", "", markdown_table(rows), ""]

    if balanced_seeds is not None:
        summary = (
            balanced_seeds.groupby("layer", as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                mean_auc=("auc", "mean"),
                min_auc=("auc", "min"),
                max_auc=("auc", "max"),
            )
            .sort_values("mean_auc", ascending=False)
        )
        rows = [
            {
                "layer": int(row.layer),
                "mean_accuracy": fmt(row.mean_accuracy),
                "mean_auc": fmt(row.mean_auc),
                "min_auc": fmt(row.min_auc),
                "max_auc": fmt(row.max_auc),
            }
            for row in summary.itertuples(index=False)
        ]
        lines += ["## Lexically Balanced Probe Seed Sensitivity", "", markdown_table(rows), ""]

    knowledge = read_csv(figures_dir / "capital_knowledge_margin_summary.csv")
    if knowledge is not None:
        selected = knowledge[
            knowledge["group"].isin(
                [
                    "heldout_rows",
                    "heldout_high_avg_token_margin",
                    "heldout_low_avg_token_margin",
                ]
            )
        ]
        rows = [
            {
                "analysis": row.analysis,
                "group": row.group,
                "rows": int(row.rows),
                "blocks": int(getattr(row, "blocks", 0)),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "auc_ci_low": fmt(getattr(row, "auc_ci_low", 0.0)),
                "auc_ci_high": fmt(getattr(row, "auc_ci_high", 0.0)),
                "direction_agnostic_auc": fmt(row.separability_auc),
                "grouping_margin_column": getattr(row, "margin_column", ""),
                "grouping_margin_mean": fmt(row.mean_margin),
            }
            for row in selected.itertuples(index=False)
        ]
        lines += [
            "## Capital Completion Margin Baseline",
            "",
            "`grouping_margin_mean` is the mean of the margin column used to define or summarize the row group; for `residual_probe` rows it is not the mean probe score.",
            "",
            "Rows named `heldout_high_avg_token_margin` and `heldout_low_avg_token_margin` are exploratory, post-hoc subsets defined by avg-token margin and are not used for confirmatory claims.",
            "",
            markdown_table(rows),
            "",
        ]

    lines += ["## Exploratory and Supplementary Diagnostics", ""]

    pca = read_csv(figures_dir / "pca_capital_layer8.csv")
    if pca is not None:
        rows = [
            {
                "layer": int(pca["layer"].iloc[0]),
                "pc1_explained_variance": fmt(pca["pc1_explained_variance"].iloc[0]),
                "pc2_explained_variance": fmt(pca["pc2_explained_variance"].iloc[0]),
                "rows": len(pca),
            }
        ]
        lines += ["### Activation PCA", "", markdown_table(rows), ""]

    readout = read_csv(figures_dir / "output_readout_baselines.csv")
    if readout is not None:
        usable = readout[readout["single_token_readout"] == True].copy()
        if not usable.empty:
            best = (
                usable.sort_values("accuracy_from_logit_sign", ascending=False)
                .groupby(["domain", "verbalizer"], as_index=False)
                .first()
                .sort_values(["domain", "accuracy_from_logit_sign"], ascending=[True, False])
            )
            rows = [
                {
                    "domain": row.domain,
                    "verbalizer": row.verbalizer,
                    "prompt": row.prompt_name,
                    "shots": int(row.shots),
                    "accuracy": fmt(row.accuracy_from_logit_sign),
                    "auc": fmt(row.auc),
                    "predicted_true_rate": fmt(row.predicted_true_rate),
                    "mean_logit_margin": fmt(row.mean_true_minus_false_logit_diff),
                }
                for row in best.itertuples(index=False)
            ]
            lines += ["### Output Readout Baselines", "", markdown_table(rows), ""]

    transfer = read_csv(figures_dir / "domain_transfer_layer8.csv")
    if transfer is not None:
        cross = transfer[transfer["source_domain"] != transfer["target_domain"]]
        if not cross.empty:
            best_cross = cross.sort_values("separability_auc", ascending=False).head(10)
            rows = [
                {
                    "source": row.source_domain,
                    "target": row.target_domain,
                    "accuracy": fmt(row.accuracy),
                    "auc": fmt(row.auc),
                    "direction_agnostic_auc": fmt(row.separability_auc),
                }
                for row in best_cross.itertuples(index=False)
            ]
            lines += ["### Cross-Domain Direction Transfer", "", markdown_table(rows), ""]

    cosine = read_csv(figures_dir / "domain_direction_cosine_layer8.csv")
    if cosine is not None:
        cross = cosine[cosine["source_domain"] != cosine["target_domain"]]
        if not cross.empty:
            rows = [
                {
                    "mean_cross_domain_cosine": fmt(cross["cosine_similarity"].mean()),
                    "min_cross_domain_cosine": fmt(cross["cosine_similarity"].min()),
                    "max_cross_domain_cosine": fmt(cross["cosine_similarity"].max()),
                }
            ]
            lines += ["### Domain Direction Cosine Summary", "", markdown_table(rows), ""]

    errors = read_csv(figures_dir / "error_analysis_capital_layer8.csv")
    if errors is not None:
        total = len(errors)
        correct = int(errors["correct"].sum())
        wrong = total - correct
        rows = [
            {
                "test_rows": total,
                "correct": correct,
                "wrong": wrong,
                "accuracy": fmt(correct / total),
            }
        ]
        lines += ["### Error Analysis", "", markdown_table(rows), ""]

        error_rows = errors.loc[~errors["correct"]].sort_values("confidence", ascending=False).head(8)
        examples = [
            {
                "statement": row.statement,
                "label": fmt_label(row.label_name),
                "prediction": fmt_label(row.predicted_name),
                "prob_true": fmt(row.prob_true),
            }
            for row in error_rows.itertuples(index=False)
        ]
        lines += ["Misclassified examples:", "", markdown_table(examples), ""]

    patching = read_csv(figures_dir / "activation_patching_capital_recall.csv")
    if patching is not None:
        best = (
            patching.sort_values("mean_recovery", ascending=False)
            .groupby("component", as_index=False)
            .first()
            .sort_values("mean_recovery", ascending=False)
        )
        rows = [
            {
                "component": row.component,
                "layer": int(row.layer),
                "mean_recovery": fmt(row.mean_recovery),
                "median_recovery": fmt(row.median_recovery),
                "patched_logit_diff": fmt(row.patched_logit_diff),
            }
            for row in best.itertuples(index=False)
        ]
        lines += ["### Activation Patching: Best Layer by Component", "", markdown_table(rows), ""]

    truth_patching = read_csv(figures_dir / "truth_verification_patching_resid.csv")
    if truth_patching is not None:
        if "control" in truth_patching.columns:
            truth_for_table = truth_patching[truth_patching["control"] == "matched_clean"]
        else:
            truth_for_table = truth_patching
        best = truth_for_table.sort_values("mean_recovery", ascending=False).head(8)
        rows = [
            {
                "component": getattr(row, "component", "resid_post"),
                "layer": int(row.layer),
                "mean_recovery": fmt(row.mean_recovery),
                "median_recovery": fmt(row.median_recovery),
                "patched_logit_diff": fmt(row.patched_true_minus_false_logit_diff),
                "mean_abs_logit_shift": fmt(row.mean_abs_logit_shift),
                "mean_abs_denominator": fmt(getattr(row, "mean_abs_clean_minus_corrupt_denominator", 0.0)),
            }
            for row in best.itertuples(index=False)
        ]
        lines += ["### Truth Verification Patching", "", markdown_table(rows), ""]

    steering = read_csv(figures_dir / "steering_capital_probe_layer8.csv")
    if steering is not None:
        rows = [
            {
                "alpha": fmt(row.alpha, digits=1),
                "logit_sign_accuracy": fmt(row.accuracy_from_logit_sign),
                "heldout_probe_threshold_accuracy": fmt(row.accuracy_from_probe_score_threshold),
                "mean_probe_score": fmt(row.mean_probe_score),
                "split": getattr(row, "split", ""),
                "threshold_source": getattr(row, "threshold_source", ""),
            }
            for row in steering.itertuples(index=False)
        ]
        lines += ["### Probe-Direction Steering", "", markdown_table(rows), ""]

    oracle = read_csv(figures_dir / "oracle_steering_capital_probe_layer8.csv")
    if oracle is not None:
        rows = [
            {
                "alpha": fmt(row.alpha, digits=1),
                "logit_sign_accuracy": fmt(row.accuracy_from_logit_sign),
                "probe_threshold_accuracy": fmt(row.accuracy_from_probe_score_threshold),
                "mean_logit_correct_margin": fmt(row.mean_logit_correct_margin),
                "mean_probe_correct_margin": fmt(row.mean_probe_correct_margin),
                "mode": row.steering_mode,
            }
            for row in oracle.itertuples(index=False)
        ]
        lines += ["### Oracle Conditional Steering", "", markdown_table(rows), ""]

    lines += ["## Main Balanced Steering Results", ""]

    completion_steering = read_csv(figures_dir / "completion_margin_steering_position_prompt_final_summary.csv")
    if completion_steering is not None:
        heldout = completion_steering[
            (completion_steering["split"] == "heldout_countries")
            & (completion_steering["alpha"].isin([-4.0, -2.0, 0.0, 2.0, 4.0]))
        ]
        rows = [
            {
                "direction": row.direction_type,
                "alpha": fmt(row.alpha, digits=1),
                "mean_delta_avg_token_margin": fmt(row.mean_delta_avg_token_margin),
                "delta_ci": f"[{fmt(row.delta_avg_token_margin_ci_low)}, {fmt(row.delta_avg_token_margin_ci_high)}]",
                "pairwise_avg_accuracy": fmt(row.pairwise_avg_token_accuracy),
                "block_exact_accuracy": fmt(row.block_exact_avg_token_accuracy),
            }
            for row in heldout.itertuples(index=False)
        ]
        lines += ["### Balanced Prompt-Final Completion-Margin Steering", "", markdown_table(rows), ""]

    completion_decomposition = read_csv(figures_dir / "completion_margin_steering_position_prompt_final_decomposition.csv")
    if completion_decomposition is not None:
        selected = completion_decomposition[
            (completion_decomposition["split"] == "heldout_countries")
            & (completion_decomposition["alpha"].isin([-4.0, 0.0, 4.0]))
        ]
        rows = [
            {
                "direction": row.direction_type,
                "alpha": fmt(row.alpha, digits=1),
                "delta_correct_logprob": fmt(row.mean_delta_correct_avg_token_logprob),
                "delta_false_logprob": fmt(row.mean_delta_false_avg_token_logprob),
                "delta_margin": fmt(row.mean_delta_avg_token_margin),
                "delta_margin_std": fmt(row.std_delta_avg_token_margin),
                "baseline_correct_shift": fmt(row.mean_delta_margin_when_baseline_prefers_correct),
                "baseline_wrong_shift": fmt(row.mean_delta_margin_when_baseline_prefers_false),
                "baseline_correct_minus_wrong": fmt(row.baseline_correct_minus_wrong_delta_margin),
                "baseline_diff_ci": (
                    f"[{fmt(row.baseline_correct_minus_wrong_delta_margin_ci_low)}, "
                    f"{fmt(row.baseline_correct_minus_wrong_delta_margin_ci_high)}]"
                ),
                "sign_flips": int(row.sign_flip_total),
                "baseline_delta_corr": fmt(row.corr_baseline_margin_delta_margin),
            }
            for row in selected.itertuples(index=False)
        ]
        lines += ["### Prompt-Final Completion-Margin Steering Decomposition", "", markdown_table(rows), ""]

    completion_paired = read_csv(figures_dir / "completion_margin_steering_position_prompt_final_paired_bootstrap.csv")
    if completion_paired is not None:
        heldout = completion_paired[completion_paired["split"] == "heldout_countries"]
        rows = [
            {
                "metric": row.metric,
                "comparison": row.comparison,
                "estimate": fmt(row.estimate),
                "ci": f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}]",
                "ci_unit": row.ci_unit,
            }
            for row in heldout.itertuples(index=False)
        ]
        lines += ["### Prompt-Final Completion Steering Paired Bootstrap", "", markdown_table(rows), ""]

    position_summary_paths = [
        figures_dir / "completion_margin_steering_summary.csv",
        figures_dir / "completion_margin_steering_position_prompt_final_summary.csv",
        figures_dir / "completion_margin_steering_position_completion_internal_summary.csv",
    ]
    if all(path.exists() for path in position_summary_paths):
        position_summary = pd.concat([pd.read_csv(path) for path in position_summary_paths], ignore_index=True)
        selected = position_summary[
            (position_summary["split"] == "heldout_countries")
            & (position_summary["direction_type"] == "learned_probe")
            & (position_summary["alpha"].isin([-4.0, 0.0, 4.0]))
        ]
        rows = [
            {
                "position_mode": row.position_mode,
                "alpha": fmt(row.alpha, digits=1),
                "mean_delta_avg_token_margin": fmt(row.mean_delta_avg_token_margin),
                "pairwise_avg_accuracy": fmt(row.pairwise_avg_token_accuracy),
                "block_exact_accuracy": fmt(row.block_exact_avg_token_accuracy),
            }
            for row in selected.itertuples(index=False)
        ]
        lines += ["### Completion Steering Position Decomposition", "", markdown_table(rows), ""]

    position_paired_paths = [
        figures_dir / "completion_margin_steering_position_prompt_final_paired_bootstrap.csv",
        figures_dir / "completion_margin_steering_position_completion_internal_paired_bootstrap.csv",
    ]
    if all(path.exists() for path in position_paired_paths):
        paired_frames = []
        for path in position_paired_paths:
            frame = pd.read_csv(path)
            mode = "prompt-final-only" if "prompt_final" in path.name else "completion-internal-only"
            frame["position_mode"] = mode
            paired_frames.append(frame)
        position_paired = pd.concat(paired_frames, ignore_index=True)
        heldout = position_paired[position_paired["split"] == "heldout_countries"]
        rows = [
            {
                "position_mode": row.position_mode,
                "metric": row.metric,
                "comparison": row.comparison,
                "estimate": fmt(row.estimate),
                "ci": f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}]",
            }
            for row in heldout.itertuples(index=False)
        ]
        lines += ["### Position Decomposition Paired Bootstrap", "", markdown_table(rows), ""]

    null_summary = read_csv(figures_dir / "completion_margin_steering_null_summary.csv")
    if null_summary is not None:
        rows = [
            {
                "control_type": row.control_type,
                "directions": int(row.directions),
                "mean_delta": fmt(row.mean_delta),
                "null_95_interval": f"[{fmt(row.q025)}, {fmt(row.q975)}]",
                "learned_effect": fmt(row.learned_effect),
                "learned_percentile": fmt(row.learned_percentile),
                "empirical_p_ge_learned": fmt(row.empirical_p_ge_learned)
                if pd.notna(row.empirical_p_ge_learned)
                else "",
            }
            for row in null_summary.itertuples(index=False)
        ]
        lines += ["### Completion Steering Null Distribution", "", markdown_table(rows), ""]

    repeated = read_csv(figures_dir / "repeated_split_completion_steering_summary.csv")
    if repeated is not None:
        aggregate = repeated[repeated["seed"].astype(str) == "aggregate"]
        split_rows = repeated[repeated["seed"].astype(str) != "aggregate"]
        rows = []
        if not aggregate.empty:
            row = aggregate.iloc[0]
            rows.append(
                {
                    "scope": "aggregate",
                    "splits": len(split_rows),
                    "learned_delta": fmt(row.learned_delta),
                    "learned_delta_std": fmt(row.learned_delta_std),
                    "learned_delta_range": f"[{fmt(row.learned_delta_min)}, {fmt(row.learned_delta_max)}]",
                    "learned_minus_random_mean": fmt(row.learned_minus_random_mean),
                    "learned_minus_permutation_mean": fmt(row.learned_minus_permutation_mean),
                    "learned_gt_all_random_splits": int(row.learned_gt_all_random),
                    "learned_gt_all_permutation_splits": int(row.learned_gt_all_permutation),
                    "baseline_pairwise_accuracy": fmt(row.baseline_pairwise_accuracy),
                    "mean_pairwise_accuracy": fmt(row.learned_pairwise_accuracy),
                    "pairwise_accuracy_change": fmt(row.pairwise_accuracy_change),
                    "total_sign_flips": int(row.learned_sign_flips),
                    "wrong_to_correct_flips": int(row.wrong_to_correct_flips),
                    "correct_to_wrong_flips": int(row.correct_to_wrong_flips),
                }
            )
        lines += ["### Repeated Split Completion Steering", "", markdown_table(rows), ""]

    ambiguous = read_csv(figures_dir / "ambiguous_fact_sensitivity_summary.csv")
    if ambiguous is not None:
        rows = []
        for row in ambiguous.itertuples(index=False):
            if row.analysis == "dataset":
                rows.append(
                    {
                        "analysis": row.analysis,
                        "blocks": int(row.blocks),
                        "heldout_blocks": "",
                        "auc": "",
                        "delta": "",
                        "pairwise_accuracy": "",
                        "sign_flips": "",
                    }
                )
            elif row.analysis == "prompt_final_steering":
                rows.append(
                    {
                        "analysis": f"{row.analysis}:{row.direction}",
                        "blocks": "",
                        "heldout_blocks": int(row.heldout_blocks),
                        "auc": "",
                        "delta": fmt(row.mean_delta_avg_token_margin),
                        "pairwise_accuracy": fmt(row.pairwise_avg_accuracy),
                        "sign_flips": int(row.sign_flips),
                    }
                )
            else:
                rows.append(
                    {
                        "analysis": row.analysis,
                        "blocks": "",
                        "heldout_blocks": int(row.heldout_blocks),
                        "auc": fmt(row.auc),
                        "delta": "",
                        "pairwise_accuracy": "",
                        "sign_flips": "",
                    }
                )
        lines += ["### Ambiguous-Fact Sensitivity", "", markdown_table(rows), ""]

    rank = read_csv(figures_dir / "candidate_rank_steering_summary.csv")
    if rank is not None:
        rows = [
            {
                "heldout_countries": int(row.heldout_countries),
                "candidate_count": int(row.candidate_count),
                "mean_rank_delta": fmt(row.mean_rank_delta),
                "rank_improved_count": int(row.rank_improved_count),
                "rank_worsened_count": int(row.rank_worsened_count),
                "baseline_top1_accuracy": fmt(row.baseline_top1_accuracy),
                "steered_top1_accuracy": fmt(row.steered_top1_accuracy),
                "top1_changed_count": int(row.top1_changed_count),
                "selected_pair_margin_delta": fmt(row.mean_selected_pair_margin_delta),
            }
            for row in rank.itertuples(index=False)
        ]
        lines += ["### Candidate-Set Rank Steering", "", markdown_table(rows), ""]

    projection_summary = read_csv(figures_dir / "unembedding_projection_baseline_summary.csv")
    if projection_summary is not None:
        selected = projection_summary[
            (projection_summary["split"] == "test")
            & (projection_summary["alpha"] == 4.0)
        ]
        rows = [
            {
                "direction": row.direction_type,
                "observed_mean": fmt(row.mean_observed_delta_avg_token_margin),
                "predicted_mean": fmt(row.mean_predicted_delta_avg_token_margin),
                "observed_abs_mean": fmt(row.mean_abs_observed_delta_avg_token_margin),
                "predicted_abs_mean": fmt(row.mean_abs_predicted_delta_avg_token_margin),
                "mean_abs_residual": fmt(row.mean_abs_residual),
                "corr": fmt(row.corr_predicted_observed),
                "corr_squared": fmt(row.corr_squared_predicted_observed),
            }
            for row in selected.itertuples(index=False)
        ]
        lines += ["### Unembedding Projection Baseline", "", markdown_table(rows), ""]

    ablation = read_csv(figures_dir / "ablation_capital_probe_layer8.csv")
    if ablation is not None:
        rows = [
            {
                "strength": fmt(row.strength, digits=2),
                "fixed_direction_score_gap": fmt(row.fixed_direction_score_gap),
                "fixed_direction_accuracy": fmt(row.fixed_direction_accuracy),
                "retrained_probe_auc": fmt(row.auc),
            }
            for row in ablation.itertuples(index=False)
        ]
        lines += ["### Probe-Direction Ablation", "", markdown_table(rows), ""]

    balanced_ablation = read_csv(figures_dir / "ablation_capital_balanced_layer6.csv")
    if balanced_ablation is not None:
        rows = [
            {
                "strength": fmt(row.strength, digits=2),
                "fixed_direction_score_gap": fmt(row.fixed_direction_score_gap),
                "fixed_direction_accuracy": fmt(row.fixed_direction_accuracy),
                "retrained_probe_auc": fmt(row.auc),
            }
            for row in balanced_ablation.itertuples(index=False)
        ]
        lines += ["### Lexically Balanced Probe-Direction Ablation", "", markdown_table(rows), ""]

    iterative_ablation = read_csv(figures_dir / "iterative_ablation_capital_layer8.csv")
    if iterative_ablation is not None:
        key_steps = iterative_ablation[
            iterative_ablation["directions_removed"].isin([0, 1, 2, 4, 8, 12, 16])
        ]
        rows = [
            {
                "control": row.control,
                "directions_removed": int(row.directions_removed),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in key_steps.itertuples(index=False)
        ]
        lines += ["### Iterative Direction Ablation", "", markdown_table(rows), ""]

    balanced_iterative = read_csv(figures_dir / "iterative_ablation_capital_balanced_layer6.csv")
    if balanced_iterative is not None:
        key_steps = balanced_iterative[
            balanced_iterative["directions_removed"].isin([0, 1, 2, 4, 8, 12, 16])
        ]
        rows = [
            {
                "control": row.control,
                "directions_removed": int(row.directions_removed),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in key_steps.itertuples(index=False)
        ]
        lines += ["### Lexically Balanced Iterative Direction Ablation", "", markdown_table(rows), ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved results summary to {out_path}")


if __name__ == "__main__":
    main()
