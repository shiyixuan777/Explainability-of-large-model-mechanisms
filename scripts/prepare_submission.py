from __future__ import annotations

from pathlib import Path


REPORT_FILES = [
    ("Final report", "reports/final_report.md"),
    ("Results summary", "reports/results_summary.md"),
    ("Reproducibility checklist", "reports/reproducibility_checklist.md"),
    ("Final deliverable checklist", "reports/final_deliverable_checklist.md"),
]

CORE_RESULT_FILES = [
    ("Dataset", "data/facts.csv"),
    ("Lexically balanced capital dataset", "data/capital_balanced.csv"),
    ("Probe sweep CSV", "figures/probe_sweep.csv"),
    ("Probe sweep figure", "figures/probe_sweep_summary.png"),
    ("Focused capital probe CSV", "figures/probe_capital_answer.csv"),
    ("Focused capital probe figure", "figures/probe_capital_answer.png"),
    ("Probe seed sensitivity CSV", "figures/probe_seed_sensitivity_capital.csv"),
    ("Probe seed sensitivity figure", "figures/probe_seed_sensitivity_capital.png"),
    ("Lexically balanced capital probe CSV", "figures/probe_capital_balanced.csv"),
    ("Lexically balanced capital probe figure", "figures/probe_capital_balanced.png"),
    (
        "Lexically balanced probe seed sensitivity CSV",
        "figures/probe_seed_sensitivity_capital_balanced.csv",
    ),
    (
        "Lexically balanced probe seed sensitivity figure",
        "figures/probe_seed_sensitivity_capital_balanced.png",
    ),
    ("Lexically balanced surface baseline CSV", "figures/surface_baselines_capital_balanced.csv"),
    ("Lexically balanced surface baseline figure", "figures/surface_baselines_capital_balanced.png"),
    ("Capital completion margin details CSV", "figures/capital_knowledge_margin_details.csv"),
    ("Capital completion margin details figure", "figures/capital_knowledge_margin_details.png"),
    ("Capital completion margin summary CSV", "figures/capital_knowledge_margin_summary.csv"),
    ("Capital completion margin summary figure", "figures/capital_knowledge_margin_summary.png"),
    ("Activation PCA CSV", "figures/pca_capital_layer8.csv"),
    ("Activation PCA figure", "figures/pca_capital_layer8.png"),
    ("Error analysis CSV", "figures/error_analysis_capital_layer8.csv"),
    ("Error examples CSV", "figures/error_analysis_capital_layer8_errors.csv"),
    ("Domain transfer CSV", "figures/domain_transfer_layer8.csv"),
    ("Domain transfer figure", "figures/domain_transfer_layer8.png"),
    ("Domain transfer separability figure", "figures/domain_transfer_layer8_separability.png"),
    ("Domain direction cosine CSV", "figures/domain_direction_cosine_layer8.csv"),
    ("Domain direction cosine figure", "figures/domain_direction_cosine_layer8.png"),
    ("Surface baseline CSV", "figures/surface_baselines.csv"),
    ("Surface baseline figure", "figures/surface_baselines.png"),
    ("Output readout CSV", "figures/output_readout_baselines.csv"),
    ("Output readout figure", "figures/output_readout_baselines.png"),
    ("Output readout best-by-domain figure", "figures/output_readout_baselines_best_by_domain.png"),
    ("Activation patching CSV", "figures/activation_patching_capital_recall.csv"),
    ("Activation patching figure", "figures/activation_patching_capital_recall.png"),
    ("Truth verification patching CSV", "figures/truth_verification_patching_resid.csv"),
    ("Truth verification patching details CSV", "figures/truth_verification_patching_details.csv"),
    ("Truth verification patching figure", "figures/truth_verification_patching_resid.png"),
    ("Truth verification patching logit-shift figure", "figures/truth_verification_patching_resid_logit_shift.png"),
    ("Truth verification patching control-shift figure", "figures/truth_verification_patching_resid_control_shift.png"),
    ("Steering CSV", "figures/steering_capital_probe_layer8.csv"),
    ("Steering logit figure", "figures/steering_capital_probe_layer8.png"),
    ("Steering accuracy figure", "figures/steering_capital_probe_layer8_accuracy.png"),
    ("Steering probe accuracy figure", "figures/steering_capital_probe_layer8_probe_accuracy.png"),
    ("Oracle steering CSV", "figures/oracle_steering_capital_probe_layer8.csv"),
    ("Oracle steering figure", "figures/oracle_steering_capital_probe_layer8.png"),
    ("Oracle steering margins figure", "figures/oracle_steering_capital_probe_layer8_margins.png"),
    ("Balanced completion-margin steering details CSV", "figures/completion_margin_steering_details.csv"),
    ("Balanced completion-margin steering summary CSV", "figures/completion_margin_steering_summary.csv"),
    ("Balanced completion-margin steering figure", "figures/completion_margin_steering_summary.png"),
    (
        "Balanced completion-margin steering pairwise figure",
        "figures/completion_margin_steering_summary_pairwise_accuracy.png",
    ),
    ("Completion steering decomposition CSV", "figures/completion_margin_steering_decomposition.csv"),
    ("Completion steering decomposition figure", "figures/completion_margin_steering_decomposition.png"),
    ("Completion steering paired bootstrap CSV", "figures/completion_margin_steering_paired_bootstrap.csv"),
    ("Completion steering paired bootstrap figure", "figures/completion_margin_steering_paired_bootstrap.png"),
    ("Completion steering position prompt-final details CSV", "figures/completion_margin_steering_position_prompt_final_details.csv"),
    ("Completion steering position prompt-final summary CSV", "figures/completion_margin_steering_position_prompt_final_summary.csv"),
    ("Completion steering position prompt-final decomposition CSV", "figures/completion_margin_steering_position_prompt_final_decomposition.csv"),
    ("Completion steering position prompt-final paired bootstrap CSV", "figures/completion_margin_steering_position_prompt_final_paired_bootstrap.csv"),
    ("Completion steering position completion-internal details CSV", "figures/completion_margin_steering_position_completion_internal_details.csv"),
    ("Completion steering position completion-internal summary CSV", "figures/completion_margin_steering_position_completion_internal_summary.csv"),
    ("Completion steering position completion-internal decomposition CSV", "figures/completion_margin_steering_position_completion_internal_decomposition.csv"),
    ("Completion steering position completion-internal paired bootstrap CSV", "figures/completion_margin_steering_position_completion_internal_paired_bootstrap.csv"),
    ("Completion steering position comparison figure", "figures/completion_margin_steering_position_comparison.png"),
    ("Completion steering null distribution CSV", "figures/completion_margin_steering_null_distribution.csv"),
    ("Completion steering null summary CSV", "figures/completion_margin_steering_null_summary.csv"),
    ("Completion steering null distribution figure", "figures/completion_margin_steering_null_distribution.png"),
    ("Repeated split steering details CSV", "figures/repeated_split_completion_steering_details.csv"),
    ("Repeated split steering summary CSV", "figures/repeated_split_completion_steering_summary.csv"),
    ("Repeated split steering figure", "figures/repeated_split_completion_steering_summary.png"),
    ("Ambiguous-fact sensitivity filtered dataset", "data/capital_balanced_no_ambiguous.csv"),
    ("Ambiguous-fact sensitivity details CSV", "figures/ambiguous_fact_sensitivity_details.csv"),
    ("Ambiguous-fact sensitivity summary CSV", "figures/ambiguous_fact_sensitivity_summary.csv"),
    ("Ambiguous-fact sensitivity AUC figure", "figures/ambiguous_fact_sensitivity_summary_auc.png"),
    ("Ambiguous-fact sensitivity steering figure", "figures/ambiguous_fact_sensitivity_summary_steering.png"),
    ("Candidate rank steering details CSV", "figures/candidate_rank_steering_details.csv"),
    ("Candidate rank steering summary CSV", "figures/candidate_rank_steering_summary.csv"),
    ("Candidate rank steering figure", "figures/candidate_rank_steering_summary.png"),
    ("Unembedding projection baseline details CSV", "figures/unembedding_projection_baseline_details.csv"),
    ("Unembedding projection baseline details figure", "figures/unembedding_projection_baseline_details.png"),
    ("Unembedding projection baseline summary CSV", "figures/unembedding_projection_baseline_summary.csv"),
    ("Unembedding projection baseline summary figure", "figures/unembedding_projection_baseline_summary.png"),
    ("Ablation CSV", "figures/ablation_capital_probe_layer8.csv"),
    ("Ablation figure", "figures/ablation_capital_probe_layer8.png"),
    ("Ablation score gap figure", "figures/ablation_capital_probe_layer8_score_gap.png"),
    ("Lexically balanced ablation CSV", "figures/ablation_capital_balanced_layer6.csv"),
    ("Lexically balanced ablation figure", "figures/ablation_capital_balanced_layer6.png"),
    ("Lexically balanced ablation score gap figure", "figures/ablation_capital_balanced_layer6_score_gap.png"),
    ("Iterative ablation CSV", "figures/iterative_ablation_capital_layer8.csv"),
    ("Iterative ablation figure", "figures/iterative_ablation_capital_layer8.png"),
    ("Lexically balanced iterative ablation CSV", "figures/iterative_ablation_capital_balanced_layer6.csv"),
    ("Lexically balanced iterative ablation figure", "figures/iterative_ablation_capital_balanced_layer6.png"),
]

SCRIPT_FILES = [
    ("Dataset builder", "scripts/build_dataset.py"),
    ("Lexically balanced dataset builder", "scripts/build_balanced_capital_dataset.py"),
    ("Environment check", "scripts/check_env.py"),
    ("Probe", "scripts/run_probe.py"),
    ("Probe sweep", "scripts/run_probe_sweep.py"),
    ("Probe seed sensitivity", "scripts/run_probe_seed_sensitivity.py"),
    ("Activation PCA", "scripts/run_activation_pca.py"),
    ("Error analysis", "scripts/run_error_analysis.py"),
    ("Domain consistency", "scripts/run_domain_consistency.py"),
    ("Surface baselines", "scripts/run_surface_baselines.py"),
    ("Output readout baselines", "scripts/run_output_readout_baselines.py"),
    ("Capital completion margin", "scripts/run_capital_knowledge_margin.py"),
    ("Activation patching", "scripts/run_activation_patching.py"),
    ("Truth verification patching", "scripts/run_truth_verification_patching.py"),
    ("Steering", "scripts/run_steering.py"),
    ("Oracle steering", "scripts/run_oracle_steering.py"),
    ("Balanced completion-margin steering", "scripts/run_completion_margin_steering.py"),
    ("Completion steering diagnostics", "scripts/analyze_completion_steering_diagnostics.py"),
    ("Completion steering null distribution", "scripts/run_completion_margin_null_distribution.py"),
    ("Repeated split completion steering", "scripts/run_repeated_split_completion_steering.py"),
    ("Ambiguous fact sensitivity", "scripts/run_ambiguous_fact_sensitivity.py"),
    ("Candidate rank steering", "scripts/run_candidate_rank_steering.py"),
    ("Unembedding projection baseline", "scripts/run_unembedding_projection_baseline.py"),
    ("Ablation", "scripts/run_ablation.py"),
    ("Iterative ablation", "scripts/run_iterative_ablation.py"),
    ("Plotting", "scripts/plot_results.py"),
    ("Results summary", "scripts/summarize_results.py"),
    ("Project validation", "scripts/validate_project.py"),
    ("Submission manifest", "scripts/prepare_submission.py"),
]


def status(path_text: str) -> str:
    path = Path(path_text)
    if not path.exists():
        return "MISSING"
    if path.is_file() and path.stat().st_size == 0:
        return "EMPTY"
    return "OK"


def table(rows: list[tuple[str, str]]) -> str:
    lines = [
        "| Item | Path | Status |",
        "|---|---|---|",
    ]
    for name, path in rows:
        lines.append(f"| {name} | `{path}` | {status(path)} |")
    return "\n".join(lines)


def main() -> None:
    lines = [
        "# Submission Manifest",
        "",
        "This file is generated by `python -m scripts.prepare_submission`.",
        "It lists the files that should be included or checked before final review.",
        "",
        "## Required Commands Before Submission",
        "",
        "```powershell",
        "python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md",
        "python -m scripts.validate_project",
        "```",
        "",
        "## Reports",
        "",
        table(REPORT_FILES),
        "",
        "## Core Results",
        "",
        table(CORE_RESULT_FILES),
        "",
        "## Code Entry Points",
        "",
        table(SCRIPT_FILES),
        "",
        "## Recommended Submission Focus",
        "",
        "- Main report: `reports/final_report.md`",
        "- Full codebase: `src/`, `scripts/`, `data/`, `figures/`, `reports/`",
        "- Reproducibility proof: `reports/reproducibility_checklist.md` and `reports/results_summary.md`",
        "",
    ]

    out_path = Path("reports/submission_manifest.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved submission manifest to {out_path}")


if __name__ == "__main__":
    main()
