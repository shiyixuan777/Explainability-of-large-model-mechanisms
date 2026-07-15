from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "README.md",
    "INSTALL.md",
    "requirements.txt",
    "data/facts.csv",
    "data/capital_balanced.csv",
    "reports/final_report.md",
    "reports/results_summary.md",
    "reports/submission_manifest.md",
    "reports/reproducibility_checklist.md",
    "reports/final_deliverable_checklist.md",
    "scripts/prepare_submission.py",
    "figures/probe_sweep.csv",
    "figures/probe_sweep_summary.png",
    "figures/probe_capital_answer.csv",
    "figures/probe_capital_answer.png",
    "figures/probe_seed_sensitivity_capital.csv",
    "figures/probe_seed_sensitivity_capital.png",
    "figures/probe_capital_balanced.csv",
    "figures/probe_capital_balanced.png",
    "figures/probe_seed_sensitivity_capital_balanced.csv",
    "figures/probe_seed_sensitivity_capital_balanced.png",
    "figures/pca_capital_layer8.csv",
    "figures/pca_capital_layer8.png",
    "figures/error_analysis_capital_layer8.csv",
    "figures/error_analysis_capital_layer8_errors.csv",
    "figures/domain_transfer_layer8.csv",
    "figures/domain_transfer_layer8.png",
    "figures/domain_transfer_layer8_separability.png",
    "figures/domain_direction_cosine_layer8.csv",
    "figures/domain_direction_cosine_layer8.png",
    "figures/surface_baselines.csv",
    "figures/surface_baselines.png",
    "figures/surface_baselines_capital_balanced.csv",
    "figures/surface_baselines_capital_balanced.png",
    "figures/capital_knowledge_margin_details.csv",
    "figures/capital_knowledge_margin_details.png",
    "figures/capital_knowledge_margin_summary.csv",
    "figures/capital_knowledge_margin_summary.png",
    "figures/output_readout_baselines.csv",
    "figures/output_readout_baselines.png",
    "figures/output_readout_baselines_best_by_domain.png",
    "figures/activation_patching_capital_recall.csv",
    "figures/activation_patching_capital_recall.png",
    "figures/truth_verification_patching_resid.csv",
    "figures/truth_verification_patching_details.csv",
    "figures/truth_verification_patching_resid.png",
    "figures/truth_verification_patching_resid_logit_shift.png",
    "figures/truth_verification_patching_resid_control_shift.png",
    "figures/steering_capital_probe_layer8.csv",
    "figures/steering_capital_probe_layer8.png",
    "figures/steering_capital_probe_layer8_accuracy.png",
    "figures/steering_capital_probe_layer8_probe_accuracy.png",
    "figures/oracle_steering_capital_probe_layer8.csv",
    "figures/oracle_steering_capital_probe_layer8.png",
    "figures/oracle_steering_capital_probe_layer8_margins.png",
    "figures/completion_margin_steering_details.csv",
    "figures/completion_margin_steering_summary.csv",
    "figures/completion_margin_steering_summary.png",
    "figures/completion_margin_steering_summary_pairwise_accuracy.png",
    "figures/completion_margin_steering_decomposition.csv",
    "figures/completion_margin_steering_decomposition.png",
    "figures/completion_margin_steering_paired_bootstrap.csv",
    "figures/completion_margin_steering_paired_bootstrap.png",
    "figures/completion_margin_steering_position_prompt_final_details.csv",
    "figures/completion_margin_steering_position_prompt_final_summary.csv",
    "figures/completion_margin_steering_position_prompt_final_decomposition.csv",
    "figures/completion_margin_steering_position_prompt_final_paired_bootstrap.csv",
    "figures/completion_margin_steering_position_completion_internal_details.csv",
    "figures/completion_margin_steering_position_completion_internal_summary.csv",
    "figures/completion_margin_steering_position_completion_internal_decomposition.csv",
    "figures/completion_margin_steering_position_completion_internal_paired_bootstrap.csv",
    "figures/completion_margin_steering_position_comparison.png",
    "figures/completion_margin_steering_null_distribution.csv",
    "figures/completion_margin_steering_null_summary.csv",
    "figures/completion_margin_steering_null_distribution.png",
    "figures/repeated_split_completion_steering_details.csv",
    "figures/repeated_split_completion_steering_summary.csv",
    "figures/repeated_split_completion_steering_summary.png",
    "data/capital_balanced_no_ambiguous.csv",
    "figures/ambiguous_fact_sensitivity_details.csv",
    "figures/ambiguous_fact_sensitivity_summary.csv",
    "figures/ambiguous_fact_sensitivity_summary_auc.png",
    "figures/ambiguous_fact_sensitivity_summary_steering.png",
    "figures/candidate_rank_steering_details.csv",
    "figures/candidate_rank_steering_summary.csv",
    "figures/candidate_rank_steering_summary.png",
    "figures/unembedding_projection_baseline_details.csv",
    "figures/unembedding_projection_baseline_details.png",
    "figures/unembedding_projection_baseline_summary.csv",
    "figures/unembedding_projection_baseline_summary.png",
    "figures/ablation_capital_probe_layer8.csv",
    "figures/ablation_capital_probe_layer8.png",
    "figures/ablation_capital_probe_layer8_score_gap.png",
    "figures/ablation_capital_balanced_layer6.csv",
    "figures/ablation_capital_balanced_layer6.png",
    "figures/ablation_capital_balanced_layer6_score_gap.png",
    "figures/iterative_ablation_capital_layer8.csv",
    "figures/iterative_ablation_capital_layer8.png",
    "figures/iterative_ablation_capital_balanced_layer6.csv",
    "figures/iterative_ablation_capital_balanced_layer6.png",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"OK  {message}")


def read_csv(path: str) -> pd.DataFrame:
    full_path = Path(path)
    check(full_path.exists(), f"{path} exists")
    return pd.read_csv(full_path)


def validate_required_files() -> None:
    for file_name in REQUIRED_FILES:
        path = Path(file_name)
        check(path.exists(), f"{file_name} exists")
        check(path.stat().st_size > 0, f"{file_name} is non-empty")


def validate_dataset() -> None:
    data = read_csv("data/facts.csv")
    check(len(data) == 528, "dataset has 528 rows")
    counts = data["label"].value_counts().to_dict()
    check(counts.get(0) == 264 and counts.get(1) == 264, "dataset is label-balanced")
    check(data["domain"].nunique() >= 7, "dataset covers at least 7 domains")

    balanced = read_csv("data/capital_balanced.csv")
    check(len(balanced) == 152, "lexically balanced capital dataset has 152 rows")
    balanced_counts = balanced["label"].value_counts().to_dict()
    check(
        balanced_counts.get(0) == 76 and balanced_counts.get(1) == 76,
        "lexically balanced capital dataset is label-balanced",
    )
    check(balanced["pair_id"].nunique() == 38, "lexically balanced capital dataset has 38 total blocks")


def validate_probe_results() -> None:
    sweep = read_csv("figures/probe_sweep.csv")
    capital_answer = sweep[(sweep["domain"] == "capital") & (sweep["prompt"] == "answer")]
    best_capital_auc = float(capital_answer["auc"].max())
    check(best_capital_auc >= 0.94, "capital+answer probe sweep AUC >= 0.94")

    probe = read_csv("figures/probe_capital_answer.csv")
    layer8 = probe.loc[probe["layer"] == 8].iloc[0]
    layer10 = probe.loc[probe["layer"] == 10].iloc[0]
    check(float(layer8["auc"]) >= 0.94, "focused capital layer 8 AUC >= 0.94")
    check(float(layer10["accuracy"]) >= 0.85, "focused capital layer 10 accuracy >= 0.85")

    seed_sensitivity = read_csv("figures/probe_seed_sensitivity_capital.csv")
    layer8_seeds = seed_sensitivity[seed_sensitivity["layer"] == 8]
    check(len(layer8_seeds) >= 6, "probe seed sensitivity covers at least 6 seeds for layer 8")
    check(float(layer8_seeds["auc"].mean()) >= 0.88, "layer 8 mean AUC across seeds >= 0.88")
    check(float(layer8_seeds["auc"].min()) >= 0.80, "layer 8 AUC stays above 0.80 across checked seeds")

    balanced_probe = read_csv("figures/probe_capital_balanced.csv")
    balanced_layer6 = balanced_probe.loc[balanced_probe["layer"] == 6].iloc[0]
    balanced_layer8 = balanced_probe.loc[balanced_probe["layer"] == 8].iloc[0]
    check(float(balanced_layer6["auc"]) >= 0.80, "balanced capital layer 6 AUC >= 0.80")
    check(float(balanced_layer8["auc"]) >= 0.79, "balanced capital layer 8 AUC >= 0.79")

    balanced_seed = read_csv("figures/probe_seed_sensitivity_capital_balanced.csv")
    balanced_layer6_seeds = balanced_seed[balanced_seed["layer"] == 6]
    balanced_layer8_seeds = balanced_seed[balanced_seed["layer"] == 8]
    check(float(balanced_layer6_seeds["auc"].mean()) >= 0.80, "balanced layer 6 mean AUC >= 0.80")
    check(float(balanced_layer8_seeds["auc"].mean()) >= 0.78, "balanced layer 8 mean AUC >= 0.78")


def validate_visual_and_error_results() -> None:
    pca = read_csv("figures/pca_capital_layer8.csv")
    check(len(pca) == 152, "PCA contains all 152 capital rows")
    check(float(pca["pc1_explained_variance"].iloc[0]) > 0, "PCA has positive PC1 variance")

    errors = read_csv("figures/error_analysis_capital_layer8_errors.csv")
    check(len(errors) == 8, "error analysis has 8 misclassified examples")

    transfer = read_csv("figures/domain_transfer_layer8.csv")
    check(transfer["source_domain"].nunique() >= 7, "domain transfer covers at least 7 source domains")
    check(transfer["target_domain"].nunique() >= 7, "domain transfer covers at least 7 target domains")
    cross_transfer = transfer[transfer["source_domain"] != transfer["target_domain"]]
    check(
        float(cross_transfer["separability_auc"].mean()) < 0.6,
        "mean cross-domain transfer separability remains weak",
    )

    cosine = read_csv("figures/domain_direction_cosine_layer8.csv")
    cross_cosine = cosine[cosine["source_domain"] != cosine["target_domain"]]
    check(
        abs(float(cross_cosine["cosine_similarity"].mean())) < 0.15,
        "mean cross-domain direction cosine is close to zero",
    )

    surface = read_csv("figures/surface_baselines.csv")
    capital_numeric = surface[(surface["domain"] == "capital") & (surface["baseline"] == "numeric_surface")].iloc[0]
    capital_bow = surface[(surface["domain"] == "capital") & (surface["baseline"] == "bag_of_words")].iloc[0]
    check(float(capital_numeric["separability_auc"]) < 0.6, "numeric surface baseline is weak on capital")
    check(float(capital_bow["separability_auc"]) > 0.9, "bag-of-words baseline exposes lexical artifacts")

    balanced_surface = read_csv("figures/surface_baselines_capital_balanced.csv")
    balanced_bow = balanced_surface[
        (balanced_surface["domain"] == "capital_balanced") & (balanced_surface["baseline"] == "bag_of_words")
    ].iloc[0]
    balanced_numeric = balanced_surface[
        (balanced_surface["domain"] == "capital_balanced") & (balanced_surface["baseline"] == "numeric_surface")
    ].iloc[0]
    check(
        abs(float(balanced_bow["separability_auc"]) - 0.5) < 0.001,
        "lexically balanced bag-of-words baseline is random",
    )
    check(
        abs(float(balanced_numeric["separability_auc"]) - 0.5) < 0.001,
        "lexically balanced numeric surface baseline is random",
    )

    readout = read_csv("figures/output_readout_baselines.csv")
    check({"all", "capital"}.issubset(set(readout["domain"])), "output readout covers all and capital domains")
    check({0, 2}.issubset(set(readout["shots"])), "output readout covers zero-shot and two-shot prompts")
    check(readout["verbalizer"].nunique() >= 4, "output readout compares at least four verbalizers")
    check(readout["single_token_readout"].all(), "all configured readout labels are single tokens")
    check(
        readout["predicted_true_rate"].between(0, 1).all(),
        "output readout predicted-true rates are valid probabilities",
    )

    knowledge = read_csv("figures/capital_knowledge_margin_summary.csv")
    heldout = knowledge[knowledge["group"] == "heldout_rows"]
    check(
        {"completion_total", "completion_avg_token", "residual_probe"}.issubset(set(heldout["analysis"])),
        "knowledge summary includes total, avg-token, and residual-probe metrics",
    )
    check(
        {"auc_ci_low", "auc_ci_high", "bootstrap_samples", "ci_unit"}.issubset(set(knowledge.columns)),
        "knowledge summary includes block bootstrap confidence intervals",
    )
    check(
        (heldout["ci_unit"] == "pair_id_block").all(),
        "knowledge bootstrap uses pair_id blocks as sampling units",
    )
    check(
        (heldout["bootstrap_samples"].astype(int) >= 1000).all(),
        "knowledge bootstrap uses at least 1000 valid resamples for held-out metrics",
    )
    total = heldout[heldout["analysis"] == "completion_total"].iloc[0]
    avg_token = heldout[heldout["analysis"] == "completion_avg_token"].iloc[0]
    probe = heldout[heldout["analysis"] == "residual_probe"].iloc[0]
    check(float(total["auc"]) > float(avg_token["auc"]), "total-logprob and avg-token completion metrics differ")
    check(float(avg_token["auc"]) < float(probe["auc"]), "avg-token completion metric does not dominate residual probe")

    knowledge_details = read_csv("figures/capital_knowledge_margin_details.csv")
    required_knowledge_columns = {
        "completion_total_margin",
        "completion_avg_token_margin",
        "correct_completion_tokens",
        "false_completion_tokens",
    }
    check(
        required_knowledge_columns.issubset(set(knowledge_details.columns)),
        "knowledge details include total and avg-token margin diagnostics",
    )
    heldout_details = knowledge_details[knowledge_details["split"] == "test"]
    token_mismatch_rows = heldout_details[
        heldout_details["correct_completion_tokens"] != heldout_details["false_completion_tokens"]
    ]
    check(len(token_mismatch_rows) > 0, "knowledge details expose held-out token-length mismatches")


def validate_interventions() -> None:
    patching = read_csv("figures/activation_patching_capital_recall.csv")
    resid11 = patching[(patching["component"] == "resid_post") & (patching["layer"] == 11)]
    check(not resid11.empty, "resid_post layer 11 patching row exists")
    check(float(resid11["mean_recovery"].iloc[0]) >= 0.99, "resid_post layer 11 recovery >= 0.99")

    truth_patching = read_csv("figures/truth_verification_patching_resid.csv")
    check(
        {"matched_clean", "shuffled_clean"}.issubset(set(truth_patching["control"])),
        "truth verification patching includes matched and shuffled controls",
    )
    check(
        {"resid_pre", "attn_out", "mlp_out", "resid_post"}.issubset(set(truth_patching["component"])),
        "truth verification patching includes residual and module components",
    )
    truth_matched_resid = truth_patching[
        (truth_patching["control"] == "matched_clean") & (truth_patching["component"] == "resid_post")
    ]
    truth_resid11 = truth_matched_resid.loc[truth_matched_resid["layer"] == 11].iloc[0]
    truth_resid8 = truth_matched_resid.loc[truth_matched_resid["layer"] == 8].iloc[0]
    check(float(truth_resid11["mean_recovery"]) >= 0.99, "truth verification resid_post layer 11 recovery >= 0.99")
    check(
        float(truth_resid11["mean_abs_logit_shift"]) < 0.1,
        "truth verification patching has small absolute logit shift",
    )
    check(
        float(truth_resid11["mean_abs_clean_minus_corrupt_denominator"]) > 0,
        "truth verification patching reports denominator size",
    )
    check(
        float(truth_resid8["mean_recovery"]) > 0.5,
        "truth verification patching has late-layer recovery by layer 8",
    )

    steering = read_csv("figures/steering_capital_probe_layer8.csv")
    check((steering["accuracy_from_logit_sign"] == 0.5).all(), "steering logit-sign accuracy stays at 0.5")
    check((steering["split"] == "group").all(), "steering uses group held-out split")
    check((steering["threshold_source"] == "train_midpoint").all(), "steering threshold is fit on train split")
    alpha0 = steering.loc[steering["alpha"] == 0].iloc[0]
    check(
        abs(float(alpha0["accuracy_from_probe_score_threshold"]) - 0.826087) < 0.001,
        "steering alpha=0 held-out probe accuracy matches focused probe accuracy",
    )
    score_delta = float(steering["mean_probe_score"].iloc[-1] - steering["mean_probe_score"].iloc[0])
    alpha_delta = float(steering["alpha"].iloc[-1] - steering["alpha"].iloc[0])
    check(abs(score_delta - alpha_delta) < 0.1, "steering probe score moves approximately with alpha")

    oracle = read_csv("figures/oracle_steering_capital_probe_layer8.csv")
    oracle_alpha0 = oracle.loc[oracle["alpha"] == 0].iloc[0]
    oracle_alpha05 = oracle.loc[oracle["alpha"] == 0.5].iloc[0]
    check(
        float(oracle_alpha05["accuracy_from_probe_score_threshold"])
        > float(oracle_alpha0["accuracy_from_probe_score_threshold"]),
        "oracle steering improves held-out probe-threshold accuracy",
    )
    check(
        (oracle["accuracy_from_logit_sign"] == 0.5).all(),
        "oracle steering still does not improve logit-sign accuracy",
    )
    check(
        float(oracle["mean_probe_correct_margin"].iloc[-1])
        > float(oracle["mean_probe_correct_margin"].iloc[0]),
        "oracle steering improves probe correct margin",
    )

    completion_steering = read_csv("figures/completion_margin_steering_summary.csv")
    check(
        {"learned_probe", "random_direction", "label_permutation"}.issubset(
            set(completion_steering["direction_type"])
        ),
        "completion-margin steering includes learned, random, and label-permutation controls",
    )
    check(
        {"heldout_countries", "all_countries"}.issubset(set(completion_steering["split"])),
        "completion-margin steering reports all and held-out country splits",
    )
    heldout_completion = completion_steering[completion_steering["split"] == "heldout_countries"]
    learned_pos = heldout_completion[
        (heldout_completion["direction_type"] == "learned_probe") & (heldout_completion["alpha"] == 4.0)
    ].iloc[0]
    learned_neg = heldout_completion[
        (heldout_completion["direction_type"] == "learned_probe") & (heldout_completion["alpha"] == -4.0)
    ].iloc[0]
    random_pos = heldout_completion[
        (heldout_completion["direction_type"] == "random_direction") & (heldout_completion["alpha"] == 4.0)
    ].iloc[0]
    permutation_pos = heldout_completion[
        (heldout_completion["direction_type"] == "label_permutation") & (heldout_completion["alpha"] == 4.0)
    ].iloc[0]
    baseline = heldout_completion[
        (heldout_completion["direction_type"] == "learned_probe") & (heldout_completion["alpha"] == 0.0)
    ].iloc[0]
    check(
        float(learned_pos["mean_delta_avg_token_margin"]) > 0,
        "positive learned completion steering increases avg-token margin",
    )
    check(
        float(learned_neg["mean_delta_avg_token_margin"]) < 0,
        "negative learned completion steering decreases avg-token margin",
    )
    check(
        abs(float(learned_pos["mean_delta_avg_token_margin"]))
        > 3 * abs(float(random_pos["mean_delta_avg_token_margin"])),
        "learned completion-margin shift is larger than random-direction shift",
    )
    check(
        abs(float(learned_pos["mean_delta_avg_token_margin"]))
        > 3 * abs(float(permutation_pos["mean_delta_avg_token_margin"])),
        "learned completion-margin shift is larger than label-permutation shift",
    )
    check(
        float(learned_pos["pairwise_avg_token_accuracy"]) == float(baseline["pairwise_avg_token_accuracy"]),
        "completion-margin steering does not change held-out pairwise preference accuracy",
    )
    check(
        int(learned_pos["bootstrap_samples"]) >= 1000,
        "completion-margin steering includes block bootstrap intervals",
    )

    completion_details = read_csv("figures/completion_margin_steering_details.csv")
    check(
        {"correct_completion_tokens", "false_completion_tokens", "delta_avg_token_margin"}.issubset(
            set(completion_details.columns)
        ),
        "completion-margin steering details include token counts and margin deltas",
    )

    completion_decomposition = read_csv("figures/completion_margin_steering_decomposition.csv")
    required_decomposition_columns = {
        "mean_delta_correct_avg_token_logprob",
        "mean_delta_false_avg_token_logprob",
        "mean_delta_margin_when_baseline_prefers_correct",
        "mean_delta_margin_when_baseline_prefers_false",
        "baseline_correct_minus_wrong_delta_margin",
        "baseline_correct_minus_wrong_delta_margin_ci_low",
        "baseline_correct_minus_wrong_delta_margin_ci_high",
        "sign_flip_total",
        "corr_baseline_margin_delta_margin",
    }
    check(
        required_decomposition_columns.issubset(set(completion_decomposition.columns)),
        "completion steering decomposition includes correct/false and baseline-preference diagnostics",
    )
    learned_decomp = completion_decomposition[
        (completion_decomposition["split"] == "heldout_countries")
        & (completion_decomposition["direction_type"] == "learned_probe")
        & (completion_decomposition["alpha"] == 4.0)
    ].iloc[0]
    check(
        float(learned_decomp["mean_delta_correct_avg_token_logprob"])
        > float(learned_decomp["mean_delta_false_avg_token_logprob"]),
        "learned completion steering raises correct avg-token logprob more than false logprob",
    )
    check(
        int(learned_decomp["sign_flip_total"]) == 0,
        "learned completion steering does not flip held-out completion preferences",
    )
    check(
        float(learned_decomp["baseline_correct_minus_wrong_delta_margin_ci_low"]) < 0
        and float(learned_decomp["baseline_correct_minus_wrong_delta_margin_ci_high"]) > 0,
        "learned completion steering does not clearly target initially wrong held-out facts",
    )

    completion_paired = read_csv("figures/completion_margin_steering_paired_bootstrap.csv")
    heldout_paired = completion_paired[completion_paired["split"] == "heldout_countries"]
    check(
        {"estimate", "ci_low", "ci_high", "ci_unit", "bootstrap_samples"}.issubset(set(completion_paired.columns)),
        "completion steering paired bootstrap includes difference CIs",
    )
    check(
        (heldout_paired["ci_unit"] == "pair_id_block").all(),
        "completion steering paired bootstrap uses pair_id blocks",
    )
    check(
        (heldout_paired["bootstrap_samples"].astype(int) >= 1000).all(),
        "completion steering paired bootstrap uses at least 1000 resamples",
    )
    check(
        (heldout_paired["ci_low"] > 0).all(),
        "held-out learned-minus-control paired bootstrap CIs are positive",
    )

    prompt_position = read_csv("figures/completion_margin_steering_position_prompt_final_summary.csv")
    internal_position = read_csv("figures/completion_margin_steering_position_completion_internal_summary.csv")
    prompt_learned = prompt_position[
        (prompt_position["split"] == "heldout_countries")
        & (prompt_position["direction_type"] == "learned_probe")
        & (prompt_position["alpha"] == 4.0)
    ].iloc[0]
    internal_learned = internal_position[
        (internal_position["split"] == "heldout_countries")
        & (internal_position["direction_type"] == "learned_probe")
        & (internal_position["alpha"] == 4.0)
    ].iloc[0]
    check(
        float(prompt_learned["mean_delta_avg_token_margin"]) > 0.12,
        "prompt-final-only steering preserves the learned completion-margin shift",
    )
    check(
        abs(float(internal_learned["mean_delta_avg_token_margin"])) < 0.01,
        "completion-internal-only steering has near-zero learned completion-margin shift",
    )
    check(
        float(prompt_learned["pairwise_avg_token_accuracy"])
        == float(internal_learned["pairwise_avg_token_accuracy"]),
        "position decomposition does not change held-out pairwise preference accuracy",
    )

    prompt_position_paired = read_csv("figures/completion_margin_steering_position_prompt_final_paired_bootstrap.csv")
    internal_position_paired = read_csv(
        "figures/completion_margin_steering_position_completion_internal_paired_bootstrap.csv"
    )
    prompt_delta_ci = prompt_position_paired[
        (prompt_position_paired["split"] == "heldout_countries")
        & (prompt_position_paired["metric"] == "delta_avg_token_margin_alpha_4")
    ]
    internal_delta_ci = internal_position_paired[
        (internal_position_paired["split"] == "heldout_countries")
        & (internal_position_paired["metric"] == "delta_avg_token_margin_alpha_4")
    ]
    check(
        (prompt_delta_ci["ci_low"] > 0).all(),
        "prompt-final-only learned-minus-control paired CIs are positive",
    )
    check(
        (internal_delta_ci["ci_low"] < 0).all() and (internal_delta_ci["ci_high"] > 0).all(),
        "completion-internal-only learned-minus-control paired CIs cross zero",
    )

    null_distribution = read_csv("figures/completion_margin_steering_null_distribution.csv")
    null_summary = read_csv("figures/completion_margin_steering_null_summary.csv")
    check(
        {"learned_probe", "random_direction", "label_permutation"}.issubset(
            set(null_distribution["control_type"])
        ),
        "completion steering null distribution includes learned, random, and permutation directions",
    )
    random_null = null_summary[null_summary["control_type"] == "random_direction"].iloc[0]
    permutation_null = null_summary[null_summary["control_type"] == "label_permutation"].iloc[0]
    check(int(random_null["directions"]) >= 50, "random null distribution uses at least 50 directions")
    check(int(permutation_null["directions"]) >= 20, "permutation null distribution uses at least 20 directions")
    check(
        float(random_null["learned_effect"]) > float(random_null["q975"]),
        "learned completion steering exceeds the sampled random null 97.5 percentile",
    )
    check(
        float(permutation_null["learned_effect"]) > float(permutation_null["q975"]),
        "learned completion steering exceeds the sampled permutation null 97.5 percentile",
    )

    repeated = read_csv("figures/repeated_split_completion_steering_summary.csv")
    repeated_splits = repeated[repeated["seed"].astype(str) != "aggregate"].copy()
    repeated_aggregate = repeated[repeated["seed"].astype(str) == "aggregate"].iloc[0]
    check(len(repeated_splits) >= 10, "repeated split steering covers at least 10 group splits")
    check(
        (repeated_splits["learned_delta"] > 0).all(),
        "learned completion steering shift is positive in every repeated split",
    )
    check(
        (repeated_splits["learned_minus_random_mean"] > 0).all(),
        "learned completion steering exceeds random-direction mean in every repeated split",
    )
    check(
        (repeated_splits["learned_minus_permutation_mean"] > 0).all(),
        "learned completion steering exceeds permutation mean in every repeated split",
    )
    check(
        int(repeated_aggregate["learned_gt_all_random"]) >= 8,
        "learned completion steering exceeds all sampled random directions in most repeated splits",
    )
    check(
        int(repeated_aggregate["learned_gt_all_permutation"]) >= 8,
        "learned completion steering exceeds all sampled permutation directions in most repeated splits",
    )
    check(
        float(repeated_aggregate["baseline_pairwise_accuracy"]) < float(repeated_aggregate["learned_pairwise_accuracy"]),
        "repeated split steering reports a small positive pairwise accuracy change",
    )
    check(
        int(repeated_aggregate["wrong_to_correct_flips"]) >= int(repeated_aggregate["correct_to_wrong_flips"]),
        "repeated split steering flip breakdown is not net harmful",
    )
    check(
        float(repeated_aggregate["learned_delta_min"]) > 0,
        "repeated split steering learned delta range stays positive",
    )

    ambiguous = read_csv("figures/ambiguous_fact_sensitivity_summary.csv")
    ambiguous_dataset = ambiguous[ambiguous["analysis"] == "dataset"].iloc[0]
    ambiguous_probe = ambiguous[ambiguous["analysis"] == "residual_probe"].iloc[0]
    ambiguous_learned = ambiguous[
        (ambiguous["analysis"] == "prompt_final_steering")
        & (ambiguous["direction"] == "learned_probe")
    ].iloc[0]
    ambiguous_random = ambiguous[
        (ambiguous["analysis"] == "prompt_final_steering")
        & (ambiguous["direction"] == "random_direction")
    ].iloc[0]
    check(int(ambiguous_dataset["removed_block_count"]) >= 3, "ambiguous-fact sensitivity removes disputed blocks")
    check(float(ambiguous_probe["auc"]) >= 0.80, "ambiguous-fact sensitivity preserves balanced probe signal")
    check(
        float(ambiguous_learned["mean_delta_avg_token_margin"]) > 0.08,
        "ambiguous-fact sensitivity preserves positive learned steering shift",
    )
    check(
        float(ambiguous_learned["mean_delta_avg_token_margin"])
        > float(ambiguous_random["mean_delta_avg_token_margin"]),
        "ambiguous-fact sensitivity learned steering exceeds random control",
    )

    candidate_rank = read_csv("figures/candidate_rank_steering_summary.csv").iloc[0]
    check(int(candidate_rank["candidate_count"]) >= 70, "candidate rank steering compares a broad capital candidate set")
    check(float(candidate_rank["mean_rank_delta"]) > 0, "candidate rank steering mildly improves correct-capital rank")
    check(
        float(candidate_rank["steered_top1_accuracy"]) <= 0.15,
        "candidate rank steering does not become strong top-1 factual choice",
    )

    projection_summary = read_csv("figures/unembedding_projection_baseline_summary.csv")
    projection_details = read_csv("figures/unembedding_projection_baseline_details.csv")
    check(
        {"predicted_delta_avg_token_margin", "observed_delta_avg_token_margin"}.issubset(
            set(projection_details.columns)
        ),
        "unembedding projection details include predicted and observed deltas",
    )
    required_projection_columns = {
        "mean_abs_predicted_delta_avg_token_margin",
        "mean_abs_observed_delta_avg_token_margin",
        "mean_abs_residual",
        "corr_predicted_observed",
        "corr_squared_predicted_observed",
    }
    check(
        required_projection_columns.issubset(set(projection_summary.columns)),
        "unembedding projection summary includes scale and fit diagnostics",
    )
    learned_projection = projection_summary[
        (projection_summary["split"] == "test")
        & (projection_summary["direction_type"] == "learned_probe")
        & (projection_summary["alpha"] == 4.0)
    ].iloc[0]
    check(
        float(learned_projection["corr_squared_predicted_observed"]) < 0.1,
        "static unembedding projection has low held-out fit to learned steering shifts",
    )

    ablation = read_csv("figures/ablation_capital_probe_layer8.csv")
    baseline_gap = float(ablation.loc[ablation["strength"] == 0, "fixed_direction_score_gap"].iloc[0])
    full_gap = float(ablation.loc[ablation["strength"] == 1.0, "fixed_direction_score_gap"].iloc[0])
    min_retrained_auc = float(ablation["auc"].min())
    check(baseline_gap > 0.5, "ablation baseline fixed-direction score gap > 0.5")
    check(abs(full_gap) < 0.01, "ablation strength=1 removes fixed-direction score gap")
    check(min_retrained_auc >= 0.94, "retrained probe AUC remains >= 0.94 after ablation")

    iterative = read_csv("figures/iterative_ablation_capital_layer8.csv")
    check(
        {"learned_iterative", "random_direction", "label_permutation"}.issubset(set(iterative["control"])),
        "iterative ablation includes learned, random, and label-permutation controls",
    )
    check(int(iterative["directions_removed"].max()) >= 16, "iterative ablation removes at least 16 directions")
    learned_step0 = iterative[
        (iterative["control"] == "learned_iterative") & (iterative["directions_removed"] == 0)
    ].iloc[0]
    check(float(learned_step0["separability_auc"]) >= 0.94, "iterative ablation starts from strong baseline AUC")

    balanced_ablation = read_csv("figures/ablation_capital_balanced_layer6.csv")
    balanced_base = balanced_ablation.loc[balanced_ablation["strength"] == 0].iloc[0]
    balanced_full = balanced_ablation.loc[balanced_ablation["strength"] == 1.0].iloc[0]
    check(
        float(balanced_base["baseline_separability_auc"]) >= 0.80,
        "balanced ablation starts from layer 6 probe AUC >= 0.80",
    )
    check(
        abs(float(balanced_full["fixed_direction_score_gap"])) < 0.01,
        "balanced ablation removes fixed-direction score gap",
    )
    check(
        float(balanced_full["separability_auc"]) >= 0.75,
        "balanced retrained probe remains above chance after one-direction ablation",
    )

    balanced_iterative = read_csv("figures/iterative_ablation_capital_balanced_layer6.csv")
    balanced_learned0 = balanced_iterative[
        (balanced_iterative["control"] == "learned_iterative") & (balanced_iterative["directions_removed"] == 0)
    ].iloc[0]
    balanced_learned16 = balanced_iterative[
        (balanced_iterative["control"] == "learned_iterative") & (balanced_iterative["directions_removed"] == 16)
    ].iloc[0]
    balanced_random16 = balanced_iterative[
        (balanced_iterative["control"] == "random_direction") & (balanced_iterative["directions_removed"] == 16)
    ].iloc[0]
    check(
        float(balanced_learned0["separability_auc"]) >= 0.80,
        "balanced iterative ablation starts from layer 6 probe AUC >= 0.80",
    )
    check(
        float(balanced_learned16["separability_auc"]) < float(balanced_learned0["separability_auc"]),
        "balanced learned iterative ablation reduces separability",
    )
    check(
        float(balanced_random16["separability_auc"]) > float(balanced_learned16["separability_auc"]),
        "balanced current-split random-direction control preserves more separability than learned removal",
    )


def validate_report_docs() -> None:
    report_text = Path("reports/final_report.md").read_text(encoding="utf-8")
    report_text_lower = report_text.lower()
    required_terms = [
        ("Problem framing", "问题定义"),
        ("Probe interpretation question", "Probe 到底读到了什么"),
        ("Cross-domain question", "是否跨领域稳定"),
        ("Downstream intervention question", "沿该方向干预是否产生下游效应"),
        ("Bao et al. reproduction target", "Bao et al."),
        ("Lexically balanced control", "词汇平衡"),
        ("Completion margin", "completion margin"),
        ("Bag-of-words artifact", "bag-of-words"),
        ("Balanced residual signal", "中等强度"),
        ("Output readout baseline", "output readout"),
        ("Truth verification patching", "truth verification patching"),
        ("Oracle steering", "oracle"),
        ("Steering output limitation", "不能改善"),
        ("Completion-margin steering", "completion-margin steering"),
        ("Steering diagnostic boundary", "早期诊断"),
        ("Ablation", "ablation"),
        ("Balanced ablation", "balanced ablation"),
        ("Iterative ablation", "iterative ablation"),
        ("Reproduction fidelity", "Reproduction Fidelity"),
        ("Project extension boundary", "本文扩展"),
        ("Layer-position localization", "layer-position level localization"),
        ("Direction-agnostic AUC note", "Direction-agnostic AUC"),
        ("Probe seed sensitivity", "seed sensitivity"),
        ("Domain consistency", "direction cosine"),
        ("Surface baseline", "surface baseline"),
        ("Repeated split steering", "repeated split steering"),
        ("Repeated split dependence caveat", "不能等同于 10 个相互独立实验"),
        ("Ambiguous fact sensitivity", "ambiguous-fact sensitivity"),
        ("Ambiguous sensitivity resplit caveat", "重新做 group split"),
        ("Candidate rank steering", "candidate-set rank"),
        ("Pairwise choice boundary", "held-out correct-vs-selected-wrong pairwise choice"),
        ("Evidence boundary", "尚不能被解释为纯粹 truth representation"),
    ]
    for label, term in required_terms:
        check(term.lower() in report_text_lower, f"final report mentions {label}")


def main() -> None:
    validate_required_files()
    validate_dataset()
    validate_probe_results()
    validate_visual_and_error_results()
    validate_interventions()
    validate_report_docs()
    print("\nProject validation passed.")


if __name__ == "__main__":
    main()
