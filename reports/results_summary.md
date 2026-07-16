# Results Summary

This file is generated from CSV artifacts by `python -m scripts.summarize_results`.
Use it as a compact table index for the report results.

Generated at: 2026-07-16T09:08:06
Git commit at generation: 0f157cf
Working tree dirty at generation: yes
Source directory: project root
Script: `scripts/summarize_results.py`

`direction_agnostic_auc = max(AUC, 1 - AUC)`. It diagnoses whether scores have a strong label-ranking relation regardless of sign; it is not a claim that the train-time label direction generalizes as a classifier.

`learned_percentile = 1.0` means no sampled null direction exceeded the learned effect in the sampled set; it is not a population percentile estimate. `mean_rank_delta > 0` means the correct candidate moved toward rank 1. Repeated-split flip counts are evaluation occurrences across overlapping splits, not necessarily unique countries.

## Core Result Index

| claim | key_result |
| --- | --- |
| Original lexical confound | layer 8 residual AUC 0.953; BOW direction-agnostic AUC 0.933 |
| Balanced readout | layer 6 AUC 0.809 |
| Score intervention | prompt-final delta 0.135 |
| Repeated split stability | 10/10 positive; mean 0.116 |
| Choice effect | pairwise change 0.025; wrong->correct events 6 |
| Candidate-set top-1 | 0.083 -> 0.125 |
| Mechanism boundary | single-direction ablation retrained AUC 0.786 |

## Original Surface Baselines

| domain | baseline | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- | --- |
| all | numeric_surface | 0.481 | 0.453 | 0.547 |
| all | bag_of_words | 0.281 | 0.192 | 0.808 |
| capital | numeric_surface | 0.478 | 0.451 | 0.549 |
| capital | bag_of_words | 0.174 | 0.067 | 0.933 |

## Probe Sweep: Top Settings

| domain | prompt | layer | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- | --- | --- |
| capital | answer | 8 | 0.826 | 0.953 | 0.953 |
| capital | statement_is | 6 | 0.804 | 0.940 | 0.940 |
| capital | question | 7 | 0.804 | 0.932 | 0.932 |
| continent | statement_is | 11 | 0.808 | 0.846 | 0.846 |
| element_symbol | answer | 3 | 0.208 | 0.181 | 0.819 |
| landmark_country | statement_is | 9 | 0.778 | 0.815 | 0.815 |
| element_symbol | statement_is | 3 | 0.292 | 0.208 | 0.792 |
| book_author | answer | 11 | 0.278 | 0.210 | 0.790 |
| continent | answer | 5 | 0.654 | 0.787 | 0.787 |
| book_author | question | 11 | 0.278 | 0.235 | 0.765 |

## Focused Capital Probe

Top layers by AUC:

| layer | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- |
| 8 | 0.826 | 0.953 | 0.953 |
| 10 | 0.870 | 0.947 | 0.947 |
| 6 | 0.826 | 0.943 | 0.943 |
| 9 | 0.804 | 0.941 | 0.941 |
| 5 | 0.848 | 0.941 | 0.941 |

Top layers by accuracy:

| layer | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- |
| 10 | 0.870 | 0.947 | 0.947 |
| 5 | 0.848 | 0.941 | 0.941 |
| 7 | 0.848 | 0.938 | 0.938 |
| 6 | 0.826 | 0.943 | 0.943 |
| 8 | 0.826 | 0.953 | 0.953 |

## Probe Seed Sensitivity

| layer | mean_accuracy | std_accuracy | mean_auc | std_auc | min_auc | max_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.790 | 0.040 | 0.899 | 0.039 | 0.832 | 0.953 |
| 5 | 0.779 | 0.069 | 0.878 | 0.046 | 0.822 | 0.941 |
| 10 | 0.797 | 0.045 | 0.873 | 0.047 | 0.824 | 0.947 |
| 11 | 0.754 | 0.038 | 0.848 | 0.038 | 0.800 | 0.902 |

## Lexically Balanced Capital Probe

| layer | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- |
| 6 | 0.625 | 0.809 | 0.809 |
| 8 | 0.625 | 0.802 | 0.802 |
| 7 | 0.667 | 0.783 | 0.783 |
| 10 | 0.667 | 0.778 | 0.778 |
| 9 | 0.625 | 0.771 | 0.771 |
| 4 | 0.625 | 0.759 | 0.759 |

## Lexically Balanced Surface Baselines

| domain | baseline | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- | --- |
| capital_balanced | numeric_surface | 0.500 | 0.500 | 0.500 |
| capital_balanced | bag_of_words | 0.500 | 0.500 | 0.500 |

## Lexically Balanced Probe Seed Sensitivity

| layer | mean_accuracy | mean_auc | min_auc | max_auc |
| --- | --- | --- | --- | --- |
| 6 | 0.740 | 0.813 | 0.750 | 0.845 |
| 8 | 0.733 | 0.804 | 0.781 | 0.825 |
| 10 | 0.722 | 0.782 | 0.773 | 0.795 |
| 11 | 0.715 | 0.755 | 0.717 | 0.807 |
| 4 | 0.677 | 0.752 | 0.695 | 0.786 |

## Capital Completion Margin Baseline

`grouping_margin_mean` is the mean of the margin column used to define or summarize the row group; for `residual_probe` rows it is not the mean probe score.

Rows named `heldout_high_avg_token_margin` and `heldout_low_avg_token_margin` are exploratory, post-hoc subsets defined by avg-token margin and are not used for confirmatory claims.

| analysis | group | rows | blocks | accuracy | auc | auc_ci_low | auc_ci_high | direction_agnostic_auc | grouping_margin_column | grouping_margin_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| completion_total | heldout_rows | 48 | 12 | 0.750 | 0.861 | 0.753 | 0.955 | 0.861 | completion_total_margin | 2.821 |
| completion_avg_token | heldout_rows | 48 | 12 | 0.625 | 0.786 | 0.674 | 0.891 | 0.786 | completion_avg_token_margin | 2.079 |
| residual_probe | heldout_rows | 48 | 12 | 0.625 | 0.809 | 0.708 | 0.922 | 0.809 | completion_avg_token_margin | 2.079 |
| completion_total | heldout_high_avg_token_margin | 24 | 11 | 0.833 | 0.944 | 0.750 | 1.000 | 0.944 | completion_total_margin | 4.289 |
| completion_avg_token | heldout_high_avg_token_margin | 24 | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | completion_avg_token_margin | 4.925 |
| residual_probe | heldout_high_avg_token_margin | 24 | 11 | 0.583 | 0.764 | 0.579 | 0.929 | 0.764 | completion_avg_token_margin | 4.925 |
| completion_total | heldout_low_avg_token_margin | 24 | 11 | 0.667 | 0.750 | 0.437 | 0.970 | 0.750 | completion_total_margin | 1.353 |
| completion_avg_token | heldout_low_avg_token_margin | 24 | 11 | 0.250 | 0.271 | 0.000 | 0.563 | 0.729 | completion_avg_token_margin | -0.767 |
| residual_probe | heldout_low_avg_token_margin | 24 | 11 | 0.667 | 0.847 | 0.681 | 1.000 | 0.847 | completion_avg_token_margin | -0.767 |

## Exploratory and Supplementary Diagnostics

### Activation PCA

| layer | pc1_explained_variance | pc2_explained_variance | rows |
| --- | --- | --- | --- |
| 8 | 0.620 | 0.117 | 152 |

### Output Readout Baselines

| domain | verbalizer | prompt | shots | accuracy | auc | predicted_true_rate | mean_logit_margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | yes_no | answer_yes_no | 0 | 0.502 | 0.542 | 0.009 | -1.995 |
| all | correct_incorrect | statement_correct | 0 | 0.500 | 0.509 | 1.000 | 2.056 |
| all | lower_true_false | statement_is | 0 | 0.500 | 0.499 | 1.000 | 1.578 |
| all | title_true_false | answer_True_False | 2 | 0.500 | 0.494 | 1.000 | 1.505 |
| capital | yes_no | answer_yes_no | 2 | 0.507 | 0.560 | 0.007 | -1.976 |
| capital | correct_incorrect | statement_correct | 0 | 0.500 | 0.495 | 1.000 | 2.052 |
| capital | lower_true_false | answer_true_false | 2 | 0.500 | 0.486 | 1.000 | 1.607 |
| capital | title_true_false | answer_True_False | 0 | 0.500 | 0.530 | 1.000 | 1.510 |

### Cross-Domain Direction Transfer

| source | target | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- | --- |
| continent | capital | 0.638 | 0.766 | 0.766 |
| landmark_country | capital | 0.539 | 0.735 | 0.735 |
| landmark_country | continent | 0.628 | 0.654 | 0.654 |
| science | capital | 0.526 | 0.648 | 0.648 |
| capital | continent | 0.488 | 0.633 | 0.633 |
| continent | landmark_country | 0.517 | 0.612 | 0.612 |
| capital | landmark_country | 0.550 | 0.574 | 0.574 |
| capital | science | 0.520 | 0.573 | 0.573 |
| science | book_author | 0.567 | 0.572 | 0.572 |
| capital | element_symbol | 0.512 | 0.571 | 0.571 |

### Domain Direction Cosine Summary

| mean_cross_domain_cosine | min_cross_domain_cosine | max_cross_domain_cosine |
| --- | --- | --- |
| 0.077 | -0.043 | 0.301 |

### Error Analysis

| test_rows | correct | wrong | accuracy |
| --- | --- | --- | --- |
| 46 | 38 | 8 | 0.826 |

Misclassified examples:

| statement | label | prediction | prob_true |
| --- | --- | --- | --- |
| The capital of Laos is Vientiane. | true | false | 0.003 |
| The capital of Canada is Amman. | false | true | 0.964 |
| The capital of Chile is Santiago. | true | false | 0.179 |
| The capital of India is New Delhi. | true | false | 0.217 |
| The capital of Morocco is Rabat. | true | false | 0.219 |
| The capital of Nigeria is Mexico City. | false | true | 0.688 |
| The capital of Kenya is Nairobi. | true | false | 0.347 |
| The capital of Nepal is Kathmandu. | true | false | 0.472 |

### Activation Patching: Best Layer by Component

| component | layer | mean_recovery | median_recovery | patched_logit_diff |
| --- | --- | --- | --- | --- |
| attn_out | 11 | 1.762 | 1.077 | -0.672 |
| resid_post | 11 | 1.000 | 1.000 | 0.264 |
| mlp_out | 7 | 0.388 | 0.074 | -1.022 |

### Truth Verification Patching

| component | layer | mean_recovery | median_recovery | patched_logit_diff | mean_abs_logit_shift | mean_abs_denominator |
| --- | --- | --- | --- | --- | --- | --- |
| resid_post | 11 | 1.000 | 1.000 | 1.547 | 0.076 | 0.076 |
| resid_post | 10 | 0.816 | 0.814 | 1.540 | 0.068 | 0.076 |
| resid_pre | 11 | 0.816 | 0.814 | 1.540 | 0.068 | 0.076 |
| resid_post | 9 | 0.607 | 0.622 | 1.538 | 0.055 | 0.076 |
| resid_pre | 10 | 0.607 | 0.622 | 1.538 | 0.055 | 0.076 |
| resid_post | 8 | 0.568 | 0.541 | 1.536 | 0.052 | 0.076 |
| resid_pre | 9 | 0.568 | 0.541 | 1.536 | 0.052 | 0.076 |
| mlp_out | 1 | 0.330 | 0.155 | 1.521 | 0.014 | 0.076 |

### Probe-Direction Steering

| alpha | logit_sign_accuracy | heldout_probe_threshold_accuracy | mean_probe_score | split | threshold_source |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.500 | 0.500 | -9.560 | group | train_midpoint |
| -4.0 | 0.500 | 0.522 | -5.560 | group | train_midpoint |
| -2.0 | 0.500 | 0.522 | -3.560 | group | train_midpoint |
| -1.0 | 0.500 | 0.522 | -2.560 | group | train_midpoint |
| 0.0 | 0.500 | 0.826 | -1.560 | group | train_midpoint |
| 1.0 | 0.500 | 0.500 | -0.560 | group | train_midpoint |
| 2.0 | 0.500 | 0.500 | 0.440 | group | train_midpoint |
| 4.0 | 0.500 | 0.500 | 2.440 | group | train_midpoint |
| 8.0 | 0.500 | 0.500 | 6.440 | group | train_midpoint |

### Oracle Conditional Steering

| alpha | logit_sign_accuracy | probe_threshold_accuracy | mean_logit_correct_margin | mean_probe_correct_margin | mode |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.500 | 0.826 | -0.025 | 0.286 | oracle_label_conditioned |
| 0.5 | 0.500 | 1.000 | -0.029 | 0.786 | oracle_label_conditioned |
| 1.0 | 0.500 | 1.000 | -0.033 | 1.286 | oracle_label_conditioned |
| 2.0 | 0.500 | 1.000 | -0.041 | 2.286 | oracle_label_conditioned |
| 4.0 | 0.500 | 1.000 | -0.057 | 4.286 | oracle_label_conditioned |
| 8.0 | 0.500 | 1.000 | -0.088 | 8.286 | oracle_label_conditioned |

## Main Balanced Steering Results

### Balanced Prompt-Final Completion-Margin Steering

| direction | alpha | mean_delta_avg_token_margin | delta_ci | pairwise_avg_accuracy | block_exact_accuracy |
| --- | --- | --- | --- | --- | --- |
| learned_probe | -4.0 | -0.130 | [-0.191, -0.072] | 0.625 | 0.250 |
| learned_probe | -2.0 | -0.066 | [-0.096, -0.036] | 0.625 | 0.250 |
| learned_probe | 0.0 | 0.000 | [0.000, 0.000] | 0.625 | 0.250 |
| learned_probe | 2.0 | 0.067 | [0.038, 0.098] | 0.625 | 0.250 |
| learned_probe | 4.0 | 0.135 | [0.077, 0.198] | 0.625 | 0.250 |
| random_direction | -4.0 | 0.029 | [0.017, 0.041] | 0.625 | 0.250 |
| random_direction | -2.0 | 0.015 | [0.008, 0.021] | 0.625 | 0.250 |
| random_direction | 0.0 | 0.000 | [0.000, 0.000] | 0.625 | 0.250 |
| random_direction | 2.0 | -0.015 | [-0.021, -0.008] | 0.625 | 0.250 |
| random_direction | 4.0 | -0.030 | [-0.042, -0.017] | 0.625 | 0.250 |
| label_permutation | -4.0 | 0.023 | [0.013, 0.035] | 0.625 | 0.250 |
| label_permutation | -2.0 | 0.012 | [0.007, 0.018] | 0.625 | 0.250 |
| label_permutation | 0.0 | 0.000 | [0.000, 0.000] | 0.625 | 0.250 |
| label_permutation | 2.0 | -0.011 | [-0.017, -0.006] | 0.625 | 0.250 |
| label_permutation | 4.0 | -0.022 | [-0.034, -0.013] | 0.625 | 0.250 |

### Prompt-Final Completion-Margin Steering Decomposition

| direction | alpha | delta_correct_logprob | delta_false_logprob | delta_margin | delta_margin_std | baseline_correct_shift | baseline_wrong_shift | baseline_correct_minus_wrong | baseline_diff_ci | sign_flips | baseline_delta_corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_permutation | -4.0 | 0.073 | 0.049 | 0.023 | 0.045 | 0.021 | 0.026 | -0.005 | [-0.038, 0.029] | 0 | 0.249 |
| label_permutation | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0 | nan |
| label_permutation | 4.0 | -0.072 | -0.050 | -0.022 | 0.046 | -0.021 | -0.025 | 0.004 | [-0.029, 0.038] | 0 | -0.278 |
| learned_probe | -4.0 | -0.287 | -0.157 | -0.130 | 0.131 | -0.126 | -0.136 | 0.010 | [-0.087, 0.117] | 0 | -0.030 |
| learned_probe | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0 | nan |
| learned_probe | 4.0 | 0.281 | 0.146 | 0.135 | 0.133 | 0.132 | 0.140 | -0.008 | [-0.110, 0.084] | 0 | 0.001 |
| random_direction | -4.0 | 0.004 | -0.025 | 0.029 | 0.050 | 0.029 | 0.030 | -0.002 | [-0.045, 0.039] | 0 | 0.281 |
| random_direction | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0 | nan |
| random_direction | 4.0 | -0.010 | 0.019 | -0.030 | 0.049 | -0.028 | -0.031 | 0.003 | [-0.038, 0.043] | 0 | -0.263 |

### Prompt-Final Completion Steering Paired Bootstrap

| metric | comparison | estimate | ci | ci_unit |
| --- | --- | --- | --- | --- |
| delta_avg_token_margin_alpha_4 | learned_probe_minus_random_direction | 0.165 | [0.096, 0.239] | pair_id_block |
| delta_avg_token_margin_alpha_4 | learned_probe_minus_label_permutation | 0.158 | [0.090, 0.232] | pair_id_block |
| slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_random_direction | 0.040 | [0.024, 0.058] | pair_id_block |
| slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_label_permutation | 0.039 | [0.023, 0.057] | pair_id_block |

### Completion Steering Position Decomposition

| position_mode | alpha | mean_delta_avg_token_margin | pairwise_avg_accuracy | block_exact_accuracy |
| --- | --- | --- | --- | --- |
| all | -4.0 | -0.126 | 0.625 | 0.250 |
| all | 0.0 | 0.000 | 0.625 | 0.250 |
| all | 4.0 | 0.133 | 0.625 | 0.250 |
| prompt-final-only | -4.0 | -0.130 | 0.625 | 0.250 |
| prompt-final-only | 0.0 | 0.000 | 0.625 | 0.250 |
| prompt-final-only | 4.0 | 0.135 | 0.625 | 0.250 |
| completion-internal-only | -4.0 | 0.003 | 0.625 | 0.250 |
| completion-internal-only | 0.0 | 0.000 | 0.625 | 0.250 |
| completion-internal-only | 4.0 | -0.002 | 0.625 | 0.250 |

### Position Decomposition Paired Bootstrap

| position_mode | metric | comparison | estimate | ci |
| --- | --- | --- | --- | --- |
| prompt-final-only | delta_avg_token_margin_alpha_4 | learned_probe_minus_random_direction | 0.165 | [0.096, 0.239] |
| prompt-final-only | delta_avg_token_margin_alpha_4 | learned_probe_minus_label_permutation | 0.158 | [0.090, 0.232] |
| prompt-final-only | slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_random_direction | 0.040 | [0.024, 0.058] |
| prompt-final-only | slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_label_permutation | 0.039 | [0.023, 0.057] |
| completion-internal-only | delta_avg_token_margin_alpha_4 | learned_probe_minus_random_direction | -0.000 | [-0.003, 0.004] |
| completion-internal-only | delta_avg_token_margin_alpha_4 | learned_probe_minus_label_permutation | 0.001 | [-0.004, 0.008] |
| completion-internal-only | slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_random_direction | 0.000 | [-0.001, 0.001] |
| completion-internal-only | slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_label_permutation | 0.000 | [-0.001, 0.002] |

### Completion Steering Null Distribution

| control_type | directions | mean_delta | null_95_interval | learned_effect | learned_percentile | empirical_p_ge_learned |
| --- | --- | --- | --- | --- | --- | --- |
| label_permutation | 20 | 0.010 | [-0.081, 0.088] | 0.135 | 1.000 | 0.048 |
| learned_probe | 1 | 0.135 | [0.135, 0.135] | 0.135 | 1.000 |  |
| random_direction | 50 | 0.009 | [-0.065, 0.091] | 0.135 | 1.000 | 0.020 |

### Repeated Split Completion Steering

| scope | splits | learned_delta | learned_delta_std | learned_delta_range | learned_minus_random_mean | learned_minus_permutation_mean | learned_gt_all_random_splits | learned_gt_all_permutation_splits | baseline_pairwise_accuracy | mean_pairwise_accuracy | pairwise_accuracy_change | total_sign_flips | wrong_to_correct_flips | correct_to_wrong_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggregate | 10 | 0.116 | 0.023 | [0.085, 0.150] | 0.125 | 0.119 | 10 | 10 | 0.700 | 0.725 | 0.025 | 6 | 6 | 0 |

### Ambiguous-Fact Sensitivity

| analysis | blocks | heldout_blocks | auc | delta | pairwise_accuracy | sign_flips |
| --- | --- | --- | --- | --- | --- | --- |
| dataset | 35 |  |  |  |  |  |
| residual_probe |  | 11 | 0.864 |  |  |  |
| completion_total |  | 11 | 0.959 |  |  |  |
| completion_avg_token |  | 11 | 0.913 |  |  |  |
| prompt_final_steering:learned_probe |  | 11 |  | 0.120 | 0.864 | 1 |
| prompt_final_steering:random_direction |  | 11 |  | -0.038 | 0.818 | 0 |
| prompt_final_steering:label_permutation |  | 11 |  | -0.005 | 0.818 | 0 |

### Candidate-Set Rank Steering

| heldout_countries | candidate_count | mean_rank_delta | rank_improved_count | rank_worsened_count | baseline_top1_accuracy | steered_top1_accuracy | top1_changed_count | selected_pair_margin_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 76 | 0.917 | 10 | 0 | 0.083 | 0.125 | 1 | 0.135 |

### Unembedding Projection Baseline

| direction | observed_mean | predicted_mean | observed_abs_mean | predicted_abs_mean | mean_abs_residual | corr | corr_squared |
| --- | --- | --- | --- | --- | --- | --- | --- |
| label_permutation | -0.026 | 0.000 | 0.042 | 0.464 | 0.452 | 0.381 | 0.145 |
| learned_probe | 0.133 | 0.000 | 0.153 | 0.621 | 0.669 | -0.188 | 0.035 |
| random_direction | -0.032 | 0.000 | 0.047 | 0.598 | 0.577 | 0.627 | 0.393 |

### Probe-Direction Ablation

| strength | fixed_direction_score_gap | fixed_direction_accuracy | retrained_probe_auc |
| --- | --- | --- | --- |
| 0.00 | 0.573 | 0.826 | 0.953 |
| 0.25 | 0.430 | 0.500 | 0.955 |
| 0.50 | 0.286 | 0.500 | 0.958 |
| 0.75 | 0.143 | 0.500 | 0.951 |
| 1.00 | -0.000 | 0.500 | 0.945 |
| 1.25 | -0.143 | 0.500 | 0.953 |
| 1.50 | -0.286 | 0.478 | 0.962 |

### Lexically Balanced Probe-Direction Ablation

| strength | fixed_direction_score_gap | fixed_direction_accuracy | retrained_probe_auc |
| --- | --- | --- | --- |
| 0.00 | 0.174 | 0.604 | 0.809 |
| 0.25 | 0.130 | 0.500 | 0.811 |
| 0.50 | 0.087 | 0.500 | 0.809 |
| 0.75 | 0.043 | 0.500 | 0.788 |
| 1.00 | 0.000 | 0.500 | 0.786 |
| 1.25 | -0.043 | 0.500 | 0.792 |
| 1.50 | -0.087 | 0.500 | 0.793 |

### Iterative Direction Ablation

| control | directions_removed | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- | --- |
| learned_iterative | 0 | 0.826 | 0.953 | 0.953 |
| learned_iterative | 1 | 0.826 | 0.945 | 0.945 |
| learned_iterative | 2 | 0.870 | 0.941 | 0.941 |
| learned_iterative | 4 | 0.848 | 0.911 | 0.911 |
| learned_iterative | 8 | 0.783 | 0.862 | 0.862 |
| learned_iterative | 12 | 0.739 | 0.832 | 0.832 |
| learned_iterative | 16 | 0.739 | 0.807 | 0.807 |
| random_direction | 0 | 0.826 | 0.953 | 0.953 |
| random_direction | 1 | 0.826 | 0.953 | 0.953 |
| random_direction | 2 | 0.826 | 0.953 | 0.953 |
| random_direction | 4 | 0.826 | 0.949 | 0.949 |
| random_direction | 8 | 0.826 | 0.953 | 0.953 |
| random_direction | 12 | 0.826 | 0.953 | 0.953 |
| random_direction | 16 | 0.826 | 0.951 | 0.951 |
| label_permutation | 0 | 0.826 | 0.953 | 0.953 |
| label_permutation | 1 | 0.826 | 0.953 | 0.953 |
| label_permutation | 2 | 0.826 | 0.953 | 0.953 |
| label_permutation | 4 | 0.826 | 0.953 | 0.953 |
| label_permutation | 8 | 0.826 | 0.945 | 0.945 |
| label_permutation | 12 | 0.848 | 0.945 | 0.945 |
| label_permutation | 16 | 0.848 | 0.953 | 0.953 |

### Lexically Balanced Iterative Direction Ablation

| control | directions_removed | accuracy | auc | direction_agnostic_auc |
| --- | --- | --- | --- | --- |
| learned_iterative | 0 | 0.625 | 0.809 | 0.809 |
| learned_iterative | 1 | 0.604 | 0.786 | 0.786 |
| learned_iterative | 2 | 0.688 | 0.778 | 0.778 |
| learned_iterative | 4 | 0.646 | 0.755 | 0.755 |
| learned_iterative | 8 | 0.688 | 0.757 | 0.757 |
| learned_iterative | 12 | 0.625 | 0.738 | 0.738 |
| learned_iterative | 16 | 0.646 | 0.726 | 0.726 |
| random_direction | 0 | 0.625 | 0.809 | 0.809 |
| random_direction | 1 | 0.625 | 0.809 | 0.809 |
| random_direction | 2 | 0.604 | 0.806 | 0.806 |
| random_direction | 4 | 0.625 | 0.804 | 0.804 |
| random_direction | 8 | 0.604 | 0.806 | 0.806 |
| random_direction | 12 | 0.625 | 0.797 | 0.797 |
| random_direction | 16 | 0.646 | 0.793 | 0.793 |
| label_permutation | 0 | 0.625 | 0.809 | 0.809 |
| label_permutation | 1 | 0.625 | 0.812 | 0.812 |
| label_permutation | 2 | 0.625 | 0.814 | 0.814 |
| label_permutation | 4 | 0.646 | 0.793 | 0.793 |
| label_permutation | 8 | 0.646 | 0.773 | 0.773 |
| label_permutation | 12 | 0.646 | 0.760 | 0.760 |
| label_permutation | 16 | 0.667 | 0.783 | 0.783 |
