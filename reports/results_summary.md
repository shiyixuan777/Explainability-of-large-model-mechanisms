# Results Summary

This file is generated from CSV artifacts by `python -m scripts.summarize_results`.
Use it as a consistency check for the report tables.

## Probe Sweep: Top Settings

| domain | prompt | layer | accuracy | auc | separability_auc |
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

| layer | accuracy | auc | separability_auc |
| --- | --- | --- | --- |
| 8 | 0.826 | 0.953 | 0.953 |
| 10 | 0.870 | 0.947 | 0.947 |
| 6 | 0.826 | 0.943 | 0.943 |
| 9 | 0.804 | 0.941 | 0.941 |
| 5 | 0.848 | 0.941 | 0.941 |

Top layers by accuracy:

| layer | accuracy | auc | separability_auc |
| --- | --- | --- | --- |
| 10 | 0.870 | 0.947 | 0.947 |
| 5 | 0.848 | 0.941 | 0.941 |
| 7 | 0.848 | 0.938 | 0.938 |
| 6 | 0.826 | 0.943 | 0.943 |
| 8 | 0.826 | 0.953 | 0.953 |

## Activation PCA

| layer | pc1_explained_variance | pc2_explained_variance | rows |
| --- | --- | --- | --- |
| 8 | 0.620 | 0.117 | 152 |

## Error Analysis

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

## Activation Patching: Best Layer by Component

| component | layer | mean_recovery | median_recovery | patched_logit_diff |
| --- | --- | --- | --- | --- |
| attn_out | 11 | 1.762 | 1.077 | -0.672 |
| resid_post | 11 | 1.000 | 1.000 | 0.264 |
| mlp_out | 7 | 0.388 | 0.074 | -1.022 |

## Probe-Direction Steering

| alpha | logit_sign_accuracy | probe_threshold_accuracy | mean_probe_score |
| --- | --- | --- | --- |
| -8.0 | 0.500 | 0.500 | -9.353 |
| -4.0 | 0.500 | 0.500 | -5.353 |
| -2.0 | 0.500 | 0.500 | -3.353 |
| -1.0 | 0.500 | 0.507 | -2.353 |
| 0.0 | 0.500 | 1.000 | -1.353 |
| 1.0 | 0.500 | 0.500 | -0.353 |
| 2.0 | 0.500 | 0.500 | 0.647 |
| 4.0 | 0.500 | 0.500 | 2.647 |
| 8.0 | 0.500 | 0.500 | 6.647 |

## Probe-Direction Ablation

| strength | fixed_direction_score_gap | fixed_direction_accuracy | retrained_probe_auc |
| --- | --- | --- | --- |
| 0.00 | 0.573 | 0.826 | 0.953 |
| 0.25 | 0.430 | 0.500 | 0.955 |
| 0.50 | 0.286 | 0.500 | 0.958 |
| 0.75 | 0.143 | 0.500 | 0.951 |
| 1.00 | -0.000 | 0.500 | 0.945 |
| 1.25 | -0.143 | 0.500 | 0.953 |
| 1.50 | -0.286 | 0.478 | 0.962 |
