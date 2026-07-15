# Reproducibility Checklist

This checklist records the commands and expected artifacts needed to reproduce the current Markdown report. It is a runbook, not the main result interpretation; exact numerical tables are centralized in `reports/results_summary.md`.

## Environment

Tested environment:

```text
OS: Windows 11 10.0.26200
Python: 3.13.7
PyTorch: 2.12.1+cpu
Transformers: 5.13.0
TransformerLens: 3.5.1
scikit-learn: 1.9.0
NumPy: 2.5.1
Pandas: 3.0.3
Device: CPU
```

GPU is not required for reproducing the small GPT-2 experiments, but CPU runs are slower. The first model-loading command downloads `gpt2-small` from Hugging Face and needs network access plus local cache space for the model weights. On the tested CPU environment, the balanced prompt-final steering command is the slowest Quick Start step and takes roughly a few minutes; the full runbook can take substantially longer because it includes exploratory diagnostics and repeated splits.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Expected result: dependencies install successfully and the dataset-building commands below can be run from the activated environment.

## Quick Start

These commands reproduce the main balanced-data evidence chain used in the final report. The full command list below keeps the auxiliary diagnostics and earlier exploratory experiments.

```powershell
python -m scripts.build_dataset
python -m scripts.build_balanced_capital_dataset --out data/capital_balanced.csv
python -m scripts.run_probe --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --prompt-template "Statement: {statement}`nAnswer true or false:" --seed 42 --out figures/probe_capital_balanced.csv
python -m scripts.run_surface_baselines --data data/capital_balanced.csv --language en --domains capital_balanced --seed 42 --out figures/surface_baselines_capital_balanced.csv
python -m scripts.run_capital_knowledge_margin --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --bootstrap-samples 2000 --out-details figures/capital_knowledge_margin_details.csv --out-summary figures/capital_knowledge_margin_summary.csv
python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --position-mode prompt-final-only --bootstrap-samples 2000 --alphas -4 -2 -1 0 1 2 4 --out-details figures/completion_margin_steering_position_prompt_final_details.csv --out-summary figures/completion_margin_steering_position_prompt_final_summary.csv
python -m scripts.run_completion_margin_null_distribution --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --position-mode prompt-final-only --alpha 4 --random-directions 50 --permutation-directions 20 --out-details figures/completion_margin_steering_null_distribution.csv --out-summary figures/completion_margin_steering_null_summary.csv
python -m scripts.run_repeated_split_completion_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seeds 0 1 2 3 4 5 6 7 8 9 --alpha 4 --random-directions 10 --permutation-directions 5 --position-mode prompt-final-only --out-details figures/repeated_split_completion_steering_details.csv --out-summary figures/repeated_split_completion_steering_summary.csv
python -m scripts.plot_results
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.validate_project
python -m compileall scripts src
```

Main sanity checks: balanced surface baselines should be near random; balanced layer 6 probe AUC should be around 0.81; prompt-final learned steering should produce a positive avg-token completion-margin shift larger than the sampled random/permutation controls.

## Dataset

```powershell
python -m scripts.build_dataset
python -m scripts.build_balanced_capital_dataset --out data/capital_balanced.csv
```

Expected artifacts:

```text
data/facts.csv
data/capital_balanced.csv
```

Expected summary:

```text
Rows: 528
label 0: 264
label 1: 264

Balanced capital:
Rows: 152
label 0: 76
label 1: 76
blocks: 38
```

## Locate: Probe Sweep

```powershell
python -m scripts.run_probe_sweep --model gpt2-small --data data/facts.csv --seed 42 --out figures/probe_sweep.csv
```

Expected artifacts:

```text
figures/probe_sweep.csv
figures/probe_sweep_summary.png
```

Key expected result:

```text
capital + answer prompt: best AUC around 0.953
mixed-domain settings: clearly weaker than capital
```

## Locate: Focused Capital Probe

```powershell
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --seed 42 --out figures/probe_capital_answer.csv
```

Expected artifacts:

```text
figures/probe_capital_answer.csv
figures/probe_capital_answer.png
```

Key expected result:

```text
layer 8 AUC around 0.953
layer 10 accuracy around 0.870
```

## Locate: Probe Seed Sensitivity

```powershell
python -m scripts.run_probe_seed_sensitivity --model gpt2-small --data data/facts.csv --language en --domain capital --layers 5 8 10 11 --seeds 0 1 2 3 4 42 --out figures/probe_seed_sensitivity_capital.csv
```

Expected artifacts:

```text
figures/probe_seed_sensitivity_capital.csv
figures/probe_seed_sensitivity_capital.png
```

Key expected result:

```text
layer 8 mean AUC across the checked seeds is around 0.899
layer 8 AUC ranges from about 0.832 to 0.953
seed=42 is a strong but optimistic split
```

## Locate: Activation PCA

```powershell
python -m scripts.run_activation_pca --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/pca_capital_layer8.csv
```

Expected artifacts:

```text
figures/pca_capital_layer8.csv
figures/pca_capital_layer8.png
```

Key expected result:

```text
PC1 explained variance around 0.620
PC2 explained variance around 0.117
The 2D PCA projection does not cleanly separate true/false.
```

## Locate: Error Analysis

```powershell
python -m scripts.run_error_analysis --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --seed 42 --out figures/error_analysis_capital_layer8.csv
```

Expected artifacts:

```text
figures/error_analysis_capital_layer8.csv
figures/error_analysis_capital_layer8_errors.csv
```

Key expected result:

```text
layer 8 test accuracy around 0.826
8 misclassified test examples
```

## Locate: Cross-Domain Direction Consistency

```powershell
python -m scripts.run_domain_consistency --model gpt2-small --data data/facts.csv --language en --layer 8 --seed 42 --out-transfer figures/domain_transfer_layer8.csv --out-cosine figures/domain_direction_cosine_layer8.csv
```

Expected artifacts:

```text
figures/domain_transfer_layer8.csv
figures/domain_transfer_layer8.png
figures/domain_transfer_layer8_separability.png
figures/domain_direction_cosine_layer8.csv
figures/domain_direction_cosine_layer8.png
```

Key expected result:

```text
mean cross-domain direction cosine is around 0.077
most cross-domain transfer AUC values are close to 0.5
the best cross transfer is continent -> capital with AUC around 0.766
```

## Surface Baselines

```powershell
python -m scripts.run_surface_baselines --data data/facts.csv --language en --domains all capital --seed 42 --out figures/surface_baselines.csv
python -m scripts.run_surface_baselines --data data/capital_balanced.csv --language en --domains capital_balanced --seed 42 --out figures/surface_baselines_capital_balanced.csv
```

Expected artifacts:

```text
figures/surface_baselines.csv
figures/surface_baselines.png
figures/surface_baselines_capital_balanced.csv
figures/surface_baselines_capital_balanced.png
```

Key expected result:

```text
numeric surface baseline is weak on capital: direction-agnostic AUC around 0.549
bag-of-words baseline exposes lexical artifacts: capital direction-agnostic AUC around 0.933
on the balanced capital dataset, both numeric surface and bag-of-words baselines are exactly random
```

## Lexically Balanced Capital Probe

```powershell
python -m scripts.run_probe --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --prompt-template "Statement: {statement}`nAnswer true or false:" --seed 42 --out figures/probe_capital_balanced.csv
python -m scripts.run_probe_seed_sensitivity --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layers 4 6 8 10 11 --seeds 0 1 2 3 4 42 --out figures/probe_seed_sensitivity_capital_balanced.csv
```

Expected artifacts:

```text
figures/probe_capital_balanced.csv
figures/probe_capital_balanced.png
figures/probe_seed_sensitivity_capital_balanced.csv
figures/probe_seed_sensitivity_capital_balanced.png
```

Key expected result:

```text
balanced capital BOW separability is 0.500
balanced residual probe layer 6 AUC is around 0.809
balanced residual probe layer 8 AUC is around 0.802
layer 6 mean AUC across seeds is around 0.813
layer 8 mean AUC across seeds is around 0.804
```

## Completion Margin: Capital Completion Compatibility

```powershell
python -m scripts.run_capital_knowledge_margin --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --bootstrap-samples 2000 --out-details figures/capital_knowledge_margin_details.csv --out-summary figures/capital_knowledge_margin_summary.csv
```

Expected artifacts:

```text
figures/capital_knowledge_margin_details.csv
figures/capital_knowledge_margin_details.png
figures/capital_knowledge_margin_summary.csv
figures/capital_knowledge_margin_summary.png
```

Key expected result:

```text
held-out completion_total AUC is around 0.861 with block-bootstrap CI around 0.753-0.955
held-out completion_avg_token AUC is around 0.786 with block-bootstrap CI around 0.674-0.891
held-out residual_probe AUC on the same split is around 0.809 with block-bootstrap CI around 0.708-0.922
24 held-out rows have different correct/false completion token counts
The completion baseline is exploratory because total and avg-token metrics disagree.
```

## Output Readout Baselines

```powershell
python -m scripts.run_output_readout_baselines --model gpt2-small --data data/facts.csv --language en --domains all capital --out figures/output_readout_baselines.csv
```

Expected artifacts:

```text
figures/output_readout_baselines.csv
figures/output_readout_baselines.png
figures/output_readout_baselines_best_by_domain.png
```

Key expected result:

```text
On capital facts, true/false, True/False, and correct/incorrect are almost always predicted as true.
On capital facts, yes/no is almost always predicted as no.
Accuracy remains around 0.5, so the output readout is not reliable for GPT-2-small.
```

## Supplementary Causal Test: Capital Recall Patching

```powershell
python -m scripts.run_activation_patching --model gpt2-small --out figures/activation_patching_capital_recall.csv --components resid_post,attn_out,mlp_out
```

Expected artifacts:

```text
figures/activation_patching_capital_recall.csv
figures/activation_patching_capital_recall.png
```

Important limitation:

```text
This is a capital recall patching experiment, not direct true/false verification patching.
```

Key expected result:

```text
resid_post layer 11 mean_recovery around 1.0
```

## Direct Causal Test: Truth Verification Patching

```powershell
python -m scripts.run_truth_verification_patching --model gpt2-small --data data/facts.csv --language en --domain capital --out figures/truth_verification_patching_resid.csv --details-out figures/truth_verification_patching_details.csv
```

Expected artifacts:

```text
figures/truth_verification_patching_resid.csv
figures/truth_verification_patching_details.csv
figures/truth_verification_patching_resid.png
figures/truth_verification_patching_resid_logit_shift.png
figures/truth_verification_patching_resid_control_shift.png
```

Key expected result:

```text
matched resid_post layer 11 mean_recovery around 1.0
matched resid_post layer 11 mean_abs_logit_shift around 0.076
shuffled resid_post layer 11 mean_recovery around 0.178
about 72.4% of denominator magnitudes are below 0.05
```

## Steering: Held-out Probe Direction

```powershell
python -m scripts.run_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --seed 42 --alphas -8 -4 -2 -1 0 1 2 4 8 --out figures/steering_capital_probe_layer8.csv
```

Expected artifacts:

```text
figures/steering_capital_probe_layer8.csv
figures/steering_capital_probe_layer8.png
figures/steering_capital_probe_layer8_accuracy.png
figures/steering_capital_probe_layer8_probe_accuracy.png
```

Key expected result:

```text
The script uses group split: 106 train rows and 46 test rows.
The probe threshold is fit on the train split.
alpha=0 held-out probe-threshold accuracy is around 0.826.
true/false logit-sign accuracy remains 0.500 across the alpha sweep.
This is an early diagnostic on original capital layer 8, not strong causal evidence for the balanced layer 6 signal.
```

## Improve Diagnostic: Oracle Conditional Steering

```powershell
python -m scripts.run_oracle_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --seed 42 --alphas 0 0.5 1 2 4 8 --out figures/oracle_steering_capital_probe_layer8.csv
```

Expected artifacts:

```text
figures/oracle_steering_capital_probe_layer8.csv
figures/oracle_steering_capital_probe_layer8.png
figures/oracle_steering_capital_probe_layer8_margins.png
```

Key expected result:

```text
probe-threshold accuracy improves from 0.826 to 1.000 under oracle labels
logit-sign accuracy remains 0.500
This is an oracle diagnostic, not a deployable improve result.
```

## Improve Diagnostic: Balanced Completion-Margin Steering

```powershell
python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --position-mode prompt-final-only --bootstrap-samples 2000 --alphas -4 -2 -1 0 1 2 4 --out-details figures/completion_margin_steering_position_prompt_final_details.csv --out-summary figures/completion_margin_steering_position_prompt_final_summary.csv
```

Expected artifacts:

```text
figures/completion_margin_steering_position_prompt_final_details.csv
figures/completion_margin_steering_position_prompt_final_summary.csv
figures/completion_margin_steering_position_prompt_final_summary.png
figures/completion_margin_steering_position_prompt_final_summary_pairwise_accuracy.png
```

Key expected result:

```text
learned_probe alpha=+4 shifts held-out avg-token completion margin by about +0.135
learned_probe alpha=-4 shifts held-out avg-token completion margin by about -0.130
random_direction alpha=+4 shifts held-out avg-token completion margin by about -0.030
label_permutation alpha=+4 shifts held-out avg-token completion margin by about -0.022
held-out pairwise avg-token preference accuracy remains 0.625 across the sweep
held-out block exact accuracy remains 0.250 across the sweep
This is weak behavioral influence on completion margin, not stable behavioral improvement.
```

## Completion Steering Diagnostics

```powershell
python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --position-mode all --bootstrap-samples 2000 --alphas -4 -2 -1 0 1 2 4 --out-details figures/completion_margin_steering_details.csv --out-summary figures/completion_margin_steering_summary.csv
python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --position-mode completion-internal-only --bootstrap-samples 1000 --alphas -4 0 4 --out-details figures/completion_margin_steering_position_completion_internal_details.csv --out-summary figures/completion_margin_steering_position_completion_internal_summary.csv
python -m scripts.run_completion_margin_null_distribution --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --position-mode prompt-final-only --alpha 4 --random-directions 50 --permutation-directions 20 --out-details figures/completion_margin_steering_null_distribution.csv --out-summary figures/completion_margin_steering_null_summary.csv
python -m scripts.run_repeated_split_completion_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seeds 0 1 2 3 4 5 6 7 8 9 --alpha 4 --random-directions 10 --permutation-directions 5 --position-mode prompt-final-only --out-details figures/repeated_split_completion_steering_details.csv --out-summary figures/repeated_split_completion_steering_summary.csv
python -m scripts.run_ambiguous_fact_sensitivity --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --alpha 4 --position-mode prompt-final-only --out-data data/capital_balanced_no_ambiguous.csv --out-details figures/ambiguous_fact_sensitivity_details.csv --out-summary figures/ambiguous_fact_sensitivity_summary.csv
python -m scripts.run_candidate_rank_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --alpha 4 --position-mode prompt-final-only --out-details figures/candidate_rank_steering_details.csv --out-summary figures/candidate_rank_steering_summary.csv
python -m scripts.analyze_completion_steering_diagnostics --details figures/completion_margin_steering_details.csv --seed 42 --bootstrap-samples 5000 --out-decomposition figures/completion_margin_steering_decomposition.csv --out-paired figures/completion_margin_steering_paired_bootstrap.csv
python -m scripts.analyze_completion_steering_diagnostics --details figures/completion_margin_steering_position_prompt_final_details.csv --seed 42 --bootstrap-samples 3000 --out-decomposition figures/completion_margin_steering_position_prompt_final_decomposition.csv --out-paired figures/completion_margin_steering_position_prompt_final_paired_bootstrap.csv
python -m scripts.analyze_completion_steering_diagnostics --details figures/completion_margin_steering_position_completion_internal_details.csv --seed 42 --bootstrap-samples 3000 --out-decomposition figures/completion_margin_steering_position_completion_internal_decomposition.csv --out-paired figures/completion_margin_steering_position_completion_internal_paired_bootstrap.csv
python -m scripts.run_unembedding_projection_baseline --model gpt2-small --data data/capital_balanced.csv --details figures/completion_margin_steering_details.csv --language en --domain capital_balanced --layer 6 --seed 42 --out-details figures/unembedding_projection_baseline_details.csv --out-summary figures/unembedding_projection_baseline_summary.csv
```

Expected artifacts:

```text
figures/completion_margin_steering_decomposition.csv
figures/completion_margin_steering_decomposition.png
figures/completion_margin_steering_paired_bootstrap.csv
figures/completion_margin_steering_paired_bootstrap.png
figures/completion_margin_steering_position_prompt_final_details.csv
figures/completion_margin_steering_position_prompt_final_summary.csv
figures/completion_margin_steering_position_prompt_final_decomposition.csv
figures/completion_margin_steering_position_prompt_final_paired_bootstrap.csv
figures/completion_margin_steering_position_completion_internal_details.csv
figures/completion_margin_steering_position_completion_internal_summary.csv
figures/completion_margin_steering_position_completion_internal_decomposition.csv
figures/completion_margin_steering_position_completion_internal_paired_bootstrap.csv
figures/completion_margin_steering_position_comparison.png
figures/completion_margin_steering_null_distribution.csv
figures/completion_margin_steering_null_summary.csv
figures/completion_margin_steering_null_distribution.png
figures/repeated_split_completion_steering_details.csv
figures/repeated_split_completion_steering_summary.csv
figures/repeated_split_completion_steering_summary.png
data/capital_balanced_no_ambiguous.csv
figures/ambiguous_fact_sensitivity_details.csv
figures/ambiguous_fact_sensitivity_summary.csv
figures/ambiguous_fact_sensitivity_summary_auc.png
figures/ambiguous_fact_sensitivity_summary_steering.png
figures/candidate_rank_steering_details.csv
figures/candidate_rank_steering_summary.csv
figures/candidate_rank_steering_summary.png
figures/unembedding_projection_baseline_details.csv
figures/unembedding_projection_baseline_details.png
figures/unembedding_projection_baseline_summary.csv
figures/unembedding_projection_baseline_summary.png
```

Key expected result:

```text
learned_probe alpha=+4 raises correct avg-token logprob by about +0.280 and false avg-token logprob by about +0.147
held-out sign_flip_total remains 0
learned minus random delta-margin paired CI is about [0.094, 0.239]
learned minus label-permutation delta-margin paired CI is about [0.092, 0.233]
learned minus random slope paired CI is about [0.024, 0.059]
learned minus label-permutation slope paired CI is about [0.023, 0.058]
prompt-final-only learned alpha=+4 shifts held-out avg-token completion margin by about +0.135
completion-internal-only learned alpha=+4 shifts held-out avg-token completion margin by about -0.002
completion-internal-only learned-control paired CIs cross zero
prompt-final-only learned effect is about +0.135, above the sampled random-direction 97.5 percentile around +0.091
prompt-final-only learned effect is also above the sampled label-permutation 97.5 percentile around +0.088
empirical upper-tail p-values are about 0.020 for random directions and 0.048 for label-permutation directions
repeated split steering has positive learned delta in 10/10 splits
repeated split learned delta has mean about +0.116, standard deviation about 0.023, and range about +0.085 to +0.150
aggregate learned-minus-random mean is about +0.125
aggregate learned-minus-permutation mean is about +0.119
repeated split baseline pairwise accuracy is about 0.700 and steered pairwise accuracy is about 0.725
repeated split learned steering has 6 wrong-to-correct flips and 0 correct-to-wrong flips
after removing three ambiguous capital blocks and re-splitting, the learned steering effect is qualitatively preserved: learned delta about +0.120
candidate-set rank steering improves mean correct rank from about 15.04 to 14.13, but top-1 accuracy remains low at about 0.125
static unembedding projection has low fit for learned alpha=+4 on held-out examples: corr squared about 0.035
```

## Ablation: Probe Direction Removal

```powershell
python -m scripts.run_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --seed 42 --out figures/ablation_capital_probe_layer8.csv
```

Expected artifacts:

```text
figures/ablation_capital_probe_layer8.csv
figures/ablation_capital_probe_layer8.png
figures/ablation_capital_probe_layer8_score_gap.png
```

Key expected result:

```text
fixed_direction_score_gap decreases from about +0.573 to 0 at strength=1.0
retrained probe AUC remains above 0.94 after one-direction ablation
```

## Ablation: Lexically Balanced Probe Direction Removal

```powershell
python -m scripts.run_ablation --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --direction-method probe --seed 42 --out figures/ablation_capital_balanced_layer6.csv
```

Expected artifacts:

```text
figures/ablation_capital_balanced_layer6.csv
figures/ablation_capital_balanced_layer6.png
figures/ablation_capital_balanced_layer6_score_gap.png
```

Key expected result:

```text
balanced fixed_direction_score_gap decreases from about +0.174 to 0 at strength=1.0
balanced retrained probe AUC remains around 0.786 after one-direction ablation
```

## Ablation: Iterative Direction Removal

```powershell
python -m scripts.run_iterative_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --seed 42 --max-directions 16 --out figures/iterative_ablation_capital_layer8.csv
```

Expected artifacts:

```text
figures/iterative_ablation_capital_layer8.csv
figures/iterative_ablation_capital_layer8.png
```

Key expected result:

```text
learned iterative ablation: AUC drops from about 0.953 to 0.807 after 16 directions
random direction control: AUC remains around 0.951 after 16 directions
label-permutation control: AUC remains around 0.953 after 16 directions
```

## Ablation: Lexically Balanced Iterative Direction Removal

```powershell
python -m scripts.run_iterative_ablation --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --max-directions 16 --out figures/iterative_ablation_capital_balanced_layer6.csv
```

Expected artifacts:

```text
figures/iterative_ablation_capital_balanced_layer6.csv
figures/iterative_ablation_capital_balanced_layer6.png
```

Key expected result:

```text
balanced learned iterative ablation: AUC drops from about 0.809 to 0.726 after 16 directions
balanced random direction control: AUC remains around 0.793 after 16 directions
balanced label-permutation control: AUC remains around 0.783 after 16 directions
The learned-vs-random gap is from one split only and should not be read as a stable effect without paired bootstrap or repeated splits.
```

## Plotting and Summary

```powershell
python -m scripts.plot_results --probe figures/probe_capital_answer.csv --probe-sweep figures/probe_sweep.csv --probe-seeds figures/probe_seed_sensitivity_capital.csv --readout figures/output_readout_baselines.csv --surface figures/surface_baselines.csv --domain-transfer figures/domain_transfer_layer8.csv --domain-cosine figures/domain_direction_cosine_layer8.csv --steering figures/steering_capital_probe_layer8.csv --oracle-steering figures/oracle_steering_capital_probe_layer8.csv --patching figures/activation_patching_capital_recall.csv --truth-patching figures/truth_verification_patching_resid.csv --ablation figures/ablation_capital_probe_layer8.csv --iterative-ablation figures/iterative_ablation_capital_layer8.csv --completion-steering figures/completion_margin_steering_position_prompt_final_summary.csv --completion-steering-null figures/completion_margin_steering_null_distribution.csv
python -m scripts.plot_results --probe figures/probe_capital_balanced.csv --probe-seeds figures/probe_seed_sensitivity_capital_balanced.csv --surface figures/surface_baselines_capital_balanced.csv --ablation figures/ablation_capital_balanced_layer6.csv --iterative-ablation figures/iterative_ablation_capital_balanced_layer6.csv --knowledge-summary figures/capital_knowledge_margin_summary.csv --knowledge-details figures/capital_knowledge_margin_details.csv --completion-steering figures/completion_margin_steering_position_prompt_final_summary.csv --completion-steering-null figures/completion_margin_steering_null_distribution.csv
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.validate_project
python -m compileall scripts src
```

Expected artifacts:

```text
reports/final_report.md
reports/results_summary.md
```

Expected validation result: `scripts.validate_project` reports required files, report images, core CSV columns, and the balanced dataset shape as valid; `compileall` completes without syntax errors. Exact numerical values should be checked in `reports/results_summary.md`; this checklist only records commands, output files, and coarse sanity checks.
