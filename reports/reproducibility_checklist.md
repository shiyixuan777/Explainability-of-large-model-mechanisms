# Reproducibility Checklist

This checklist records the commands and artifacts needed to reproduce the current project results.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.check_env
```

Expected result: all required packages are marked `OK`.

## Dataset

```powershell
python -m scripts.build_dataset
```

Expected artifact:

```text
data/facts.csv
```

Expected summary:

```text
Rows: 528
label 0: 264
label 1: 264
```

## Locate: Probe Sweep

```powershell
python -m scripts.run_probe_sweep --model gpt2-small --data data/facts.csv --out figures/probe_sweep.csv
python -m scripts.plot_results --probe-sweep figures/probe_sweep.csv
```

Expected artifacts:

```text
figures/probe_sweep.csv
figures/probe_sweep_summary.png
```

Key expected result:

```text
capital + answer prompt: best AUC around 0.95
```

## Locate: Focused Capital Probe

```powershell
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_answer.csv
python -m scripts.plot_results --probe figures/probe_capital_answer.csv
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
The figure gives a qualitative two-dimensional view of layer-8 capital activations.
It does not need to separate true/false perfectly, because PCA keeps high-variance
directions rather than the supervised probe direction.
```

## Locate: Error Analysis

```powershell
python -m scripts.run_error_analysis --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/error_analysis_capital_layer8.csv
```

Expected artifacts:

```text
figures/error_analysis_capital_layer8.csv
figures/error_analysis_capital_layer8_errors.csv
```

Key expected result:

```text
layer 8 test accuracy around 0.826 and AUC around 0.953
8 misclassified test examples under the fixed 0.5 threshold
```

## Causal Locate: Activation Patching

```powershell
python -m scripts.run_activation_patching --model gpt2-small --out figures/activation_patching_capital_recall.csv --components resid_post,attn_out,mlp_out
python -m scripts.plot_results --patching figures/activation_patching_capital_recall.csv
```

Expected artifacts:

```text
figures/activation_patching_capital_recall.csv
figures/activation_patching_capital_recall.png
```

Key expected result:

```text
resid_post layer 11 mean_recovery around 1.0
attn_out layer 11 shows strong recovery signal
mlp_out is weaker
```

## Steer: Probe-Direction Steering

```powershell
python -m scripts.run_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --alphas -8 -4 -2 -1 0 1 2 4 8 --out figures/steering_capital_probe_layer8.csv
python -m scripts.plot_results --steering figures/steering_capital_probe_layer8.csv
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
Probe score moves with alpha, but true/false logit-sign accuracy remains around 0.5.
```

## Improve/Ablate: Probe-Direction Ablation

```powershell
python -m scripts.run_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --out figures/ablation_capital_probe_layer8.csv
python -m scripts.plot_results --ablation figures/ablation_capital_probe_layer8.csv
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
```

## Report

Main report draft:

```text
reports/project_report.md
```

Machine-generated result summary:

```powershell
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
```

Expected artifact:

```text
reports/results_summary.md
```

Report should mention:

- Dataset construction
- Linear probe localization
- Domain-wise truth-direction consistency
- Activation patching
- Probe-direction steering
- Probe-direction ablation
- Limitations and next steps
