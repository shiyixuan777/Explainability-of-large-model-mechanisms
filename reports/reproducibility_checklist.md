# Reproducibility Checklist

This checklist records the commands and expected artifacts needed to reproduce the current Markdown report.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.check_env
```

Expected result: required packages are marked `OK`.

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
capital + answer prompt: best AUC around 0.953
mixed-domain settings: clearly weaker than capital
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
PC1 explained variance around 0.620
PC2 explained variance around 0.117
The 2D PCA projection does not cleanly separate true/false.
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
layer 8 test accuracy around 0.826
8 misclassified test examples
```

## Supplementary Causal Test: Capital Recall Patching

```powershell
python -m scripts.run_activation_patching --model gpt2-small --out figures/activation_patching_capital_recall.csv --components resid_post,attn_out,mlp_out
python -m scripts.plot_results --patching figures/activation_patching_capital_recall.csv
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

## Steering: Held-out Probe Direction

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
The script uses group split: 106 train rows and 46 test rows.
The probe threshold is fit on the train split.
alpha=0 held-out probe-threshold accuracy is around 0.826.
true/false logit-sign accuracy remains 0.500 across the alpha sweep.
```

## Ablation: Probe Direction Removal

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
retrained probe AUC remains above 0.94 after ablation
```

## Report and Validation

```powershell
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.prepare_submission
python -m scripts.validate_project
python -m compileall scripts src
```

Expected artifacts:

```text
reports/final_report.md
reports/results_summary.md
reports/submission_manifest.md
```
