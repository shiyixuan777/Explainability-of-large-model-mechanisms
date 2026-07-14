# GPT-2-small Truth/False Readability and Intervention Limits

本仓库是“大模型机制可解释性”课程项目代码库。项目围绕 GPT-2-small 的事实真假判断展开，核心结论不是“找到一个可直接控制输出的全局 truth button”，而是：

> GPT-2-small 在结构化 capital fact verification 中存在强线性可读的 true/false 信息；但该信息不构成跨领域稳定、可通过 naive global steering 直接改善输出的单一 truth direction，更像分布在冗余子空间中的可读信号。

正式报告为 Markdown 文件：

```text
reports/final_report.md
```

## Research Questions

1. GPT-2-small 的 residual stream 中是否存在可由 linear probe 读取的 true/false 信息？
2. 这种可读性是否跨领域、跨 prompt 稳定？
3. true/false verification residual patching 是否能提供更直接的因果定位证据？
4. probe direction steering 是否能改变内部 probe score，并进一步改善输出行为？
5. ablation 能否说明 true/false 信息是否局限于单一方向？

## Method Overview

- **Locate**: layer-wise linear probe, domain/prompt sweep, PCA, error analysis.
- **Patching**: related capital recall patching and direct true/false verification residual patching.
- **Steering and ablation**: naive global steering, oracle conditional steering, and probe-direction ablation.
- **Reproduction target**: partial reproduction of Bao et al. 2025; Marks and Tegmark 2023 as background.

## Key Results

- Mixed-domain truth direction is weak: the best all-domain separability is only moderate.
- In capital fact verification, layer-8 residual stream reaches AUC 0.953 and layer-10 accuracy reaches 0.870.
- PCA does not cleanly separate true/false in two dimensions, so the signal is not simply the highest-variance direction.
- Truth verification residual patching shows late-layer recovery, but the average absolute logit shift is small.
- Naive probe-direction steering moves the internal probe score, but logit-sign true/false accuracy stays at 0.500.
- Oracle conditional steering improves held-out probe-threshold accuracy from 0.826 to 1.000, but still does not improve logit-sign output accuracy.
- Direction ablation removes the fixed-direction score gap, but a retrained probe still recovers AUC above 0.94, suggesting redundant subspace structure.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.check_env
python -m scripts.build_dataset
```

The first `gpt2-small` run may download weights from Hugging Face. Later runs use the local cache.

## Full Reproduction Pipeline

```powershell
python -m scripts.run_probe_sweep --model gpt2-small --data data/facts.csv --out figures/probe_sweep.csv
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_answer.csv
python -m scripts.run_activation_pca --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/pca_capital_layer8.csv
python -m scripts.run_error_analysis --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/error_analysis_capital_layer8.csv
python -m scripts.run_activation_patching --model gpt2-small --out figures/activation_patching_capital_recall.csv --components resid_post,attn_out,mlp_out
python -m scripts.run_truth_verification_patching --model gpt2-small --data data/facts.csv --language en --domain capital --out figures/truth_verification_patching_resid.csv
python -m scripts.run_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --alphas -8 -4 -2 -1 0 1 2 4 8 --out figures/steering_capital_probe_layer8.csv
python -m scripts.run_oracle_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --alphas 0 0.5 1 2 4 8 --out figures/oracle_steering_capital_probe_layer8.csv
python -m scripts.run_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --out figures/ablation_capital_probe_layer8.csv
python -m scripts.plot_results --probe figures/probe_capital_answer.csv --probe-sweep figures/probe_sweep.csv --steering figures/steering_capital_probe_layer8.csv --oracle-steering figures/oracle_steering_capital_probe_layer8.csv --patching figures/activation_patching_capital_recall.csv --truth-patching figures/truth_verification_patching_resid.csv --ablation figures/ablation_capital_probe_layer8.csv
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.prepare_submission
python -m scripts.validate_project
```

## Repository Structure

```text
.
├── configs/        # experiment configuration
├── data/           # fact-verification dataset
├── figures/        # CSV and PNG outputs
├── reports/        # Markdown report and checklists
├── scripts/        # runnable experiment scripts
└── src/            # reusable model/data/intervention helpers
```

## Main Artifacts

- Dataset: `data/facts.csv`
- Probe sweep: `figures/probe_sweep.csv`, `figures/probe_sweep_summary.png`
- Focused capital probe: `figures/probe_capital_answer.csv`, `figures/probe_capital_answer.png`
- PCA visualization: `figures/pca_capital_layer8.csv`, `figures/pca_capital_layer8.png`
- Error analysis: `figures/error_analysis_capital_layer8.csv`, `figures/error_analysis_capital_layer8_errors.csv`
- Capital recall patching: `figures/activation_patching_capital_recall.csv`, `figures/activation_patching_capital_recall.png`
- Truth verification patching: `figures/truth_verification_patching_resid.csv`, `figures/truth_verification_patching_resid*.png`
- Held-out probe-direction steering: `figures/steering_capital_probe_layer8.csv`, `figures/steering_capital_probe_layer8*.png`
- Oracle conditional steering: `figures/oracle_steering_capital_probe_layer8.csv`, `figures/oracle_steering_capital_probe_layer8*.png`
- Probe-direction ablation: `figures/ablation_capital_probe_layer8.csv`, `figures/ablation_capital_probe_layer8*.png`
- Final report: `reports/final_report.md`
- Reproducibility checklist: `reports/reproducibility_checklist.md`
- Submission manifest: `reports/submission_manifest.md`

## Notes

The default dataset contains 528 English true/false fact-verification examples, balanced as 264 true and 264 false, covering seven domains: `capital`, `continent`, `element_symbol`, `book_author`, `landmark_country`, `science`, and `math`.

The default model is `gpt2-small`, chosen because TransformerLens supports its hook points reliably. Larger models such as Qwen2.5 can be explored later, but that would require additional hook compatibility checks.
