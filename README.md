# GPT-2-small 事实配对标签信号的线性可分性、补全兼容度与干预边界

本仓库研究 GPT-2-small 在人工事实验证数据集上的 final-token residual activation 是否包含可线性解码的事实配对标签相关信号，并进一步检查这种信号是否稳定、是否与输出行为相关、以及沿该方向干预会产生什么下游效应。

最终报告：

```text
reports/final_report.md
```

## 核心结论

本项目没有证明 GPT-2-small 存在跨领域稳定、可直接控制输出的全局 truth direction。更准确的结论是：

> 原始 capital 数据集上 residual probe 表现很强，但 bag-of-words baseline 也几乎同样可分，说明原始结果受到明显词汇伪线索影响。构造词汇平衡 capital 数据集后，BOW baseline 降为随机，而 residual probe 仍保留约 0.8 AUC 的 fact-pair label signal。completion-margin steering 显示 balanced layer 6 verification-associated direction 能小幅移动 avg-token completion margin；在当前主 split 上，learned effect 超过全部已采样的 50 条 random directions 和 20 条 label-permutation directions。10 个 repeated group splits 中，learned shift 均为正，learned delta 均值为 +0.116，范围为 +0.085 到 +0.150；pairwise accuracy 只从 0.700 到 0.725。争议事实 sensitivity 在删除 3 个 block 后重新划分，信号和 steering effect 定性保留；candidate-set rank 检查只显示正确首都 rank 小幅改善，top-1 accuracy 仍很低。因此它说明该方向改变的是 correct-vs-selected-wrong completion scoring preference，不是稳定事实纠错机制，也不是完整机制定位。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.check_env
python -m scripts.build_dataset
python -m scripts.build_balanced_capital_dataset
```

如果 Windows 当前 PATH 中的 `python` 不可用，建议直接使用虚拟环境解释器运行同一命令，例如：

```powershell
.\.venv\Scripts\python.exe -m scripts.check_env
.\.venv\Scripts\python.exe -m scripts.validate_project
```

首次运行 `gpt2-small` 会从 Hugging Face 下载模型权重；之后会使用本地缓存。

## 完整复现实验

```powershell
python -m scripts.run_probe_sweep --model gpt2-small --data data/facts.csv --out figures/probe_sweep.csv
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_answer.csv
python -m scripts.run_probe_seed_sensitivity --model gpt2-small --data data/facts.csv --language en --domain capital --layers 5 8 10 11 --seeds 0 1 2 3 4 42 --out figures/probe_seed_sensitivity_capital.csv
python -m scripts.run_surface_baselines --data data/facts.csv --language en --domains all capital --out figures/surface_baselines.csv
python -m scripts.run_probe --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_balanced.csv
python -m scripts.run_probe_seed_sensitivity --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layers 4 6 8 10 11 --seeds 0 1 2 3 4 42 --out figures/probe_seed_sensitivity_capital_balanced.csv
python -m scripts.run_surface_baselines --data data/capital_balanced.csv --language en --domains capital_balanced --out figures/surface_baselines_capital_balanced.csv
python -m scripts.run_capital_knowledge_margin --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --out-details figures/capital_knowledge_margin_details.csv --out-summary figures/capital_knowledge_margin_summary.csv
python -m scripts.run_activation_pca --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/pca_capital_layer8.csv
python -m scripts.run_error_analysis --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/error_analysis_capital_layer8.csv
python -m scripts.run_domain_consistency --model gpt2-small --data data/facts.csv --language en --layer 8 --out-transfer figures/domain_transfer_layer8.csv --out-cosine figures/domain_direction_cosine_layer8.csv
python -m scripts.run_output_readout_baselines --model gpt2-small --data data/facts.csv --language en --domains all capital --out figures/output_readout_baselines.csv
python -m scripts.run_activation_patching --model gpt2-small --out figures/activation_patching_capital_recall.csv --components resid_post,attn_out,mlp_out
python -m scripts.run_truth_verification_patching --model gpt2-small --data data/facts.csv --language en --domain capital --out figures/truth_verification_patching_resid.csv --details-out figures/truth_verification_patching_details.csv
python -m scripts.run_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --alphas -8 -4 -2 -1 0 1 2 4 8 --out figures/steering_capital_probe_layer8.csv
python -m scripts.run_oracle_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --alphas 0 0.5 1 2 4 8 --out figures/oracle_steering_capital_probe_layer8.csv
python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --bootstrap-samples 2000 --alphas -4 -2 -1 0 1 2 4 --out-details figures/completion_margin_steering_details.csv --out-summary figures/completion_margin_steering_summary.csv
python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --position-mode prompt-final-only --bootstrap-samples 1000 --alphas -4 0 4 --out-details figures/completion_margin_steering_position_prompt_final_details.csv --out-summary figures/completion_margin_steering_position_prompt_final_summary.csv
python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --position-mode completion-internal-only --bootstrap-samples 1000 --alphas -4 0 4 --out-details figures/completion_margin_steering_position_completion_internal_details.csv --out-summary figures/completion_margin_steering_position_completion_internal_summary.csv
python -m scripts.run_completion_margin_null_distribution --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --position-mode prompt-final-only --alpha 4 --random-directions 50 --permutation-directions 20 --out-details figures/completion_margin_steering_null_distribution.csv --out-summary figures/completion_margin_steering_null_summary.csv
python -m scripts.run_repeated_split_completion_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seeds 0 1 2 3 4 5 6 7 8 9 --alpha 4 --random-directions 10 --permutation-directions 5 --position-mode prompt-final-only --out-details figures/repeated_split_completion_steering_details.csv --out-summary figures/repeated_split_completion_steering_summary.csv
python -m scripts.run_ambiguous_fact_sensitivity --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --alpha 4 --position-mode prompt-final-only --out-data data/capital_balanced_no_ambiguous.csv --out-details figures/ambiguous_fact_sensitivity_details.csv --out-summary figures/ambiguous_fact_sensitivity_summary.csv
python -m scripts.run_candidate_rank_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --alpha 4 --position-mode prompt-final-only --out-details figures/candidate_rank_steering_details.csv --out-summary figures/candidate_rank_steering_summary.csv
python -m scripts.analyze_completion_steering_diagnostics --details figures/completion_margin_steering_details.csv --bootstrap-samples 5000 --out-decomposition figures/completion_margin_steering_decomposition.csv --out-paired figures/completion_margin_steering_paired_bootstrap.csv
python -m scripts.analyze_completion_steering_diagnostics --details figures/completion_margin_steering_position_prompt_final_details.csv --bootstrap-samples 3000 --out-decomposition figures/completion_margin_steering_position_prompt_final_decomposition.csv --out-paired figures/completion_margin_steering_position_prompt_final_paired_bootstrap.csv
python -m scripts.analyze_completion_steering_diagnostics --details figures/completion_margin_steering_position_completion_internal_details.csv --bootstrap-samples 3000 --out-decomposition figures/completion_margin_steering_position_completion_internal_decomposition.csv --out-paired figures/completion_margin_steering_position_completion_internal_paired_bootstrap.csv
python -m scripts.run_unembedding_projection_baseline --model gpt2-small --data data/capital_balanced.csv --details figures/completion_margin_steering_details.csv --language en --domain capital_balanced --layer 6 --out-details figures/unembedding_projection_baseline_details.csv --out-summary figures/unembedding_projection_baseline_summary.csv
python -m scripts.run_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --out figures/ablation_capital_probe_layer8.csv
python -m scripts.run_iterative_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --max-directions 16 --out figures/iterative_ablation_capital_layer8.csv
python -m scripts.run_ablation --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --direction-method probe --out figures/ablation_capital_balanced_layer6.csv
python -m scripts.run_iterative_ablation --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --max-directions 16 --out figures/iterative_ablation_capital_balanced_layer6.csv
python -m scripts.plot_results --probe figures/probe_capital_answer.csv --probe-sweep figures/probe_sweep.csv --probe-seeds figures/probe_seed_sensitivity_capital.csv --readout figures/output_readout_baselines.csv --surface figures/surface_baselines.csv --domain-transfer figures/domain_transfer_layer8.csv --domain-cosine figures/domain_direction_cosine_layer8.csv --steering figures/steering_capital_probe_layer8.csv --oracle-steering figures/oracle_steering_capital_probe_layer8.csv --patching figures/activation_patching_capital_recall.csv --truth-patching figures/truth_verification_patching_resid.csv --ablation figures/ablation_capital_probe_layer8.csv --iterative-ablation figures/iterative_ablation_capital_layer8.csv --completion-steering figures/completion_margin_steering_summary.csv --completion-steering-decomposition figures/completion_margin_steering_decomposition.csv --completion-steering-paired figures/completion_margin_steering_paired_bootstrap.csv --completion-steering-null figures/completion_margin_steering_null_distribution.csv --unembedding-projection-summary figures/unembedding_projection_baseline_summary.csv --unembedding-projection-details figures/unembedding_projection_baseline_details.csv
python -m scripts.plot_results --probe figures/probe_capital_balanced.csv --probe-seeds figures/probe_seed_sensitivity_capital_balanced.csv --surface figures/surface_baselines_capital_balanced.csv --ablation figures/ablation_capital_balanced_layer6.csv --iterative-ablation figures/iterative_ablation_capital_balanced_layer6.csv --knowledge-summary figures/capital_knowledge_margin_summary.csv --knowledge-details figures/capital_knowledge_margin_details.csv --completion-steering figures/completion_margin_steering_summary.csv --completion-steering-decomposition figures/completion_margin_steering_decomposition.csv --completion-steering-paired figures/completion_margin_steering_paired_bootstrap.csv --completion-steering-null figures/completion_margin_steering_null_distribution.csv --unembedding-projection-summary figures/unembedding_projection_baseline_summary.csv --unembedding-projection-details figures/unembedding_projection_baseline_details.csv
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.prepare_submission
python -m scripts.validate_project
```

## 主要结果文件

- 原始数据集：`data/facts.csv`
- 词汇平衡 capital 数据集：`data/capital_balanced.csv`
- 原始 capital probe：`figures/probe_capital_answer.csv`
- 原始 capital 多 seed：`figures/probe_seed_sensitivity_capital.csv`
- 词汇平衡 probe：`figures/probe_capital_balanced.csv`
- 词汇平衡多 seed：`figures/probe_seed_sensitivity_capital_balanced.csv`
- surface baselines：`figures/surface_baselines.csv`, `figures/surface_baselines_capital_balanced.csv`
- completion margin baseline：`figures/capital_knowledge_margin_summary.csv`
- completion-margin steering：`figures/completion_margin_steering_summary.csv`, `figures/completion_margin_steering_decomposition.csv`, `figures/completion_margin_steering_paired_bootstrap.csv`
- completion-margin steering null distribution：`figures/completion_margin_steering_null_distribution.csv`, `figures/completion_margin_steering_null_summary.csv`
- repeated split steering：`figures/repeated_split_completion_steering_summary.csv`
- ambiguous-fact sensitivity：`figures/ambiguous_fact_sensitivity_summary.csv`
- candidate-set rank steering：`figures/candidate_rank_steering_summary.csv`
- position decomposition：`figures/completion_margin_steering_position_prompt_final_summary.csv`, `figures/completion_margin_steering_position_completion_internal_summary.csv`
- unembedding projection baseline：`figures/unembedding_projection_baseline_summary.csv`
- output readout baseline：`figures/output_readout_baselines.csv`
- domain transfer/cosine：`figures/domain_transfer_layer8.csv`, `figures/domain_direction_cosine_layer8.csv`
- patching：`figures/truth_verification_patching_resid.csv`
- steering：`figures/steering_capital_probe_layer8.csv`
- oracle steering：`figures/oracle_steering_capital_probe_layer8.csv`
- ablation：`figures/ablation_capital_probe_layer8.csv`, `figures/iterative_ablation_capital_layer8.csv`
- balanced ablation：`figures/ablation_capital_balanced_layer6.csv`, `figures/iterative_ablation_capital_balanced_layer6.csv`
- 自动结果摘要：`reports/results_summary.md`

## 目录结构

```text
.
├── data/           # 事实验证数据集
├── figures/        # CSV 与 PNG 结果
├── reports/        # 报告、摘要和复现说明
├── scripts/        # 可运行实验脚本
└── src/            # 数据、hook、steering、probe 工具函数
```
