# 复现检查清单

本文档记录复现当前 Markdown 报告所需的命令和预期产物。它是实验运行手册，不承担主要结果解释；精确数值表集中在 `reports/results_summary.md`。

## 环境

已测试环境：

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

复现 GPT-2-small 小模型实验不需要 GPU，但 CPU 运行会更慢。第一次加载模型时会从 Hugging Face 下载 `gpt2-small`，需要网络连接和本地模型缓存空间。在已测试的 CPU 环境中，词汇平衡数据上的提示词末位置干预命令是快速复现中最慢的一步，通常需要几分钟；完整运行手册包含探索性诊断和重复划分实验，因此耗时会更长。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

预期结果：依赖安装成功，并且可以在激活后的环境中运行下面的数据构建命令。

## 快速复现

以下命令复现正式报告中的核心证据链：原始数据的词汇伪线索诊断、词汇平衡后的线性读出、补全兼容度分析和提示词末位置干预。下方完整命令列表保留了辅助诊断和早期探索性实验。

```powershell
python -m scripts.build_dataset
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --seed 42 --out figures/probe_capital_answer.csv
python -m scripts.run_surface_baselines --data data/facts.csv --language en --domains all capital --seed 42 --out figures/surface_baselines.csv
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

主要基本检查：原始 capital 数据上的 BOW 方向无关 AUC 应较高；词汇平衡后的表面特征基线应接近随机；词汇平衡数据第 6 层探针 AUC 应约为 0.81；提示词末位置的学习方向干预应产生正的平均词元补全得分差变化，并大于采样随机方向和标签置乱方向对照。

## 数据集

```powershell
python -m scripts.build_dataset
python -m scripts.build_balanced_capital_dataset --out data/capital_balanced.csv
```

预期产物：

```text
data/facts.csv
data/capital_balanced.csv
```

预期摘要：

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

## 定位：探针扫描

```powershell
python -m scripts.run_probe_sweep --model gpt2-small --data data/facts.csv --seed 42 --out figures/probe_sweep.csv
```

预期产物：

```text
figures/probe_sweep.csv
figures/probe_sweep_summary.png
```

关键预期结果：

```text
capital + answer 提示词：最佳 AUC 约为 0.953
混合领域设置明显弱于 capital
```

## 定位：首都任务重点探针

```powershell
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --seed 42 --out figures/probe_capital_answer.csv
```

预期产物：

```text
figures/probe_capital_answer.csv
figures/probe_capital_answer.png
```

关键预期结果：

```text
第 8 层 AUC 约为 0.953
第 10 层准确率约为 0.870
```

## 定位：探针随机种子敏感性

```powershell
python -m scripts.run_probe_seed_sensitivity --model gpt2-small --data data/facts.csv --language en --domain capital --layers 5 8 10 11 --seeds 0 1 2 3 4 42 --out figures/probe_seed_sensitivity_capital.csv
```

预期产物：

```text
figures/probe_seed_sensitivity_capital.csv
figures/probe_seed_sensitivity_capital.png
```

关键预期结果：

```text
第 8 层在所检查随机种子上的平均 AUC 约为 0.899
第 8 层 AUC 范围约为 0.832 到 0.953
seed=42 是较强但偏乐观的划分
```

## 定位：激活 PCA

```powershell
python -m scripts.run_activation_pca --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/pca_capital_layer8.csv
```

预期产物：

```text
figures/pca_capital_layer8.csv
figures/pca_capital_layer8.png
```

关键预期结果：

```text
PC1 解释方差约为 0.620
PC2 解释方差约为 0.117
二维 PCA 投影不能清晰分离 true/false。
```

## 定位：错误分析

```powershell
python -m scripts.run_error_analysis --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --seed 42 --out figures/error_analysis_capital_layer8.csv
```

预期产物：

```text
figures/error_analysis_capital_layer8.csv
figures/error_analysis_capital_layer8_errors.csv
```

关键预期结果：

```text
第 8 层测试准确率约为 0.826
测试集中有 8 个误分类样本
```

## 定位：跨领域方向一致性

```powershell
python -m scripts.run_domain_consistency --model gpt2-small --data data/facts.csv --language en --layer 8 --seed 42 --out-transfer figures/domain_transfer_layer8.csv --out-cosine figures/domain_direction_cosine_layer8.csv
```

预期产物：

```text
figures/domain_transfer_layer8.csv
figures/domain_transfer_layer8.png
figures/domain_transfer_layer8_separability.png
figures/domain_direction_cosine_layer8.csv
figures/domain_direction_cosine_layer8.png
```

关键预期结果：

```text
平均跨领域方向余弦相似度约为 0.077
多数跨领域迁移 AUC 接近 0.5
最强跨领域迁移是 continent -> capital，AUC 约为 0.766
```

## 表面特征基线

```powershell
python -m scripts.run_surface_baselines --data data/facts.csv --language en --domains all capital --seed 42 --out figures/surface_baselines.csv
python -m scripts.run_surface_baselines --data data/capital_balanced.csv --language en --domains capital_balanced --seed 42 --out figures/surface_baselines_capital_balanced.csv
```

预期产物：

```text
figures/surface_baselines.csv
figures/surface_baselines.png
figures/surface_baselines_capital_balanced.csv
figures/surface_baselines_capital_balanced.png
```

关键预期结果：

```text
数值表面特征基线在 capital 上较弱：方向无关 AUC 约为 0.549
词袋基线暴露词汇伪线索：capital 方向无关 AUC 约为 0.933
在词汇平衡首都数据上，数值表面特征基线和词袋基线均为随机水平
```

## 词汇平衡首都探针

```powershell
python -m scripts.run_probe --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --prompt-template "Statement: {statement}`nAnswer true or false:" --seed 42 --out figures/probe_capital_balanced.csv
python -m scripts.run_probe_seed_sensitivity --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layers 4 6 8 10 11 --seeds 0 1 2 3 4 42 --out figures/probe_seed_sensitivity_capital_balanced.csv
```

预期产物：

```text
figures/probe_capital_balanced.csv
figures/probe_capital_balanced.png
figures/probe_seed_sensitivity_capital_balanced.csv
figures/probe_seed_sensitivity_capital_balanced.png
```

关键预期结果：

```text
词汇平衡首都数据的 BOW 可分性为 0.500
词汇平衡残差流探针第 6 层 AUC 约为 0.809
词汇平衡残差流探针第 8 层 AUC 约为 0.802
第 6 层跨随机种子平均 AUC 约为 0.813
第 8 层跨随机种子平均 AUC 约为 0.804
```

## 补全得分差：首都补全兼容度

```powershell
python -m scripts.run_capital_knowledge_margin --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --bootstrap-samples 2000 --out-details figures/capital_knowledge_margin_details.csv --out-summary figures/capital_knowledge_margin_summary.csv
```

预期产物：

```text
figures/capital_knowledge_margin_details.csv
figures/capital_knowledge_margin_details.png
figures/capital_knowledge_margin_summary.csv
figures/capital_knowledge_margin_summary.png
```

关键预期结果：

```text
留出测试集 completion_total AUC 约为 0.861，数据块自助法 CI 约为 0.753-0.955
留出测试集 completion_avg_token AUC 约为 0.786，数据块自助法 CI 约为 0.674-0.891
同一划分上的留出 residual_probe AUC 约为 0.809，数据块自助法 CI 约为 0.708-0.922
24 行留出测试样本的正确/错误补全词元数不同
补全基线属于探索性证据，因为总 logprob 与平均词元指标不完全一致。
```

## 输出读出基线

```powershell
python -m scripts.run_output_readout_baselines --model gpt2-small --data data/facts.csv --language en --domains all capital --out figures/output_readout_baselines.csv
```

预期产物：

```text
figures/output_readout_baselines.csv
figures/output_readout_baselines.png
figures/output_readout_baselines_best_by_domain.png
```

关键预期结果：

```text
在 capital 事实上，true/false、True/False 和 correct/incorrect 几乎总被预测为 true。
在 capital 事实上，yes/no 几乎总被预测为 no。
准确率保持在约 0.5，因此输出读出基线对 GPT-2-small 不可靠。
```

## 补充因果测试：首都召回激活修补

```powershell
python -m scripts.run_activation_patching --model gpt2-small --out figures/activation_patching_capital_recall.csv --components resid_post,attn_out,mlp_out
```

预期产物：

```text
figures/activation_patching_capital_recall.csv
figures/activation_patching_capital_recall.png
```

重要限制：

```text
这是首都召回激活修补实验，不是直接的 true/false 事实验证修补。
```

关键预期结果：

```text
`resid_post` 第 11 层 mean_recovery 约为 1.0
```

## 直接因果测试：事实验证激活修补

```powershell
python -m scripts.run_truth_verification_patching --model gpt2-small --data data/facts.csv --language en --domain capital --out figures/truth_verification_patching_resid.csv --details-out figures/truth_verification_patching_details.csv
```

预期产物：

```text
figures/truth_verification_patching_resid.csv
figures/truth_verification_patching_details.csv
figures/truth_verification_patching_resid.png
figures/truth_verification_patching_resid_logit_shift.png
figures/truth_verification_patching_resid_control_shift.png
```

关键预期结果：

```text
matched `resid_post` 第 11 层 mean_recovery 约为 1.0
匹配条件下 `resid_post` 第 11 层 mean_abs_logit_shift 约为 0.076
置乱条件下 `resid_post` 第 11 层 mean_recovery 约为 0.178
约 72.4% 的分母绝对值低于 0.05
```

## 激活干预：留出测试集探针方向

```powershell
python -m scripts.run_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --seed 42 --alphas -8 -4 -2 -1 0 1 2 4 8 --out figures/steering_capital_probe_layer8.csv
```

预期产物：

```text
figures/steering_capital_probe_layer8.csv
figures/steering_capital_probe_layer8.png
figures/steering_capital_probe_layer8_accuracy.png
figures/steering_capital_probe_layer8_probe_accuracy.png
```

关键预期结果：

```text
该脚本使用分组划分：106 行训练样本和 46 行测试样本。
探针阈值在训练划分上拟合。
alpha=0 时，留出测试集探针阈值准确率约为 0.826。
整个 alpha 扫描中，true/false logit 符号准确率保持为 0.500。
这是原始 capital 第 8 层上的早期诊断，不是词汇平衡第 6 层信号的强因果证据。
```

## 改进诊断：Oracle 条件干预

```powershell
python -m scripts.run_oracle_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --seed 42 --alphas 0 0.5 1 2 4 8 --out figures/oracle_steering_capital_probe_layer8.csv
```

预期产物：

```text
figures/oracle_steering_capital_probe_layer8.csv
figures/oracle_steering_capital_probe_layer8.png
figures/oracle_steering_capital_probe_layer8_margins.png
```

关键预期结果：

```text
使用 oracle 标签时，探针阈值准确率从 0.826 提升到 1.000
logit 符号准确率保持为 0.500
这是 oracle 诊断，不是可部署的改进结果。
```

## 改进诊断：词汇平衡补全得分差干预

```powershell
python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --position-mode prompt-final-only --bootstrap-samples 2000 --alphas -4 -2 -1 0 1 2 4 --out-details figures/completion_margin_steering_position_prompt_final_details.csv --out-summary figures/completion_margin_steering_position_prompt_final_summary.csv
```

预期产物：

```text
figures/completion_margin_steering_position_prompt_final_details.csv
figures/completion_margin_steering_position_prompt_final_summary.csv
figures/completion_margin_steering_position_prompt_final_summary.png
figures/completion_margin_steering_position_prompt_final_summary_pairwise_accuracy.png
```

关键预期结果：

```text
learned_probe 在 alpha=+4 时将留出测试集平均词元补全得分差移动约 +0.135
learned_probe 在 alpha=-4 时将留出测试集平均词元补全得分差移动约 -0.130
random_direction 在 alpha=+4 时将留出测试集平均词元补全得分差移动约 -0.030
label_permutation 在 alpha=+4 时将留出测试集平均词元补全得分差移动约 -0.022
整个扫描中，留出测试集配对平均词元偏好准确率保持为 0.625
整个扫描中，留出测试集数据块完全正确率保持为 0.250
这是对补全得分差的弱行为影响，不是稳定的行为改进。
```

## 补全干预诊断

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

预期产物：

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

关键预期结果：

```text
全位置诊断：learned_probe 在 alpha=+4 时使正确补全平均词元 logprob 上升约 +0.280，错误补全上升约 +0.147，得分差上升约 +0.133
仅提示词末位置主结果：learned_probe 在 alpha=+4 时使正确补全平均词元 logprob 上升约 +0.281，错误补全上升约 +0.146，得分差上升约 +0.135
留出测试集 sign_flip_total 保持为 0
全位置下，学习方向减随机方向的得分差变化配对 CI 约为 [0.094, 0.239]
全位置下，学习方向减标签置乱方向的得分差变化配对 CI 约为 [0.092, 0.233]
仅提示词末位置下，学习方向减随机方向的得分差变化配对 CI 约为 [0.096, 0.239]
仅提示词末位置下，学习方向减标签置乱方向的得分差变化配对 CI 约为 [0.090, 0.232]
学习方向减随机方向的斜率配对 CI 约为 [0.024, 0.059]
学习方向减标签置乱方向的斜率配对 CI 约为 [0.023, 0.058]
仅提示词末位置下，学习方向在 alpha=+4 时将留出测试集平均词元补全得分差移动约 +0.135
仅补全文本内部位置下，学习方向在 alpha=+4 时将留出测试集平均词元补全得分差移动约 -0.002
仅补全文本内部位置下，学习方向与对照方向的配对 CI 跨过 0
仅提示词末位置下，学习方向效果约为 +0.135，高于采样随机方向 97.5 分位数（约 +0.091）
仅提示词末位置下，学习方向效果也高于采样标签置乱方向 97.5 分位数（约 +0.088）
经验上尾 p 值约为：随机方向 0.020，标签置乱方向 0.048
重复划分干预中，10/10 个划分的学习方向变化均为正
重复划分中，学习方向变化均值约为 +0.116，标准差约为 0.023，范围约为 +0.085 到 +0.150
汇总后，学习方向减随机方向的均值约为 +0.125
汇总后，学习方向减标签置乱方向的均值约为 +0.119
重复划分中的基线配对准确率约为 0.700，干预后配对准确率约为 0.725
重复划分学习方向干预产生 6 次 wrong-to-correct 翻转和 0 次 correct-to-wrong 翻转
删除三个争议首都数据块并重新划分后，学习方向干预效果在定性上保留：变化约为 +0.120
候选集排名干预将正确首都平均排名从约 15.04 改善到 14.13，但 top-1 准确率仍较低，约为 0.125
静态 unembedding 投影对留出样本上 learned alpha=+4 的拟合较弱：corr squared 约为 0.035
```

## 消融：移除探针方向

```powershell
python -m scripts.run_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --seed 42 --out figures/ablation_capital_probe_layer8.csv
```

预期产物：

```text
figures/ablation_capital_probe_layer8.csv
figures/ablation_capital_probe_layer8.png
figures/ablation_capital_probe_layer8_score_gap.png
```

关键预期结果：

```text
strength=1.0 时，fixed_direction_score_gap 从约 +0.573 降到 0
单方向消融后，重新训练探针的 AUC 仍高于 0.94
```

## 消融：移除词汇平衡探针方向

```powershell
python -m scripts.run_ablation --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --direction-method probe --seed 42 --out figures/ablation_capital_balanced_layer6.csv
```

预期产物：

```text
figures/ablation_capital_balanced_layer6.csv
figures/ablation_capital_balanced_layer6.png
figures/ablation_capital_balanced_layer6_score_gap.png
```

关键预期结果：

```text
词汇平衡设置下，strength=1.0 时 fixed_direction_score_gap 从约 +0.174 降到 0
词汇平衡设置下，单方向消融后重新训练探针的 AUC 仍约为 0.786
```

## 消融：迭代移除方向

```powershell
python -m scripts.run_iterative_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --seed 42 --max-directions 16 --out figures/iterative_ablation_capital_layer8.csv
```

预期产物：

```text
figures/iterative_ablation_capital_layer8.csv
figures/iterative_ablation_capital_layer8.png
```

关键预期结果：

```text
学习方向迭代消融：移除 16 个方向后，AUC 从约 0.953 降到 0.807
随机方向对照：移除 16 个方向后，AUC 仍约为 0.951
标签置乱方向对照：移除 16 个方向后，AUC 仍约为 0.953
```

## 消融：词汇平衡迭代移除方向

```powershell
python -m scripts.run_iterative_ablation --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seed 42 --max-directions 16 --out figures/iterative_ablation_capital_balanced_layer6.csv
```

预期产物：

```text
figures/iterative_ablation_capital_balanced_layer6.csv
figures/iterative_ablation_capital_balanced_layer6.png
```

关键预期结果：

```text
词汇平衡学习方向迭代消融：移除 16 个方向后，AUC 从约 0.809 降到 0.726
词汇平衡随机方向对照：移除 16 个方向后，AUC 仍约为 0.793
词汇平衡标签置乱方向对照：移除 16 个方向后，AUC 仍约为 0.783
学习方向与随机方向之间的差距只来自一个划分；在没有配对自助法或重复划分支持时，不应解读为稳定效应。
```

## 绘图与结果汇总

```powershell
python -m scripts.plot_results --probe figures/probe_capital_answer.csv --probe-sweep figures/probe_sweep.csv --probe-seeds figures/probe_seed_sensitivity_capital.csv --readout figures/output_readout_baselines.csv --surface figures/surface_baselines.csv --domain-transfer figures/domain_transfer_layer8.csv --domain-cosine figures/domain_direction_cosine_layer8.csv --steering figures/steering_capital_probe_layer8.csv --oracle-steering figures/oracle_steering_capital_probe_layer8.csv --patching figures/activation_patching_capital_recall.csv --truth-patching figures/truth_verification_patching_resid.csv --ablation figures/ablation_capital_probe_layer8.csv --iterative-ablation figures/iterative_ablation_capital_layer8.csv --completion-steering figures/completion_margin_steering_position_prompt_final_summary.csv --completion-steering-null figures/completion_margin_steering_null_distribution.csv
python -m scripts.plot_results --probe figures/probe_capital_balanced.csv --probe-seeds figures/probe_seed_sensitivity_capital_balanced.csv --surface figures/surface_baselines_capital_balanced.csv --ablation figures/ablation_capital_balanced_layer6.csv --iterative-ablation figures/iterative_ablation_capital_balanced_layer6.csv --knowledge-summary figures/capital_knowledge_margin_summary.csv --knowledge-details figures/capital_knowledge_margin_details.csv --completion-steering figures/completion_margin_steering_position_prompt_final_summary.csv --completion-steering-null figures/completion_margin_steering_null_distribution.csv
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.validate_project
python -m compileall scripts src
```

预期生成或刷新的产物：

```text
figures/*.png
reports/results_summary.md
```

`reports/final_report.md` 是维护中的源文档；绘图命令不会自动生成该报告。`scripts.validate_project` 会检查报告引用的图片是否存在，并检查必要文件、核心 CSV 列、词汇平衡数据形状、Markdown 链接、运行手册产物和脚本模块。预期验证结果是：`scripts.validate_project` 通过，`compileall` 无语法错误。精确数值应以 `reports/results_summary.md` 为准；本清单只记录命令、输出文件和粗粒度基本检查。
