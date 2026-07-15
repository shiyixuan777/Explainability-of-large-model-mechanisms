# Research Integrity Checklist

正式报告：

```text
reports/final_report.md
```

这个文件只做收口核对：哪些结论有证据、哪些实验支撑报告主线、哪些表述需要保持克制。它不是报告正文的一部分。

## Evidence Map

| Research question | Evidence files |
|---|---|
| residual stream 中是否存在可线性读出的标签相关信号 | `figures/probe_capital_answer.csv`, `figures/probe_capital_balanced.csv`, `figures/probe_seed_sensitivity_capital_balanced.csv` |
| 原始 probe 是否被词汇伪线索污染 | `figures/surface_baselines.csv`, `figures/surface_baselines_capital_balanced.csv`, `data/capital_balanced.csv` |
| 剩余信号是否接近模型补全兼容度 | `figures/capital_knowledge_margin_details.csv`, `figures/capital_knowledge_margin_summary.csv` |
| 该信号是否跨领域稳定 | `figures/domain_transfer_layer8.csv`, `figures/domain_direction_cosine_layer8.csv` |
| 输出端 true/false verbalizer 是否可靠 | `figures/output_readout_baselines.csv` |
| patching 是否支持局部因果影响 | `figures/activation_patching_capital_recall.csv`, `figures/truth_verification_patching_resid.csv`, `figures/truth_verification_patching_details.csv` |
| steering 是否改变模型补全评分 | `figures/completion_margin_steering_details.csv`, `figures/completion_margin_steering_summary.csv`, `figures/completion_margin_steering_paired_bootstrap.csv` |
| steering effect 是否超过任意方向或随机标签方向 | `figures/completion_margin_steering_null_distribution.csv`, `figures/completion_margin_steering_null_summary.csv` |
| steering effect 是否跨 split 重复 | `figures/repeated_split_completion_steering_details.csv`, `figures/repeated_split_completion_steering_summary.csv` |
| 删除争议事实后信号是否定性保留 | `data/capital_balanced_no_ambiguous.csv`, `figures/ambiguous_fact_sensitivity_summary.csv` |
| steering effect 来自哪个 token position | `figures/completion_margin_steering_position_prompt_final_summary.csv`, `figures/completion_margin_steering_position_completion_internal_summary.csv`, `figures/completion_margin_steering_position_comparison.png` |
| steering 是否真的改善 choice-level 行为 | `figures/completion_margin_steering_decomposition.csv`, `figures/completion_margin_steering_summary_pairwise_accuracy.png`, `figures/candidate_rank_steering_summary.csv` |
| 简单 unembedding projection 是否足以解释 shift | `figures/unembedding_projection_baseline_details.csv`, `figures/unembedding_projection_baseline_summary.csv` |
| 移除方向后信号是否消失 | `figures/ablation_capital_probe_layer8.csv`, `figures/iterative_ablation_capital_layer8.csv`, `figures/ablation_capital_balanced_layer6.csv`, `figures/iterative_ablation_capital_balanced_layer6.csv` |
| 与 Bao et al. 的复现边界 | `reports/final_report.md` 的 Reproduction Fidelity 小节 |

## Claims To Keep

1. 原始 capital residual probe 很强，layer 8 AUC 约 0.953；但 bag-of-words 方向无关 AUC 约 0.933，因此原始结果不能直接解释为 truth direction。方向无关 AUC 只诊断分数与标签的强排序关系，不表示标签方向能稳定泛化。
2. 词汇平衡 capital 数据集使 numeric surface 与 BOW baseline 都降为 0.500；在这个设置下，layer 6 residual probe AUC 约 0.809，layer 8 AUC 约 0.802。
3. 多 seed 检查显示 balanced layer 6 mean AUC 约 0.813，layer 8 mean AUC 约 0.804，说明该信号不是 seed=42 的孤立现象。
4. Completion margin 的 total logprob AUC 为 0.861，但 avg-token AUC 为 0.786，且与 residual probe 的 bootstrap interval 高度重叠；它只能作为补全兼容度线索，而不是事实知识证明。
5. Output readout baseline 显示 GPT-2-small 对 true/false、yes/no 等 verbalizer 有强偏置，因此早期 steering 的 logit-sign 指标不能作为主要行为结论。
6. Truth verification patching 支持后层 residual state 对输出 logit difference 有局部影响，但 denominator 很小，不能直接上升为已定位 truth mechanism。
7. Balanced completion-margin steering 在当前主 split 的 alpha=+4 时把 held-out avg-token margin 推动约 +0.133 到 +0.135；paired bootstrap 与 sampled null distribution 都支持 learned direction 的 effect 高于当前 controls。
8. Repeated split steering 显示 10/10 个 group splits 中 learned shift 均为正，learned delta 均值约 +0.116，std 约 0.023，范围为 +0.085 到 +0.150。
9. Position decomposition 显示 prompt-final-only 几乎复现 all-positions 效果，而 completion-internal-only 近似为 0，因此当前 effect 主要来自 prompt final residual position。
10. Steering decomposition 显示 correct 与 false completion logprob 都被抬高，只是 correct 抬得更多；shared uplift 约 +0.214，大于 correct-over-wrong differential +0.133。
11. Repeated split 的 pairwise accuracy 只从 0.700 到 0.725，6 个 flips 均为 wrong -> correct，0 个为 correct -> wrong；这是很弱的 choice-level 改善。
12. Candidate-set rank steering 只显示正确首都 rank 小幅改善，top-1 accuracy 仍很低，因此不能写成稳定事实选择改善。
13. Ambiguous-fact sensitivity 删除 3 个争议 block 后重新划分；probe 和 steering 方向定性保留，但不与主 split 数值直接比较。
14. Unembedding projection baseline 不能用简单 `direction @ W_U` 解释逐样本 observed shift，机制路径仍未完成定位。
15. Balanced ablation 显示单方向移除后 retrained AUC 仍约 0.786，iterative learned removal 降到约 0.726，说明该方向相关但不充分。

## Claims To Avoid

1. 不写“发现了 GPT-2-small 的 truth direction”。
2. 不写“steering 改善了模型事实判断”；最多说它弱移动 correct-vs-selected-wrong completion scoring margin，并带来很小的 choice-level change。
3. 不把 completion total logprob 高 AUC 当作强证据，因为它受 completion token count 影响。
4. 不把 cross-domain transfer 的负结果写成已经证明不存在统一方向；当前只能说没有提供支持。
5. 不把 patching recovery 写成完整机制定位；本文完成的是 layer-position level localization，目前仍缺少 head/MLP/path-level decomposition。
6. 不把 null distribution 写成严格显著性检验；它是 sampled controls，经验 p 值受采样数量限制。

## Final Commands

```powershell
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.prepare_submission
python -m scripts.validate_project
python -m compileall scripts src
```

## Core Files

- `reports/final_report.md`
- `reports/results_summary.md`
- `reports/reproducibility_checklist.md`
- `reports/submission_manifest.md`
- `reports/final_deliverable_checklist.md`
- `data/facts.csv`
- `data/capital_balanced.csv`
- `src/`
- `scripts/`
- `figures/`
