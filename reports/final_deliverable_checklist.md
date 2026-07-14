# Final Deliverable Checklist

本文档用于把课程要求逐条映射到当前代码、实验结果和报告章节。它不是替代正式报告，而是最后提交前的核查表。

## 课程要求对应关系

| 课程要求 | 当前完成情况 | 证据文件 |
|---|---|---|
| 深入理解 Transformer 架构细节 | 报告解释 residual stream 与 hook 点，实验使用 `hook_resid_post`、`hook_attn_out`、`hook_mlp_out` | `reports/project_report.md` 第 3、8 节；`src/model_hooks.py`；`scripts/run_activation_patching.py` |
| 掌握 Hook 机制 | 使用 TransformerLens 读取和修改中间层激活 | `src/model_hooks.py`；`src/steering.py`；`scripts/run_activation_patching.py` |
| Locate：定位关键层与模块 | 完成分领域 probe sweep、focused capital probe、PCA 辅助可视化、activation patching | `figures/probe_sweep.csv`；`figures/probe_capital_answer.csv`；`figures/pca_capital_layer8.png`；`figures/activation_patching_capital_recall.csv` |
| Steer & Improve：推理阶段干预 | 完成 mean-difference/probe-direction steering 与 probe-direction ablation | `scripts/run_steering.py`；`scripts/run_ablation.py`；`figures/steering_capital_probe_layer8.csv`；`figures/ablation_capital_probe_layer8.csv` |
| 顶会/Arxiv 机制可解释性论文复现 | 对齐 truth direction / geometry of truth 工作，复现 true/false 线性结构、可视化、因果干预思想 | `reports/project_report.md` 第 2 节；`reports/results_summary.md` |
| 代码与详细可视化对比结果 | 提供脚本、CSV、PNG、自动结果汇总 | `scripts/`；`src/`；`figures/`；`reports/results_summary.md` |
| 总结自己的分析与想法 | 报告包含负结果、错误样本、局限与个人分析 | `reports/project_report.md` 第 6.5、7、9、11 节 |

## 当前核心结论

1. 混合领域事实判断中的通用 truth direction 较弱，最佳 separability AUC 只有中等水平。
2. 在结构一致的 capital fact verification 中，GPT-2-small 中后层 residual stream 有强线性 truth/false 信号，最佳 AUC 为 0.953。
3. 二维 PCA 不能清晰分开 true/false，说明可分方向主要存在于高维 supervised probe 方向中。
4. Activation patching 显示 capital recall 的因果信号集中在后层 residual stream，最后层 residual patching 可以恢复目标首都 logit。
5. Probe-direction steering 可以移动内部 probe score，但 naive global steering 没有提升 true/false logit-sign accuracy。
6. 单方向 ablation 可以移除已发现 probe direction 上的 score gap，但重新训练 probe 仍能恢复高 AUC，提示信息存在冗余子空间。

## 最后提交前还应检查

- 运行 `python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md`，确保报告中的数字与 CSV 一致。
- 运行 `python -m scripts.validate_project`，检查关键文件、图表和核心指标是否齐全。
- 运行 `python -m compileall scripts src`，确认脚本语法无误。
- 检查 `reports/project_report.md` 是否已经转成老师要求的 PDF 或 Word 格式。
- 检查所有报告中引用的图表路径是否存在。
- 如需要答辩，按 `reports/presentation_outline.md` 制作 8-10 页 slides。
