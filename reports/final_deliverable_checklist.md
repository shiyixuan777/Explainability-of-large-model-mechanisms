# Final Deliverable Checklist

本清单用于提交前逐项核对课程要求。正式报告只要求 Markdown：

```text
reports/final_report.md
```

## 课程要求对应关系

| 课程要求 | 当前完成情况 | 证据文件 |
|---|---|---|
| 深入理解 Transformer 架构细节 | 报告解释 residual stream 与 hook points，实验读取 `hook_resid_post`、`hook_attn_out`、`hook_mlp_out` | `reports/final_report.md`；`src/model_hooks.py` |
| 掌握 Hook 机制 | 使用 TransformerLens 提取和修改中间层 activation | `src/model_hooks.py`；`src/steering.py`；`scripts/run_activation_patching.py` |
| Locate | 完成 domain/prompt sweep、focused capital probe、PCA、error analysis、truth verification residual patching | `figures/probe_sweep.csv`；`figures/probe_capital_answer.csv`；`figures/pca_capital_layer8.png`；`figures/truth_verification_patching_resid.csv` |
| Steer & Improve | 完成 held-out probe-direction steering 和 oracle conditional steering；后者改善内部 probe 指标，但输出仍未改善 | `figures/steering_capital_probe_layer8.csv`；`figures/oracle_steering_capital_probe_layer8.csv`；`scripts/run_steering.py`；`scripts/run_oracle_steering.py` |
| Ablation | 移除 probe direction 并重新训练 probe，验证信息不局限于单方向 | `figures/ablation_capital_probe_layer8.csv`；`scripts/run_ablation.py` |
| 顶会/Arxiv 复现 | 以 Bao et al. 2025 为主复现对象，对齐 truth direction probing、generalization 和 intervention 思路 | `reports/final_report.md` 第 2.2 节 |
| 代码与可视化 | 提供可复现实验脚本、CSV、PNG 和自动结果摘要 | `scripts/`；`src/`；`figures/`；`reports/results_summary.md` |
| 自己的分析与想法 | 报告明确分析 probe 过度解释、capital recall 限定、steering 失败和 redundant subspace | `reports/final_report.md` 第 7-9 节 |

## 核心结论核对

1. Mixed-domain truth direction 较弱，不能包装成全局 truth button。
2. Capital fact verification 中存在强线性可读信号，layer 8 AUC 为 0.953。
3. PCA 二维图不清楚分离 true/false，说明高维 supervised direction 与低维主成分不同。
4. Capital recall patching 是 related factual recall 的补充实验；truth verification residual patching 直接补充了主任务因果证据。
5. Probe-direction steering 使用 group held-out split；`alpha=0` held-out probe accuracy 为 0.826，而不是全量拟合后的 1.000。
6. Oracle conditional steering 可以把内部 probe accuracy 提升到 1.000，但 logit-sign accuracy 仍为 0.500。
7. Ablation 显示单方向 score gap 可被移除，但 retrained probe AUC 仍高于 0.94，支持 redundant subspace 解释。

## 提交前命令

```powershell
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.prepare_submission
python -m scripts.validate_project
python -m compileall scripts src
```

## 需要提交的核心文件

- `reports/final_report.md`
- `reports/results_summary.md`
- `reports/reproducibility_checklist.md`
- `reports/submission_manifest.md`
- `data/facts.csv`
- `src/`
- `scripts/`
- `figures/`
