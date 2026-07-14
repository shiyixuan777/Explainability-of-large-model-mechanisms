# Mechanistic Interpretability: Truth Direction

本项目用于两周内完成“大模型机制可解释性”课程项目：围绕事实判断现象，定位 Transformer 中表示 truth/false 的关键层，并用 truth direction 在推理阶段干预模型输出。

## Recommended Project Title

**基于 Truth Direction 的事实判断机制定位与推理时干预研究**

## Research Questions

1. 模型在第几层开始线性地区分真实陈述和错误陈述？
2. 这些信息主要出现在 residual stream、attention output 还是 MLP output？
3. 从激活中提取的 truth direction 是否能因果性地改变模型判断？
4. 英文样本学到的 truth direction 能否迁移到中文事实判断？

## Method

- **Locate**: Logit Lens, linear probe, activation patching
- **Steer**: truth direction vector arithmetic
- **Improve**: 在推理时加入或减去方向向量，比较准确率和 logit difference
- **Reproduce**: 复现 truth/false 表示具有线性结构、且可被干预改变的核心结论

## Quick Start

先按 [INSTALL.md](INSTALL.md) 安装 Python 和依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.check_env
python -m scripts.build_dataset
```

如果 `transformer-lens` 或 `torch` 安装很慢，可以先只运行：

```powershell
python -m scripts.build_dataset
```

## Full Reproduction Pipeline

下面这组命令复现当前报告中的主要结果。第一次加载 `gpt2-small` 时需要下载 Hugging Face 权重，后续会使用本地缓存。

```powershell
python -m scripts.run_probe_sweep --model gpt2-small --data data/facts.csv --out figures/probe_sweep.csv
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_answer.csv
python -m scripts.run_activation_pca --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/pca_capital_layer8.csv
python -m scripts.run_error_analysis --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/error_analysis_capital_layer8.csv
python -m scripts.run_activation_patching --model gpt2-small --out figures/activation_patching_capital_recall.csv --components resid_post,attn_out,mlp_out
python -m scripts.run_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --alphas -8 -4 -2 -1 0 1 2 4 8 --out figures/steering_capital_probe_layer8.csv
python -m scripts.run_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --out figures/ablation_capital_probe_layer8.csv
python -m scripts.plot_results --probe figures/probe_capital_answer.csv --probe-sweep figures/probe_sweep.csv --steering figures/steering_capital_probe_layer8.csv --patching figures/activation_patching_capital_recall.csv --ablation figures/ablation_capital_probe_layer8.csv
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
python -m scripts.validate_project
```

更详细的复现实验清单见：

```text
reports/reproducibility_checklist.md
```

## Repository Structure

```text
.
├── configs/              # 实验配置
├── data/                 # 小型事实判断数据集
├── experiments/          # 实验记录和临时输出
├── figures/              # 图表和 CSV 结果
├── notebooks/            # 可视化分析 notebook
├── reports/              # 项目报告草稿
├── scripts/              # 命令行入口
└── src/                  # 可复用代码
```

## Expected Deliverables

- 每层 probe accuracy / AUC 曲线：`figures/probe_capital_answer.png`
- 分领域 probe sweep：`figures/probe_sweep_summary.png`
- 激活 PCA 散点图：`figures/pca_capital_layer8.png`
- 样本级错误分析：`figures/error_analysis_capital_layer8_errors.csv`
- 模块级 activation patching：`figures/activation_patching_capital_recall.png`
- Probe-direction steering：`figures/steering_capital_probe_layer8*.png`
- Probe-direction ablation：`figures/ablation_capital_probe_layer8*.png`
- 自动结果汇总：`reports/results_summary.md`
- 一键交付验证：`python -m scripts.validate_project`
- 正式版项目报告：`reports/final_report.md`
- 项目报告草稿：`reports/project_report.md`
- 复现实验清单：`reports/reproducibility_checklist.md`
- 最终交付检查表：`reports/final_deliverable_checklist.md`
- 答辩提纲：`reports/presentation_outline.md`

当前正式版报告见：

```text
reports/final_report.md
```

提交前建议按以下文件逐项核对：

```text
reports/final_deliverable_checklist.md
reports/presentation_outline.md
```

## Notes

默认数据集 `data/facts.csv` 包含 528 条英文事实判断样本，true/false 各 264 条，覆盖 capital、continent、element_symbol、book_author、landmark_country、science、math 七个领域。默认模型是 `gpt2-small`，因为它和 TransformerLens 配合最稳定。完成第一版后，可以把模型替换为 `Qwen/Qwen2.5-0.5B` 或 `Qwen/Qwen2.5-1.5B`，但可能需要根据框架支持情况调整 hook 代码。

可以用领域过滤做小实验：

```powershell
python -m scripts.run_probe --domain capital,science
python -m scripts.run_steering --layer 8 --domain capital,science
```

运行 probe-direction steering：

```powershell
python -m scripts.run_steering --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --alphas -8 -4 -2 -1 0 1 2 4 8 --out figures/steering_capital_probe_layer8.csv
```

运行 probe-direction ablation：

```powershell
python -m scripts.run_ablation --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --direction-method probe --out figures/ablation_capital_probe_layer8.csv
python -m scripts.plot_results --ablation figures/ablation_capital_probe_layer8.csv
```

运行分领域 probe sweep：

```powershell
python -m scripts.run_probe_sweep --model gpt2-small --data data/facts.csv --out figures/probe_sweep.csv
```

运行模块级 activation patching：

```powershell
python -m scripts.run_activation_patching --components resid_post,attn_out,mlp_out
python -m scripts.plot_results --patching figures/activation_patching_capital_recall.csv
```
