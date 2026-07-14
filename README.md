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
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --out figures/probe_layers.csv
python -m scripts.run_steering --model gpt2-small --data data/facts.csv --language en --layer 8 --out figures/steering_alpha.csv
python -m scripts.run_activation_patching --model gpt2-small --out figures/activation_patching_capital_recall.csv
```

如果 `transformer-lens` 或 `torch` 安装很慢，可以先只运行：

```powershell
python -m scripts.build_dataset
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

- 每层 probe accuracy / AUC 曲线
- Logit Lens 层级曲线
- Activation patching heatmap
- Steering alpha-response 曲线
- 中文/英文迁移对比表
- 一份包含方法、结果、失败案例和个人分析的报告

当前报告草稿见：

```text
reports/project_report.md
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
