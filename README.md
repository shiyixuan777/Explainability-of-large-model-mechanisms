# GPT-2-small 事实配对标签信号机制可解释性实验

本项目研究 GPT-2-small 在人工事实验证数据集中的 final-token residual activation 是否包含可线性读出的事实配对标签信号，并进一步分析该信号与数据伪线索、补全兼容度、跨领域泛化和推理时干预效果之间的关系。

最终结论保持克制：项目没有证明 GPT-2-small 存在跨领域稳定、可直接控制输出的全局 truth direction。更准确地说，词汇平衡首都事实中存在一个在 layer 6 prompt-final residual state 上可读的事实配对标签信号；沿该方向干预能弱移动“正确首都 vs 选定错误首都”的补全评分 margin，但尚不能稳定改变成对选择、候选集合 top-1 选择或自由生成事实行为。

## 主要文件

```text
reports/final_report.md              # 最终报告
reports/results_summary.md           # 从 CSV 结果自动汇总出的表格索引
reports/reproducibility_checklist.md  # 完整复现实验命令与预期产物

data/                                # 原始与词汇平衡事实验证数据
figures/                             # 实验 CSV 结果和 PNG 可视化
scripts/                             # 数据构建、probe、patching、steering、ablation 等实验入口
src/                                 # 数据、模型 hook、probe 与 steering 工具函数
requirements.txt                     # Python 依赖
```

## 环境

建议使用 Python 3.11 或 3.12。首次运行 GPT-2-small 会从 Hugging Face 下载模型权重。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果 Windows 当前 PATH 中的 `python` 不可用，可以直接使用虚拟环境解释器运行同一命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.build_dataset
```

## 复现顺序

完整命令见 `reports/reproducibility_checklist.md`。核心流程如下：

```powershell
python -m scripts.build_dataset
python -m scripts.build_balanced_capital_dataset

python -m scripts.run_probe_sweep --model gpt2-small --data data/facts.csv --out figures/probe_sweep.csv
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_answer.csv
python -m scripts.run_probe --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_balanced.csv

python -m scripts.run_surface_baselines --data data/facts.csv --language en --domains all capital --out figures/surface_baselines.csv
python -m scripts.run_surface_baselines --data data/capital_balanced.csv --language en --domains capital_balanced --out figures/surface_baselines_capital_balanced.csv
python -m scripts.run_capital_knowledge_margin --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --out-details figures/capital_knowledge_margin_details.csv --out-summary figures/capital_knowledge_margin_summary.csv

python -m scripts.run_completion_margin_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --bootstrap-samples 2000 --alphas -4 -2 -1 0 1 2 4 --out-details figures/completion_margin_steering_details.csv --out-summary figures/completion_margin_steering_summary.csv
python -m scripts.run_completion_margin_null_distribution --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --position-mode prompt-final-only --alpha 4 --random-directions 50 --permutation-directions 20 --out-details figures/completion_margin_steering_null_distribution.csv --out-summary figures/completion_margin_steering_null_summary.csv
python -m scripts.run_repeated_split_completion_steering --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --seeds 0 1 2 3 4 5 6 7 8 9 --alpha 4 --random-directions 10 --permutation-directions 5 --position-mode prompt-final-only --out-details figures/repeated_split_completion_steering_details.csv --out-summary figures/repeated_split_completion_steering_summary.csv

python -m scripts.run_ablation --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --direction-method probe --out figures/ablation_capital_balanced_layer6.csv
python -m scripts.run_iterative_ablation --model gpt2-small --data data/capital_balanced.csv --language en --domain capital_balanced --layer 6 --max-directions 16 --out figures/iterative_ablation_capital_balanced_layer6.csv

python -m scripts.plot_results
python -m scripts.summarize_results --figures-dir figures --out reports/results_summary.md
```

## 实验主线

1. **Locate**：用 residual probe、seed sensitivity、surface baseline、domain transfer 和 patching 分析标签信号出现的层、位置与跨域边界。
2. **Steer & Improve**：用 completion-margin steering、position decomposition、sampled null distribution、repeated split 和 candidate-rank 检查推理时干预效果。
3. **复现与扩展**：参考 Bao et al. 2025 关于 truth direction 泛化与误读风险的问题意识，在 GPT-2-small 上做小模型受控复现，并加入词汇平衡、长度归一化 completion margin、随机方向/乱标签方向等控制。

## 关键结果索引

- 原始数据：`data/facts.csv`
- 词汇平衡首都数据：`data/capital_balanced.csv`
- residual probe：`figures/probe_capital_answer.csv`, `figures/probe_capital_balanced.csv`
- surface baseline：`figures/surface_baselines.csv`, `figures/surface_baselines_capital_balanced.csv`
- completion margin baseline：`figures/capital_knowledge_margin_summary.csv`
- completion-margin steering：`figures/completion_margin_steering_summary.csv`
- position decomposition：`figures/completion_margin_steering_position_prompt_final_summary.csv`, `figures/completion_margin_steering_position_completion_internal_summary.csv`
- null distribution：`figures/completion_margin_steering_null_summary.csv`
- repeated split steering：`figures/repeated_split_completion_steering_summary.csv`
- ambiguous-fact sensitivity：`figures/ambiguous_fact_sensitivity_summary.csv`
- candidate-set rank steering：`figures/candidate_rank_steering_summary.csv`
- ablation：`figures/ablation_capital_balanced_layer6.csv`, `figures/iterative_ablation_capital_balanced_layer6.csv`
