# GPT-2-small 事实配对标签信号机制可解释性实验

本项目研究 GPT-2-small 在人工事实验证数据集中的残差激活是否包含可线性读出的事实配对标签信号，并分析该信号与词汇伪线索、补全兼容度、跨领域泛化和推理时激活干预效果之间的关系。

最终结论保持克制：项目没有证明 GPT-2-small 存在跨领域稳定、可直接控制输出的全局真值方向（truth direction）。更准确地说，词汇平衡首都事实中，在事实验证提示词的第 6 层最后词元残差激活上可以读出标签相关信号；将该方向迁移到直接补全提示词的第 6 层提示词末位置后，可弱移动“正确首都 vs 选定错误首都”的补全得分差。当前尚不能稳定改变成对选择或受限候选集合 top-1 选择；开放式自由生成尚未系统评估。

## 主要文件

- [最终报告](reports/final_report.md)：面向老师阅读的正式小论文。
- [结果汇总](reports/results_summary.md)：从 CSV 自动生成的结果索引。
- [复现说明](reports/reproducibility_checklist.md)：完整实验运行手册和快速复现命令。
- [直接依赖](requirements.txt)：固定版本的直接 Python 依赖。

`figures/` 中已包含预计算 CSV 和 PNG；只阅读报告不需要先重跑实验。

## 环境

当前结果测试于 Windows 11、Python 3.13.7、PyTorch 2.12.1 CPU、TransformerLens 3.5.1。关键依赖见 `requirements.txt`；首次运行 GPT-2-small 会从 Hugging Face 下载模型权重。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 快速复现

复现主线实验请直接运行 [复现说明的“快速复现”](reports/reproducibility_checklist.md#快速复现)。README 不再复制完整命令，避免两处命令不一致。

最短阅读路径：

1. 先看 [final_report.md](reports/final_report.md) 的摘要、实验结果和讨论。
2. 需要核对数值时看 [results_summary.md](reports/results_summary.md)。
3. 需要重跑实验时按 [reproducibility_checklist.md](reports/reproducibility_checklist.md) 执行。

## 项目结构

```text
data/       原始与词汇平衡事实验证数据
figures/    实验 CSV 结果和 PNG 可视化
reports/    正式报告、复现说明和结果索引
scripts/    数据构建、探针、激活修补、激活干预、消融等实验入口
src/        数据、模型 hook、探针与激活干预工具函数
```

## 实验主线

1. **定位（Locate）**：用残差流探针、表面特征基线和跨领域迁移分析读出位置与跨域边界，并以激活修补作为补充诊断。
2. **干预与改进（Steer & Improve）**：用提示词末位置补全得分差干预、位置分解、采样零分布、重复划分和候选集排名检查推理时干预效果及 Improve 的边界。
3. **复现与扩展**：参考真值方向泛化相关工作，在 GPT-2-small 上做小模型受控复现，并加入词汇平衡、长度归一化补全得分差、随机方向/标签置乱方向等控制。
