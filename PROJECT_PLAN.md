# Two-Week Project Plan

## Topic

**基于 Truth Direction 的事实判断机制定位与推理时干预研究**

## Scope

本项目选择一个明确现象：模型判断陈述是否为真。实验目标不是解释整个模型，而是完成一个可复现、可视化充分、可被因果干预验证的小闭环。

## Day 1-2: Environment and Hook Basics

- 建立项目仓库和 Python 环境
- 跑通 TransformerLens 加载 `gpt2-small`
- 熟悉 `resid_pre`, `resid_post`, `attn_out`, `mlp_out` 等 hook 点
- 输出一个单句每层 residual 激活 shape

## Day 3-4: Dataset

- 构建英文 true/false 陈述，先保证 GPT-2-small 能稳定跑通
- 领域包括首都、国家所在洲、化学元素符号、书籍作者、地标国家、科学常识、数学事实
- 后续再单独构建中文版本，避免英文 GPT-2-small 被中文样本干扰
- 固定 train/test split

## Day 5-6: Locate with Linear Probe

- 提取每层 final-token residual stream
- 训练每层 logistic probe
- 输出每层 accuracy / AUC
- 找出 truth/false 最可分的层

## Day 7-8: Activation Patching

- 选择 true/false 成对样本
- 将 true 样本某层 residual patch 到 false 样本
- 观察 true/false logit difference 是否改变
- 输出 layer-wise patching heatmap

## Day 9-10: Steering

- 构造 `truth_direction = mean(h_true) - mean(h_false)`
- 在关键层 residual stream 上加减方向
- 扫描 alpha: `[-4, -2, -1, 0, 1, 2, 4]`
- 记录准确率、logit difference 和副作用

## Day 11: Extension

- 英文 direction -> 中文事实判断
- 中文 direction -> 英文事实判断
- 或 GPT-2-small 与 Qwen2.5-0.5B 对比

## Day 12: Visualization

- Probe layer curve
- Patching heatmap
- Steering alpha-response curve
- PCA/UMAP 激活散点图

## Day 13: Report

- 写方法、实验设置、结果和分析
- 强调 locate / steer / improve 三部分对应关系
- 加入失败样本和局限讨论

## Day 14: Cleanup and Presentation

- 清理代码和 README
- 固定随机种子
- 准备展示 slides 或报告 PDF

## Minimum Viable Submission

如果时间不够，保证完成：

1. `data/facts.csv`
2. 每层 probe 曲线
3. 一个 truth direction steering 实验
4. 报告中解释哪些层最关键、干预是否有效、局限是什么
