# 基于线性 Truth Direction 的 GPT-2-small 事实判断机制可解释性研究

## 摘要

本项目围绕“大模型机制可解释性”的 Locate、Steer & Improve 和论文复现要求，研究 GPT-2-small 在事实判断任务中是否存在可线性读取的 true/false 表征。我们构建了 528 条英文事实判断数据，覆盖 capital、continent、element_symbol、book_author、landmark_country、science、math 七个领域，并使用 TransformerLens 提取每层 residual stream 激活。初始混合领域实验表明，整体 truth/false 表征较弱；进一步按领域和 prompt 扫描后发现，capital fact verification 中存在非常强的线性可分信号，最佳层 AUC 达到 0.953。这说明在结构一致的事实任务中，GPT-2-small 的中后层 residual stream 包含可被线性 probe 读取的真假信息。然而，当前 mean-difference steering 尚未提升 true/false logit-sign accuracy，提示可读性不必然等于直接可控性，后续需要 activation patching 或 probe-direction steering 提供更强因果证据。

## 1. 项目目标

本项目选择“事实判断”作为具体可解释性现象：给定一句陈述，例如 `The capital of France is Paris.`，模型内部是否能区分该陈述为真或为假？

围绕课程要求，项目被拆成三部分：

- Locate：用线性 probe 和分层激活分析定位 true/false 信息出现在哪些层。
- Steer & Improve：构造 truth direction，在推理阶段修改 residual stream，观察模型输出是否改变。
- 复现与拓展：复现近期机制可解释性工作中关于“truth/falsehood 在激活空间中具有线性结构”的核心思想，并分析该结构是否跨领域稳定。

## 2. 复现对象与本项目对应关系

本项目主要复现和拓展 truth direction 相关工作。核心思想来自 Marks and Tegmark 的 *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*。该工作提出用简单 true/false statement 数据集研究 LLM 内部的真假表征，并从三类证据支持 truth direction：线性结构可视化、probe 跨数据集泛化、以及前向传播中的因果干预。

为了更贴合课程对 2024-2026 年顶会/Arxiv 工作的要求，本项目还对齐 Bao et al. 2025 年 arXiv 论文 *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*。该论文关心三个问题：LLM 是否普遍存在一致的 truth direction、简单 probe 是否足以识别 truth direction、truth direction 是否能跨上下文或任务泛化。本项目在 GPT-2-small 上做了一个小规模可复现实验，重点复现其中的前两个问题，并通过分领域 sweep 观察 truth direction 的稳定性。

对应关系如下：

| 论文/要求中的思想 | 本项目实现 |
|---|---|
| True/false statement datasets | 构建 528 条英文事实判断样本 |
| Linear truth representation | 每层 residual stream linear probe |
| Layer localization | `probe_sweep` 和 capital layer probe |
| Cross-domain consistency | capital、continent、element_symbol、book_author、landmark_country、science、math 分领域比较 |
| Causal intervention | residual/attention/MLP activation patching |
| Vector arithmetic | probe-direction steering 和 ablation |

## 3. 背景方法

Transformer 的 residual stream 可以被理解为模型在每一层积累和传递信息的主通道。TransformerLens 提供 hook 机制，可以在模型前向传播时读取或修改这些中间激活。本项目主要使用以下 hook 点：

```text
blocks.{layer}.hook_resid_post
```

对每个输入 prompt，我们提取最后一个 token 位置的 residual stream 激活，并训练 logistic regression probe 判断该陈述的标签：

```text
label = 1: true statement
label = 0: false statement
```

如果某一层的激活可以被线性 probe 高精度区分 true/false，则说明该层中存在可线性读取的真假相关信息。

## 4. 数据集

当前数据集位于：

```text
data/facts.csv
```

数据规模为 528 条英文事实判断样本，true/false 完全平衡：

```text
true  = 264
false = 264
```

领域分布如下：

| Domain | Rows |
|---|---:|
| capital | 152 |
| continent | 86 |
| element_symbol | 80 |
| book_author | 60 |
| landmark_country | 60 |
| science | 50 |
| math | 40 |

每个 `pair_id` 包含一条 true statement 和一条对应的 false statement。实验中使用 group split，保证同一个事实对不会同时出现在训练集和测试集中，从而减少数据泄漏。

## 5. 实验设置

模型：

```text
gpt2-small
```

框架：

```text
TransformerLens
```

主要 prompt 模板包括：

```text
Statement: {statement}
The statement is
```

```text
Statement: {statement}
Answer true or false:
```

```text
Question: Is the following statement true or false?
{statement}
Answer:
```

评价指标：

- Accuracy：probe 在测试集上的分类准确率。
- AUC：probe 输出概率对 true/false 的排序能力。
- Separability AUC：`max(AUC, 1 - AUC)`，用于衡量激活是否可线性分离，即使方向反了也能体现可分性。

## 6. Locate 实验结果

### 6.1 混合领域结果

在全部 528 条样本上训练每层 probe 时，最佳结果只有中等强度。以 `question` prompt 为例，最佳 separability AUC 约为 0.653。这说明将所有事实类型混在一起时，GPT-2-small 中不存在非常稳定的通用 truth direction，或者该方向不能被当前 final-token residual probe 稳定读出。

这也是第一轮实验中 AUC 接近随机的原因：不同领域的事实结构差异很大，统一线性边界被稀释。

### 6.2 分领域 sweep 结果

我们进一步运行 `scripts/run_probe_sweep.py`，比较不同领域和 prompt。结果显示，truth/false 可分性具有明显领域差异。

当前最强的若干设置为：

| Domain | Prompt | Best Layer | Accuracy | AUC | Separability AUC |
|---|---|---:|---:|---:|---:|
| capital | answer | 8 | 0.826 | 0.953 | 0.953 |
| capital | statement_is | 6 | 0.804 | 0.940 | 0.940 |
| capital | question | 7 | 0.804 | 0.932 | 0.932 |
| continent | statement_is | 11 | 0.808 | 0.846 | 0.846 |
| landmark_country | statement_is | 9 | 0.778 | 0.815 | 0.815 |

完整结果见：

```text
figures/probe_sweep.csv
figures/probe_sweep_summary.png
```

### 6.3 Capital fact verification 结果

由于 capital 领域结果最稳定，我们将其作为当前主实验对象。使用 prompt：

```text
Statement: {statement}
Answer true or false:
```

各层结果显示，中后层出现强线性可分信号：

| Layer | Accuracy | AUC |
|---:|---:|---:|
| 5 | 0.848 | 0.941 |
| 6 | 0.826 | 0.943 |
| 7 | 0.848 | 0.938 |
| 8 | 0.826 | 0.953 |
| 9 | 0.804 | 0.941 |
| 10 | 0.870 | 0.947 |

这表明在 capital fact verification 中，GPT-2-small 的 residual stream 确实包含可线性读取的真假信息。最佳 AUC 出现在第 8 层，最佳 accuracy 出现在第 10 层。考虑 GPT-2-small 共 12 层，这说明真假相关信息主要在中后层形成。

对应图表：

```text
figures/probe_capital_answer.png
```

### 6.4 Activation PCA 可视化

为了给 linear probe 结果提供更直观的辅助证据，我们新增了第 8 层 residual activation 的 PCA 可视化。该实验不训练分类器，而是直接把 capital 样本在第 8 层 final-token residual stream 中的高维激活降到二维，观察 true/false 样本是否出现可见的空间结构。

运行命令：

```powershell
python -m scripts.run_activation_pca --model gpt2-small --data data/facts.csv --language en --domain capital --layer 8 --out figures/pca_capital_layer8.csv
```

输出文件：

```text
figures/pca_capital_layer8.csv
figures/pca_capital_layer8.png
```

当前 PCA 图中 true/false 样本没有在二维平面上完全分开，这反而提醒我们：第 8 层 AUC=0.953 的可分性主要由高维 probe 方向捕捉，未必会直接出现在解释方差最大的前两个主成分上。绘图时 CSV 保留全部样本，但 PNG 默认裁掉二维坐标两端 1% 极端点，避免个别长实体样本拉伸坐标轴。

因此，PCA 不能代替 probe 的定量结论。如果图上 true/false 没有完全分开，也不意味着高维空间不可分；本项目仍以 group split probe AUC 作为主要定位指标，PCA 图只作为报告中的直观补充。

## 7. Steering 与 Ablation 结果

我们首先尝试最简单的 mean-difference truth direction：

```text
truth_direction = mean(h_true) - mean(h_false)
```

在 capital 领域第 8 层上进行 alpha sweep，结果如下：

| Alpha | Mean Logit Diff | True Mean | False Mean | Sign Accuracy |
|---:|---:|---:|---:|---:|
| -4 | 1.584 | 1.576 | 1.592 | 0.500 |
| -2 | 1.570 | 1.562 | 1.577 | 0.500 |
| -1 | 1.562 | 1.554 | 1.570 | 0.500 |
| 0 | 1.555 | 1.547 | 1.562 | 0.500 |
| 1 | 1.547 | 1.540 | 1.554 | 0.500 |
| 2 | 1.539 | 1.532 | 1.546 | 0.500 |
| 4 | 1.523 | 1.516 | 1.530 | 0.500 |

这个结果说明干预确实改变了 `true` 与 `false` token 的 logit difference，但没有提高基于 logit sign 的判断准确率。更重要的是，true 和 false 样本的 logit diff 几乎一起移动，因此当前 naive steering 更像是在整体改变输出偏置，而不是按样本真假改善判断。

因此目前结论是：

> Linear probe 证明了 capital 任务中存在可读的真假信息，但 mean-difference steering 尚未证明该方向具有直接可控的因果作用。

### 7.1 Probe-direction steering

为了更贴近 Locate 阶段发现的线性边界，我们进一步使用 logistic regression probe 的权重方向作为 steering direction。具体做法是先在第 8 层 capital activations 上训练 probe，再把标准化空间中的线性权重转换回原始 residual stream 坐标，得到单位向量：

```text
v_probe = normalized(probe_weight_in_activation_space)
```

然后在第 8 层最后 token 位置执行：

```text
h = h + alpha * v_probe
```

实验输出两个指标：

- `accuracy_from_logit_sign`：根据模型输出 `logit(" true") - logit(" false")` 判断。
- `accuracy_from_probe_score_threshold`：根据内部 probe projection score 和校准阈值判断。

结果如下：

| Alpha | Logit-sign Accuracy | Probe-threshold Accuracy | Mean Probe Score |
|---:|---:|---:|---:|
| -8 | 0.500 | 0.500 | -9.353 |
| -4 | 0.500 | 0.500 | -5.353 |
| -2 | 0.500 | 0.500 | -3.353 |
| -1 | 0.500 | 0.507 | -2.353 |
| 0 | 0.500 | 1.000 | -1.353 |
| 1 | 0.500 | 0.500 | -0.353 |
| 2 | 0.500 | 0.500 | 0.647 |
| 4 | 0.500 | 0.500 | 2.647 |
| 8 | 0.500 | 0.500 | 6.647 |

这个结果说明 probe direction 确实控制了内部表示的 projection score：alpha 每增加 1，mean probe score 也近似增加 1。然而，因为我们对所有样本统一加入同一个方向，true 和 false 样本会一起移动，导致 calibrated probe accuracy 在 alpha 偏离 0 后反而下降。这是一个重要负结果：

> Probe direction 可以控制内部表示的“真假方向坐标”，但 naive global steering 不能自动 improve true/false 分类；要改善行为，需要输入条件化 steering、ablation 或更精细的 causal intervention。

对应图表：

```text
figures/steering_capital_probe_layer8.csv
figures/steering_capital_probe_layer8.png
figures/steering_capital_probe_layer8_probe_accuracy.png
```

### 7.2 Probe-direction ablation

为了进一步验证 probe direction 是否确实承载了可读的 true/false 信息，我们做了 ablation 实验。做法是先在训练 split 上学习第 8 层 capital probe direction，然后从 residual activation 中移除该方向上的投影：

```text
h_ablated = h - strength * (h · v_probe) * v_probe
```

实验同时报告两类指标：

- Fixed direction：继续使用原来的 probe direction 读出 true/false，观察该方向本身是否被移除。
- Retrained probe：在 ablated activation 上重新训练一个 probe，观察信息是否还能被其他方向读出。

关键结果如下：

| Strength | Fixed Direction Score Gap | Fixed Direction Accuracy | Retrained Probe AUC |
|---:|---:|---:|---:|
| 0.00 | +0.573 | 0.826 | 0.953 |
| 0.25 | +0.430 | 0.500 | 0.955 |
| 0.50 | +0.286 | 0.500 | 0.958 |
| 0.75 | +0.143 | 0.500 | 0.951 |
| 1.00 | -0.000 | 0.500 | 0.945 |
| 1.25 | -0.143 | 0.500 | 0.953 |
| 1.50 | -0.286 | 0.478 | 0.962 |

这个结果说明，沿 `v_probe` 移除投影后，原方向上的 true/false score gap 从 `+0.573` 逐步下降到 0，并在过度 ablation 后反向。这证明我们确实可以用 vector arithmetic 控制该线性方向上的信息。然而，重新训练 probe 后 AUC 仍保持在 0.94 以上，说明 capital true/false 信息并不是只存在于单一方向中，而是可能分布在多个相关方向或子空间中。

因此 ablation 给出的结论比 naive steering 更细：

> 我们可以有效移除一个已发现的 probe direction，但 GPT-2-small 的 capital truth/false 信息具有冗余表示，单方向 ablation 不足以摧毁所有可线性读取的信息。

对应图表：

```text
figures/ablation_capital_probe_layer8.csv
figures/ablation_capital_probe_layer8.png
figures/ablation_capital_probe_layer8_score_gap.png
```

## 8. Activation Patching 结果

为了补充更直接的因果定位实验，我们新增了 capital recall 形式的 activation patching。这个实验不再让 GPT-2-small 输出 `true` 或 `false`，而是使用更适合自回归语言模型的事实召回 prompt：

```text
The capital of France is
```

实验构造 clean/corrupt prompt pair。例如：

```text
clean:   The capital of France is
corrupt: The capital of Germany is
```

目标指标为 clean capital token 相对 corrupt capital token 的 logit difference：

```text
logit(" Paris") - logit(" Berlin")
```

然后在 corrupt prompt 的前向传播中，将某一层最后 token 位置的 clean activation patch 到 corrupt run 中，观察目标 logit difference 是否恢复。当前实验比较了三类 hook：

```text
blocks.{layer}.hook_resid_post
blocks.{layer}.hook_attn_out
blocks.{layer}.hook_mlp_out
```

当前实验自动跳过 GPT-2 tokenizer 中不是 single token 的首都名，最终使用 22 对 single-token capital pairs。首先看 residual stream patching：

| Layer | Patched Logit Diff | Mean Recovery |
|---:|---:|---:|
| 0 | -0.788 | 0.123 |
| 1 | -0.764 | 0.078 |
| 2 | -0.722 | 0.150 |
| 3 | -0.691 | 0.226 |
| 4 | -0.730 | -0.018 |
| 5 | -0.763 | -0.753 |
| 6 | -0.775 | -1.520 |
| 7 | -0.757 | -1.450 |
| 8 | -0.657 | -1.290 |
| 9 | -0.359 | -1.203 |
| 10 | -0.154 | -0.813 |
| 11 | 0.264 | 1.000 |

其中 clean baseline 的平均 logit difference 为 `+0.264`，corrupt baseline 为 `-1.039`。第 11 层 patch 后恢复到 `+0.264`，说明 residual stream 最后一层携带了足以恢复目标首都输出的信息。第 9-10 层 patched logit diff 也明显向 clean 方向移动，但尚未完全恢复。

模块级 patching 进一步显示：

| Component | Best Layer | Patched Logit Diff | Mean Recovery | Median Recovery |
|---|---:|---:|---:|---:|
| resid_post | 11 | 0.264 | 1.000 | 1.000 |
| attn_out | 11 | -0.672 | 1.762 | 1.077 |
| mlp_out | 7 | -1.022 | 0.388 | 0.074 |

这个结果提供了比线性 probe 更强的因果证据：在 capital recall 任务中，替换关键层 residual stream 会直接改变目标首都 token 的输出 logit。模块对比提示，后层 attention output 对恢复目标首都信息有明显贡献，而 MLP output 的单独 patching 效果较弱。需要注意的是，`attn_out` 的 mean recovery 较高，但平均 patched logit diff 仍未翻正，因此它更适合被解释为“对恢复有强贡献”，而不是“单独足以完成恢复”。整体 causal effect 最清晰地出现在最后层 residual stream。

对应结果文件：

```text
figures/activation_patching_capital_recall.csv
figures/activation_patching_capital_recall.png
```

## 9. 当前结论

目前最可靠的结论有三点：

1. 在混合领域事实判断中，GPT-2-small 的 truth/false 线性结构较弱。
2. 在结构一致的 capital fact verification 中，truth/false 信息可被高精度线性 probe 读取，最佳 AUC 达到 0.953。
3. 在 capital recall 的 activation patching 中，后层 residual stream patching 可以恢复目标首都 logit，模块级结果显示最后层 attention output 贡献明显，MLP output 单独贡献较弱。
4. Probe-direction steering 可以控制内部 probe score，但 naive global steering 没有提升 true/false 输出判断，说明可读性不等于直接可改善性。
5. Probe-direction ablation 可以移除已发现方向上的 score gap，但重新训练 probe 仍能恢复高 AUC，提示 truth/false 信息存在冗余子空间。

## 10. 下一步计划

后续需要补强课程要求中的因果干预部分：

1. 将 activation patching 扩展到 head-level attention patching。
2. 尝试子空间 ablation，例如移除多个 probe/PC directions，而不是单一方向。
3. 可选模型拓展：在 Qwen2.5-0.5B 或 Qwen2.5-0.5B-Instruct 上复现实验。

## 11. 个人分析

这组结果说明，“truth direction”不一定是一个跨所有事实领域共享的单一方向。至少在 GPT-2-small 上，更合理的解释是：某些结构一致的任务，例如首都事实判断，会在中后层形成稳定的线性可分表示；而当任务混合了首都、元素符号、书籍作者、数学和科学常识后，统一 probe 的效果明显下降。

这也提示机制可解释性实验需要非常重视任务定义。如果现象定义过宽，实验可能得到“没有信号”的结论；但通过控制领域和 prompt，可以把模型内部表示中的局部结构暴露出来。后续的关键问题是：这个线性结构只是 probe 可读，还是模型实际用于输出判断的因果机制？这需要 activation patching 和 steering 实验进一步验证。

## 12. 参考文献

1. Samuel Marks and Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*. arXiv:2310.06824. https://arxiv.org/abs/2310.06824
2. Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Zhengwen Feng, Hao Peng, Jianwei Yin. *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*. arXiv:2506.00823. https://arxiv.org/abs/2506.00823
3. Leonard Bereska and Efstratios Gavves. *Mechanistic Interpretability for AI Safety -- A Review*. arXiv:2404.14082.
4. *Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in LLMs*. arXiv 2026.
