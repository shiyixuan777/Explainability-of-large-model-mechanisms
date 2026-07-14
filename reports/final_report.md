# 基于 Truth Direction 的 GPT-2-small 事实判断机制定位与推理时干预研究

## 摘要

本项目围绕“大模型机制可解释性”课程要求，研究 GPT-2-small 在事实判断任务中是否存在可定位、可干预、可复现的 true/false 内部表征。我们构建了 528 条英文事实判断样本，覆盖 capital、continent、element_symbol、book_author、landmark_country、science、math 七个领域，并使用 TransformerLens 提取每层 residual stream 激活。实验包括三部分：第一，使用 linear probe、PCA 和 activation patching 定位事实真假信息所在层与模块；第二，使用 vector arithmetic、steering 和 ablation 在推理阶段干预模型内部表示；第三，对齐 truth direction / geometry of truth 相关论文，复现“真假表征在激活空间中呈线性结构”的核心思想，并分析其局限。

主要结果表明：在混合领域事实判断中，GPT-2-small 的通用 truth direction 较弱；但在结构一致的 capital fact verification 中，中后层 residual stream 出现强线性可分信号，最佳 AUC 为 0.953。Activation patching 显示，在 capital recall 任务中，最后层 residual stream patching 可以恢复目标首都 token 的 logit。Probe-direction steering 能稳定移动内部 probe score，但 naive global steering 没有提升 true/false 输出判断；单方向 ablation 能移除已发现方向上的 score gap，但重新训练 probe 仍能恢复高 AUC，提示真假信息存在冗余或分布式表示。

## 1. 研究问题与项目目标

本项目选择“事实判断”作为具体可解释性现象。给定一句陈述，例如：

```text
The capital of France is Paris.
```

我们希望回答以下问题：

1. GPT-2-small 的哪些层开始包含可线性读取的 true/false 信息？
2. 这些信息主要体现在 residual stream、attention output 还是 MLP output 中？
3. 从激活中学到的 truth direction 是否能在推理阶段改变模型内部表示或输出行为？
4. 这种 truth/false 线性结构是否跨事实领域稳定？

课程要求中的三项任务对应如下：

| 课程要求 | 本项目实现 |
|---|---|
| Locate | Linear probe、PCA、activation patching 定位关键层与模块 |
| Steer & Improve | Probe-direction steering 与 probe-direction ablation |
| 顶会/Arxiv 复现 | 对齐 truth direction / geometry of truth 工作，复现线性真假表征和因果干预思想 |

## 2. 相关工作与复现对象

本项目主要对齐两类近期机制可解释性工作。

第一类是 Marks and Tegmark 的 *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*。该工作提出，LLM 对 true/false statements 的内部表示可能存在线性结构，并通过 probe、可视化和因果干预论证 truth direction 的存在。

第二类是 Bao et al. 2025 年 arXiv 论文 *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*。该论文关注 truth direction 是否稳定、是否能被简单 probe 识别、以及是否能跨上下文或任务泛化。

本项目没有直接复现大模型上的全部设置，而是在 GPT-2-small 上做一个小规模、可复现、可视化充分的版本。复现点包括：

- 构建 true/false statement dataset；
- 对每层 residual stream 训练 linear probe；
- 用 PCA 辅助观察激活空间结构；
- 用 activation patching 和 steering/ablation 测试方向的因果作用；
- 通过分领域 sweep 检查 truth direction 是否跨领域稳定。

## 3. 数据集与实验设置

数据集位于：

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

每个 `pair_id` 包含一条 true statement 和一条对应 false statement。实验使用 group split，避免同一个事实对同时出现在训练集和测试集中，从而降低数据泄漏。

模型和框架：

```text
Model: gpt2-small
Framework: TransformerLens
Hook points: hook_resid_post, hook_attn_out, hook_mlp_out
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

主要指标：

- Accuracy：固定阈值下的分类准确率；
- AUC：probe 对 true/false 样本的排序能力；
- Separability AUC：`max(AUC, 1 - AUC)`，用于衡量可分性，即使方向反了也能体现线性结构；
- Logit difference：目标 token 与对照 token 的输出 logit 差；
- Recovery：activation patching 后目标 logit difference 的恢复比例。

## 4. Locate: 线性定位与可视化

### 4.1 分领域 probe sweep

我们首先对所有领域和 prompt 进行 sweep。结果显示，混合领域的通用 truth direction 较弱，而结构一致的 capital 领域最稳定。

| Domain | Prompt | Best Layer | Accuracy | AUC | Separability AUC |
|---|---|---:|---:|---:|---:|
| capital | answer | 8 | 0.826 | 0.953 | 0.953 |
| capital | statement_is | 6 | 0.804 | 0.940 | 0.940 |
| capital | question | 7 | 0.804 | 0.932 | 0.932 |
| continent | statement_is | 11 | 0.808 | 0.846 | 0.846 |
| landmark_country | statement_is | 9 | 0.778 | 0.815 | 0.815 |

结果文件：

```text
figures/probe_sweep.csv
figures/probe_sweep_summary.png
```

这一结果说明，truth direction 不应被简单理解成一个跨所有事实领域共享的单一方向。至少在 GPT-2-small 上，领域和 prompt 对真假可分性有很大影响。

### 4.2 Focused capital probe

由于 capital 领域最稳定，我们使用如下 prompt 进行 focused probe：

```text
Statement: {statement}
Answer true or false:
```

中后层结果最强：

| Layer | Accuracy | AUC |
|---:|---:|---:|
| 5 | 0.848 | 0.941 |
| 6 | 0.826 | 0.943 |
| 7 | 0.848 | 0.938 |
| 8 | 0.826 | 0.953 |
| 9 | 0.804 | 0.941 |
| 10 | 0.870 | 0.947 |

最佳 AUC 出现在第 8 层，最佳 accuracy 出现在第 10 层。考虑 GPT-2-small 共 12 层，这说明 capital fact verification 的真假信息主要在中后层形成。

结果文件：

```text
figures/probe_capital_answer.csv
figures/probe_capital_answer.png
```

### 4.3 PCA 辅助可视化

我们对第 8 层 capital residual activation 做 PCA 可视化：

```text
figures/pca_capital_layer8.csv
figures/pca_capital_layer8.png
```

PCA 的 PC1 explained variance 为 0.620，PC2 explained variance 为 0.117。二维图没有清楚分开 true/false，这一点很重要：linear probe 的强 AUC 不一定对应解释方差最大的主成分。换句话说，真假可分性主要存在于 supervised high-dimensional probe direction 中，而不是无监督 PCA 的前两个方向中。

### 4.4 样本级错误分析

第 8 层 capital probe 在测试集上共有 46 条样本，其中 38 条预测正确，8 条预测错误，accuracy=0.826，AUC=0.953。

代表性错误如下：

| Statement | Label | Prediction | prob_true |
|---|---|---|---:|
| The capital of Laos is Vientiane. | true | false | 0.003 |
| The capital of Canada is Amman. | false | true | 0.964 |
| The capital of Chile is Santiago. | true | false | 0.179 |
| The capital of India is New Delhi. | true | false | 0.217 |
| The capital of Morocco is Rabat. | true | false | 0.219 |
| The capital of Nigeria is Mexico City. | false | true | 0.688 |

结果文件：

```text
figures/error_analysis_capital_layer8.csv
figures/error_analysis_capital_layer8_errors.csv
```

这个结果说明：probe 的排序能力很强，因此 AUC 很高；但固定 0.5 阈值仍会产生错误。也就是说，当前实验更能证明“该层存在可线性读取的真假信息”，而不是证明该 probe 本身已经是完美事实判断器。

## 5. Activation Patching: 因果定位

为了获得比 probe 更强的因果证据，我们构造 capital recall 任务：

```text
clean:   The capital of France is
corrupt: The capital of Germany is
metric:  logit(" Paris") - logit(" Berlin")
```

在 corrupt prompt 的前向传播中，将 clean prompt 在某层最后 token 位置的 activation patch 进去，观察目标 logit difference 是否恢复。实验比较三类 hook：

```text
blocks.{layer}.hook_resid_post
blocks.{layer}.hook_attn_out
blocks.{layer}.hook_mlp_out
```

模块级最好结果如下：

| Component | Best Layer | Patched Logit Diff | Mean Recovery | Median Recovery |
|---|---:|---:|---:|---:|
| resid_post | 11 | 0.264 | 1.000 | 1.000 |
| attn_out | 11 | -0.672 | 1.762 | 1.077 |
| mlp_out | 7 | -1.022 | 0.388 | 0.074 |

结果文件：

```text
figures/activation_patching_capital_recall.csv
figures/activation_patching_capital_recall.png
```

最后层 residual stream patching 可以把平均 logit difference 恢复到 clean baseline，说明后层 residual stream 携带了足以恢复目标首都 token 的信息。`attn_out` 在第 11 层的 mean recovery 很高，但 patched logit diff 仍未翻正，因此更适合解释为“对恢复有强贡献”，而不是“单独足以完成恢复”。`mlp_out` 单独 patching 效果较弱。

## 6. Steer & Improve: 推理阶段干预

### 6.1 Probe-direction steering

我们在第 8 层 capital activation 上训练 logistic probe，并将 probe 权重方向转换回原始 residual stream 坐标，得到单位方向 `v_probe`。随后在推理时执行：

```text
h = h + alpha * v_probe
```

alpha sweep 结果如下：

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

结果文件：

```text
figures/steering_capital_probe_layer8.csv
figures/steering_capital_probe_layer8.png
figures/steering_capital_probe_layer8_accuracy.png
figures/steering_capital_probe_layer8_probe_accuracy.png
```

这个结果说明，steering 可以稳定控制内部 probe score，且 alpha 每增加 1，mean probe score 近似增加 1。但 naive global steering 对所有样本加入同一方向，true 和 false 样本会一起移动，因此没有提升输出层 true/false logit-sign accuracy。这是一个重要负结果：可读性不等于直接可控性。

### 6.2 Probe-direction ablation

为了进一步检验该方向是否承载可读信息，我们从 residual activation 中移除 probe direction 上的投影：

```text
h_ablated = h - strength * (h · v_probe) * v_probe
```

结果如下：

| Strength | Fixed Direction Score Gap | Fixed Direction Accuracy | Retrained Probe AUC |
|---:|---:|---:|---:|
| 0.00 | 0.573 | 0.826 | 0.953 |
| 0.25 | 0.430 | 0.500 | 0.955 |
| 0.50 | 0.286 | 0.500 | 0.958 |
| 0.75 | 0.143 | 0.500 | 0.951 |
| 1.00 | -0.000 | 0.500 | 0.945 |
| 1.25 | -0.143 | 0.500 | 0.953 |
| 1.50 | -0.286 | 0.478 | 0.962 |

结果文件：

```text
figures/ablation_capital_probe_layer8.csv
figures/ablation_capital_probe_layer8.png
figures/ablation_capital_probe_layer8_score_gap.png
```

单方向 ablation 能将原 probe direction 上的 score gap 从 0.573 降到约 0，这说明该方向确实承载可读的 true/false 信息。然而，在 ablated activations 上重新训练 probe 后，AUC 仍保持在 0.94 以上。这提示 capital truth/false 信息并不只存在于单一方向中，而可能分布在多个相关方向或子空间中。

## 7. 论文复现与拓展总结

本项目复现了 truth direction / geometry of truth 相关工作的几个核心思想：

| 论文思想 | 本项目复现 |
|---|---|
| True/false statement datasets | 构建 528 条英文事实判断样本 |
| Linear truth representation | 每层 residual stream linear probe |
| Layer localization | 分领域 sweep 和 focused capital probe |
| Visualization | PCA 激活空间可视化 |
| Causal intervention | Activation patching、steering、ablation |
| Generalization analysis | 比较不同领域和 prompt 的可分性 |

本项目的拓展主要在于：没有只报告成功结果，而是明确分析了三个负结果或局限。

第一，混合领域 truth direction 明显弱于 capital 单领域，说明 truth direction 的泛化并不自动成立。

第二，PCA 二维投影不能清楚分离 true/false，说明无监督高方差方向不一定等于可解释的 truth direction。

第三，steering 可以移动内部 probe score，但不能直接改善输出判断，说明“probe 可读”与“模型实际用于输出的因果机制”之间仍有距离。

## 8. 个人分析与局限

这组实验最重要的启发是：机制可解释性中的“现象定义”非常关键。如果把任务定义为“所有事实判断”，GPT-2-small 的通用 truth direction 并不强；但如果把任务收窄到结构一致的 capital fact verification，中后层 residual stream 中会出现很强的线性可分结构。这说明可解释性实验不能只问“模型有没有某个抽象概念”，还要问该概念在什么数据分布、prompt、层和模块中稳定出现。

另一个重要启发是：linear probe 的成功不能直接等价于因果解释。第 8 层 probe AUC 达到 0.953，但 naive steering 没有提升输出行为；单方向 ablation 移除一个 direction 后，重新训练 probe 仍然能恢复高 AUC。这说明模型内部信息可能是冗余和分布式的，单一方向只能解释其中一部分结构。

当前局限包括：

1. 模型只使用 GPT-2-small，尚未扩展到 Qwen2.5-0.5B 或 Qwen2.5-1.5B。
2. Activation patching 主要是 layer/module 级，还没有做 attention head 级 patching。
3. Steering 使用全局固定方向，没有做输入条件化 steering。
4. 数据集是人工构造的小规模事实判断集，虽然便于可复现，但覆盖面有限。

## 9. 结论

本项目完成了一个围绕事实判断的机制可解释性小闭环：先用 linear probe 定位 truth/false 信息，再用 PCA 和错误分析理解其表现，随后用 activation patching、steering 和 ablation 检验其因果与可控性。

结论可以概括为：

1. GPT-2-small 在结构一致的 capital fact verification 中存在强线性 truth/false 表征，最佳 AUC 为 0.953。
2. 该表征主要出现在中后层 residual stream 中。
3. Capital recall 的 activation patching 提供了更直接的因果证据，最后层 residual stream patching 可以恢复目标首都 logit。
4. Probe direction 可以被 vector arithmetic 控制，但 naive global steering 不能直接 improve 输出判断。
5. Truth/false 信息不是单一方向独占的，单方向 ablation 后仍可被重新训练 probe 读出，说明存在冗余子空间。

因此，本项目支持一个更谨慎的观点：truth direction 在特定结构化任务中确实可以被定位和干预，但它不是跨所有事实领域无条件稳定的单一按钮。机制可解释性实验需要同时报告成功定位、失败案例和负结果，才能避免把 probe 可读性过度解释为完整因果机制。

## 参考文献

1. Samuel Marks and Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*. arXiv:2310.06824. https://arxiv.org/abs/2310.06824
2. Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Zhengwen Feng, Hao Peng, Jianwei Yin. *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*. arXiv:2506.00823. https://arxiv.org/abs/2506.00823
3. Leonard Bereska and Efstratios Gavves. *Mechanistic Interpretability for AI Safety -- A Review*. arXiv:2404.14082.
4. *Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in LLMs*. arXiv 2026.
