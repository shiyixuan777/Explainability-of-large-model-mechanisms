# GPT-2-small 事实真假表征的可读性、干预效果与冗余子空间

## 摘要

本项目围绕“大模型机制可解释性”课程要求，研究 GPT-2-small 在事实真假判断任务中是否存在可定位、可干预、可复现的 true/false 内部表征。我们构建了 528 条英文事实判断样本，覆盖 `capital`、`continent`、`element_symbol`、`book_author`、`landmark_country`、`science`、`math` 七个领域，并使用 TransformerLens 提取每层 residual stream 激活。

本文的核心结论经过刻意降调：我们没有证明存在一个跨领域稳定、可直接控制输出的全局 truth direction。更准确地说，GPT-2-small 在结构化的 capital fact verification 中存在强线性可读的 true/false 信息，layer 8 AUC 达到 0.953，layer 10 accuracy 达到 0.870；但混合领域上的 truth direction 明显更弱，PCA 二维投影不能清楚分离真假样本，naive global steering 只能移动内部 probe score，不能改善 true/false 输出判断。

进一步的干预实验给出了更关键的结论。第一，true/false verification residual patching 显示后层 residual stream 能恢复 clean/corrupt 之间的 truth-logit 差异，layer 11 mean recovery 为 1.000，但平均绝对 logit shift 只有 0.076，说明该通道存在但输出影响较弱。第二，oracle conditional steering 可以把 held-out probe accuracy 从 0.826 提升到 1.000，但 true/false logit-sign accuracy 仍保持 0.500。第三，probe-direction ablation 能把已发现方向上的 score gap 从 0.573 降到约 0，但在 ablated activations 上重新训练 probe 后，AUC 仍保持在 0.945 以上。

因此，本项目支持的 thesis 是：

> GPT-2-small 在结构化 capital fact verification 中存在强线性可读的 true/false 信息；但该信息不是跨领域稳定的全局 truth direction，也不是可通过单一方向直接改善输出的控制按钮。它更接近一个可线性读取、可局部干预、但具有冗余性的子空间现象。

## 1. 研究问题

本项目选择“事实真假判断”作为机制可解释性现象。给定句子：

```text
The capital of France is Paris.
```

我们关心模型内部是否包含与 true/false 标签相关的可读信息，以及这些信息是否能被 activation patching、steering 或 ablation 干预。

具体问题如下：

1. GPT-2-small 哪些层的 residual stream 中存在可由 linear probe 读取的 true/false 信息？
2. 这种可读性是否跨事实领域和 prompt 形式稳定？
3. true/false verification 本身能否通过 residual stream patching 获得因果定位证据？
4. probe direction steering 是否能改变内部 probe score，并进一步改善输出行为？
5. 如果移除一个 probe direction，true/false 信息是否仍能从其它方向读出？

课程要求与本项目对应关系如下：

| 课程要求 | 本项目实现 |
|---|---|
| Locate | layer-wise linear probe、domain/prompt sweep、PCA、error analysis、truth verification residual patching |
| Steer & Improve | naive global steering 与 oracle conditional steering；后者改善内部 probe 指标，但仍不改善输出 logit-sign accuracy |
| 顶会/Arxiv 复现 | 以 Bao et al. 2025 为主复现对象，做 partial reproduction：truth direction probing、跨领域一致性分析和干预式验证 |

## 2. 背景与复现对象

### 2.1 Transformer 与 Hook 点

Transformer 的 residual stream 是每层之间传递和累积信息的主通道。对 GPT-2-small 而言，我们主要使用以下 TransformerLens hook 点：

```text
blocks.{layer}.hook_resid_post
blocks.{layer}.hook_attn_out
blocks.{layer}.hook_mlp_out
```

其中 `hook_resid_post` 用于读取每层输出后的 residual activation，`hook_attn_out` 和 `hook_mlp_out` 用于模块级 activation patching。实验默认使用 final token position 的 activation，因为该位置最接近模型在 prompt 之后做预测时可用的上下文表示。

### 2.2 Partial Reproduction: Bao et al. 2025

本项目的主复现对象是 Bao et al. 2025 的 *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*。Marks and Tegmark 2023 的 *The Geometry of Truth* 作为背景工作，因为它提出了 true/false 激活空间线性结构这一核心视角，但其年份不属于课程要求的 2024-2026 范围。

需要明确的是，本文属于 partial reproduction，而不是 full reproduction。由于算力和时间限制，我们不复现原论文的大模型、多任务完整设置，而是复现一个最小实验闭环：

1. 构造 true/false statement dataset；
2. 对 residual stream 做 layer-wise linear probing；
3. 检查不同 domain/prompt 中 truth direction 的一致性；
4. 通过 steering、patching 和 ablation 检验方向的干预效果与局限。

对应关系如下：

| Bao et al. 2025 思路 | 本项目实现 | 结果 |
|---|---|---|
| True/false statement datasets | 528 条英文事实判断样本 | balanced true/false，覆盖 7 个领域 |
| Layer-wise truth probing | 对每层 residual stream 训练 linear probe | capital layer 8 AUC 0.953 |
| Consistency/generalization | 比较 domain 与 prompt sweep | mixed-domain 弱，capital 强 |
| Geometry visualization | PCA 可视化 activation | 二维 PCA 不足以解释 probe 可分性 |
| Intervention check | steering、patching、ablation | 内部指标可控，输出未改善；信息不局限于单方向 |

这个 partial reproduction 的价值在于：它复现了结构化任务中的强线性可读性，同时展示了跨领域一致性、直接输出控制和单方向定位都存在明显限制。

## 3. 数据与实验设置

数据集由 `scripts/build_dataset.py` 生成，保存为：

```text
data/facts.csv
```

数据规模如下：

| 属性 | 数值 |
|---|---:|
| 总样本数 | 528 |
| True 样本 | 264 |
| False 样本 | 264 |
| 领域数 | 7 |

领域分布：

| Domain | Rows |
|---|---:|
| capital | 152 |
| continent | 86 |
| element_symbol | 80 |
| book_author | 60 |
| landmark_country | 60 |
| science | 50 |
| math | 40 |

为了避免同一个事实 pair 的 true/false 两个版本同时进入训练集和测试集，probe、steering、ablation 默认使用 `pair_id` 做 group split。这一点很重要，因为否则 probe 可能利用实体或模板重复，而不是泛化到 held-out fact pairs。

主要 prompt 形式包括：

```text
Statement: {statement}
The statement is
```

以及：

```text
Statement: {statement}
Answer true or false:
```

## 4. Methods

### 4.1 Activation Extraction

所有 layer-wise probe 和 steering 实验都读取 final token position 的 `hook_resid_post` activation。GPT-2-small 共有 12 层，层编号为 0-11。

### 4.2 Linear Probe

linear probe 使用 `StandardScaler + LogisticRegression(max_iter=2000, class_weight="balanced")`。如果实验涉及同一事实 pair 的 true/false 两个版本，则使用 `GroupShuffleSplit(test_size=0.3, random_state=42)`，按 `pair_id` 划分训练集和测试集。

### 4.3 Steering

probe-direction steering 先在 train split 上训练 logistic regression probe，再把标准化空间中的 probe weight 转回原始 activation 坐标，并归一化为单位向量 `v_probe`。naive global steering 对所有 test 样本执行：

```text
h = h + alpha * v_probe
```

oracle conditional steering 使用标签作为诊断信号，对 true 样本加 `+alpha * v_probe`，对 false 样本加 `-alpha * v_probe`。它不是可部署的推理方法，而是 controlled intervention，用来检验方向本身能否按预期改变内部 probe 指标。

### 4.4 Ablation

probe-direction ablation 从 activation 中移除当前 probe direction 的投影：

```text
h_ablated = h - strength * (h · v_probe) * v_probe
```

如果移除该方向后重新训练 probe 仍能恢复高 AUC，则说明 true/false 信息不局限于单一方向。

### 4.5 Patching Metrics

activation patching 使用 clean/corrupt prompt。对于 truth verification patching，metric 为：

```text
logit(" true") - logit(" false")
```

recovery 定义为：

```text
(patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)
```

## 5. Locate: 线性可读性定位

### 5.1 分领域与 Prompt Sweep

我们首先在不同 domain 与 prompt template 上做 layer-wise probe sweep。每个设置下，对每层 residual stream final-token activation 训练 logistic regression probe，并报告 held-out group split 上的 accuracy、AUC 和 separability AUC。

输出文件：

```text
figures/probe_sweep.csv
figures/probe_sweep_summary.png
```

主要结果：

| Domain | Prompt | Best Layer | Accuracy | AUC |
|---|---|---:|---:|---:|
| capital | answer | 8 | 0.826 | 0.953 |
| capital | statement_is | 6 | 0.826 | 0.940 |
| capital | question | 7 | 0.783 | 0.932 |
| continent | statement_is | 11 | 0.692 | 0.846 |
| element_symbol | question | 4 | 0.818 | 0.793 |
| landmark_country | statement_is | 11 | 0.706 | 0.813 |
| all | question | 11 | 0.588 | 0.653 |

解释上必须谨慎：高 AUC 说明该层 activation 中存在可被监督线性分类器读取的 true/false 相关信息，但不等同于“模型已经在该层形成了完整真假判断机制”。更准确的表述是：

> 第 8-10 层 residual stream 中存在强 true/false linear readability，尤其在结构一致的 capital fact verification 中。

mixed-domain 结果弱也不应简单解释为“模型没有 truth”。更合理的解释是：人类标签 true/false 过于抽象，混合了多种不同判断机制。capital 是 country-capital relation，element_symbol 是化学符号关系，book_author 是作者关系，math 是符号或算术正确性，science 是概念事实。它们共享 true/false 标签，但模型内部未必共享单一表征。

### 5.2 Focused Capital Probe

由于 sweep 显示 capital domain 最稳定，我们进一步聚焦：

```powershell
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_answer.csv
```

结果文件：

```text
figures/probe_capital_answer.csv
figures/probe_capital_answer.png
```

关键层结果：

| Layer | Accuracy | AUC |
|---:|---:|---:|
| 5 | 0.848 | 0.941 |
| 8 | 0.826 | 0.953 |
| 10 | 0.870 | 0.947 |
| 11 | 0.783 | 0.902 |

这说明 capital 任务中，中后层 residual stream 的线性可读性很强。但它仍然是 probe evidence，不是直接 causal evidence。

### 5.3 PCA 辅助可视化

我们对 layer 8 capital residual activation 做 PCA：

```text
figures/pca_capital_layer8.csv
figures/pca_capital_layer8.png
```

结果：

| Layer | PC1 Explained Variance | PC2 Explained Variance | Rows |
|---:|---:|---:|---:|
| 8 | 0.620 | 0.117 | 152 |

二维 PCA 图没有清楚分离 true/false。这不是与 probe AUC 矛盾，而是说明 probe 发现的方向不一定是解释方差最大的无监督主成分。换句话说，真假可读性更像存在于 supervised high-dimensional direction 或子空间中，而不是 PCA 前两个方向中。

### 5.4 Error Analysis

我们对 layer 8 capital probe 做错误样本分析：

```text
figures/error_analysis_capital_layer8.csv
figures/error_analysis_capital_layer8_errors.csv
```

测试集结果：

| Test Rows | Correct | Wrong | Accuracy |
|---:|---:|---:|---:|
| 46 | 38 | 8 | 0.826 |

典型错误包括：

| Statement | Label | Prediction | Prob True |
|---|---|---|---:|
| The capital of Laos is Vientiane. | true | false | 0.003 |
| The capital of Canada is Amman. | false | true | 0.964 |
| The capital of Chile is Santiago. | true | false | 0.179 |
| The capital of India is New Delhi. | true | false | 0.217 |

错误大致可以分为三类：

1. 低频国家或首都的 true 样本被判为 false，例如 Laos-Vientiane、Morocco-Rabat、Kenya-Nairobi。
2. 带有 plausible capital token 的 false 样本被判为 true，例如 Canada-Amman、Nigeria-Mexico City。
3. 多 token 首都或 tokenization 更复杂的样本，例如 New Delhi、Sri Jayawardenepura Kotte。

这说明 final-token activation 的 linear probe 可能同时受到事实知识、实体熟悉度、模板规律和 tokenization granularity 的混合影响。

## 6. Activation Patching

### 6.1 Capital Recall Patching

第一组 patching 是 related capital recall 实验，用来定位首都知识召回信息。它不是 true/false verification 的直接因果定位。

实验 prompt：

```text
clean:   The capital of France is
corrupt: The capital of Germany is
metric:  logit(" Paris") - logit(" Berlin")
```

输出文件：

```text
figures/activation_patching_capital_recall.csv
figures/activation_patching_capital_recall.png
```

模块级最佳结果：

| Component | Best Layer | Patched Logit Diff | Mean Recovery | Median Recovery |
|---|---:|---:|---:|---:|
| resid_post | 11 | 0.264 | 1.000 | 1.000 |
| attn_out | 11 | -0.672 | 1.762 | 1.077 |
| mlp_out | 7 | -1.022 | 0.388 | 0.074 |

在 capital recall 任务中，最后层 residual stream patching 可以恢复到 clean baseline，说明后层 residual stream 对目标首都 token 输出有直接因果作用。这个结果只能作为 related factual recall 的补充因果证据。

### 6.2 Truth Verification Residual Patching

为了让 patching 更贴近主任务，我们补充了 true/false verification residual patching。实验构造同一 `pair_id` 下的 true statement 与 false statement：

```text
clean:   Statement: The capital of France is Paris.
         Answer true or false:

corrupt: Statement: The capital of France is Berlin.
         Answer true or false:

metric:  logit(" true") - logit(" false")
```

这里只做 `hook_resid_post` layer-wise patching。输出文件：

```text
figures/truth_verification_patching_resid.csv
figures/truth_verification_patching_resid.png
figures/truth_verification_patching_resid_logit_shift.png
```

关键结果：

| Layer | Mean Recovery | Median Recovery | Patched Logit Diff | Mean Abs Logit Shift |
|---:|---:|---:|---:|---:|
| 8 | 0.568 | 0.541 | 1.536 | 0.052 |
| 9 | 0.607 | 0.622 | 1.538 | 0.055 |
| 10 | 0.816 | 0.815 | 1.540 | 0.068 |
| 11 | 1.000 | 1.000 | 1.547 | 0.076 |

这个结果比上一组 capital recall patching 更贴近 true/false verification。它说明后层 residual stream 确实携带了能恢复 clean/corrupt truth-logit difference 的信息。但平均绝对 logit shift 很小，且 clean/corrupt 本身的 mean true-minus-false logit diff 都为正，这与后续 steering 的失败一致：GPT-2-small 的输出层并没有稳定地把该内部可读信息转化为 true/false 行为。

## 7. Steering and Ablation

### 7.1 Naive Global Steering: Internal Control without Output Improvement

naive global steering 使用 group split：

- train split: 106 rows，用于拟合 `v_probe` 和 `probe_score_threshold`；
- test split: 46 rows，用于评估 steering 后的 logit-sign accuracy 和 probe-threshold accuracy；
- threshold source: `train_midpoint`。

实验操作：

```text
h = h + alpha * v_probe
```

输出文件：

```text
figures/steering_capital_probe_layer8.csv
figures/steering_capital_probe_layer8.png
figures/steering_capital_probe_layer8_accuracy.png
figures/steering_capital_probe_layer8_probe_accuracy.png
```

结果：

| Alpha | Logit-sign Accuracy | Held-out Probe-threshold Accuracy | Mean Probe Score |
|---:|---:|---:|---:|
| -8 | 0.500 | 0.500 | -9.560 |
| -4 | 0.500 | 0.522 | -5.560 |
| -2 | 0.500 | 0.522 | -3.560 |
| -1 | 0.500 | 0.522 | -2.560 |
| 0 | 0.500 | 0.826 | -1.560 |
| 1 | 0.500 | 0.500 | -0.560 |
| 2 | 0.500 | 0.500 | 0.440 |
| 4 | 0.500 | 0.500 | 2.440 |
| 8 | 0.500 | 0.500 | 6.440 |

`alpha=0` 的 held-out probe-threshold accuracy 为 0.826，与 focused capital probe 的 layer 8 held-out accuracy 对齐。沿 probe direction 加法可以线性移动内部 probe score，但所有样本一起平移，因此容易破坏阈值分类。true/false logit-sign accuracy 始终为 0.500，说明 naive global steering 没有改善输出行为。

### 7.2 Oracle Conditional Steering: 内部指标可 Improve，但输出仍不改善

为了回应课程中 “Steer & Improve” 的要求，我们补充 oracle conditional steering。该实验不是实际推理方法，因为它使用标签决定方向；它是一个 diagnostic intervention，用来检查方向本身能否按预期改善内部指标。

操作如下：

```text
true sample:  h = h + alpha * v_probe
false sample: h = h - alpha * v_probe
```

输出文件：

```text
figures/oracle_steering_capital_probe_layer8.csv
figures/oracle_steering_capital_probe_layer8.png
figures/oracle_steering_capital_probe_layer8_margins.png
```

结果：

| Alpha | Logit-sign Accuracy | Probe-threshold Accuracy | Probe Correct Margin | Logit Correct Margin |
|---:|---:|---:|---:|---:|
| 0.0 | 0.500 | 0.826 | 0.286 | -0.025 |
| 0.5 | 0.500 | 1.000 | 0.786 | -0.029 |
| 1.0 | 0.500 | 1.000 | 1.286 | -0.033 |
| 2.0 | 0.500 | 1.000 | 2.286 | -0.041 |
| 4.0 | 0.500 | 1.000 | 4.286 | -0.057 |
| 8.0 | 0.500 | 1.000 | 8.286 | -0.088 |

这个实验给出了一个更精确的 Improve 结论：在 oracle-controlled setting 下，probe-threshold accuracy 和 probe correct margin 可以被明显改善；但 true/false logit-sign accuracy 仍保持 0.500，logit correct margin 甚至略微变差。这说明 probe direction 是内部可控方向，但它没有与模型输出 true/false logits 直接对齐。

### 7.3 Probe-direction Ablation

为了检验 true/false 信息是否局限于一个方向，我们从 activation 中移除 probe direction 上的投影：

```text
h_ablated = h - strength * (h · v_probe) * v_probe
```

输出文件：

```text
figures/ablation_capital_probe_layer8.csv
figures/ablation_capital_probe_layer8.png
figures/ablation_capital_probe_layer8_score_gap.png
```

结果：

| Strength | Fixed Direction Score Gap | Fixed Direction Accuracy | Retrained Probe AUC |
|---:|---:|---:|---:|
| 0.00 | 0.573 | 0.826 | 0.953 |
| 0.25 | 0.430 | 0.500 | 0.955 |
| 0.50 | 0.286 | 0.500 | 0.958 |
| 0.75 | 0.143 | 0.500 | 0.951 |
| 1.00 | -0.000 | 0.500 | 0.945 |
| 1.25 | -0.143 | 0.500 | 0.953 |
| 1.50 | -0.286 | 0.478 | 0.962 |

这是本项目最关键的结果之一。移除当前 probe direction 后，该方向上的 score gap 从 0.573 降到约 0，fixed direction accuracy 退化到 0.500；但如果在 ablated activation 上重新训练 probe，AUC 仍保持在 0.945 以上。

这说明我们发现的不是唯一 truth direction，而是一个线性可读方向。true/false 信息可能分布在多个相关方向或冗余子空间中。这个负结果比单纯报告 AUC 更有机制解释价值。

## 8. 综合讨论

本项目的实验链条可以概括为：

```text
capital 领域具有强线性可读性
-> 混合领域泛化明显变弱
-> PCA 二维投影不能解释 probe 可分性
-> capital recall 和 truth verification 的后层 residual patching 都有恢复信号
-> naive global steering 不能改善输出
-> oracle steering 能改善内部 probe 指标但仍不改善输出
-> ablation 移除单方向后仍可重新读出 true/false 信息
-> true/false information 更像冗余子空间，而不是单一 truth button
```

因此，本文最重要的结论不是“truth direction 可以直接 steering 成功”，而是：

> 可读性强，不代表单方向因果控制强；capital truth/false 信息更像 redundant subspace 现象，而不是一个全局稳定按钮。

这也解释了为什么 mixed-domain 结果弱而 capital 结果强。truth/false 不是脱离任务内容的抽象二值属性；它依赖事实类型、prompt 形式、实体分布和模型对相关知识的掌握程度。人类标签 true/false 可能混合了多个领域特定机制，未必对应模型内部的单一表征。

## 9. 局限

1. 模型只使用 GPT-2-small，尚未扩展到 Qwen2.5-0.5B 或 Qwen2.5-1.5B。
2. true/false verification patching 目前只做了 layer-wise `resid_post`，尚未做到 attention head 或 neuron 级别。
3. oracle conditional steering 使用真实标签，不是可部署推理方法，只能作为诊断实验。
4. steering 使用单一固定方向，没有探索输入条件化的无监督策略或多方向 subspace steering。
5. 数据集为人工构造的小规模事实判断集，虽然可复现，但仍可能存在模板、实体频率和 tokenization 偏差。

## 10. 结论

本项目完成了一个较完整的机制可解释性小闭环：先用 linear probe 定位 true/false 信息，再用 domain/prompt sweep、PCA 和 error analysis 检验其稳定性与错误模式，随后用 capital recall patching、truth verification patching、naive steering、oracle steering 和 ablation 检验其因果与可控性边界。

最终结论如下：

1. GPT-2-small 在结构化 capital fact verification 中存在强线性可读的 true/false 信息，layer 8 AUC 为 0.953。
2. 这种信号并不构成跨领域稳定的全局 truth direction，mixed-domain separability 明显更弱。
3. PCA 不能清楚分离 true/false，说明高维 supervised direction 与低维高方差方向不同。
4. Truth verification residual patching 显示后层 residual stream 存在恢复信号，但输出 true/false logits 的平均变化很小。
5. Oracle conditional steering 可以改善内部 probe 指标，但仍然 without output improvement。
6. Probe-direction ablation 表明，true/false 信息不局限于单一方向，更可能存在于 redundant subspace 中。

## 参考文献

1. Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Zhengwen Feng, Hao Peng, Jianwei Yin. *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*. arXiv:2506.00823. https://arxiv.org/abs/2506.00823
2. Leonard Bereska and Efstratios Gavves. *Mechanistic Interpretability for AI Safety -- A Review*. arXiv:2404.14082.
3. Hengyuan Zhang, Zhihao Zhang, Mingyang Wang, Zunhai Su, Yiwei Wang, Qianli Wang, Shuzhou Yuan, Ercong Nie, Xufeng Duan, Qibo Xue, Zeping Yu, Chenming Shang, Xiao Liang, Jing Xiong, Hui Shen, Chaofan Tao, Zhengwu Liu, Senjie Jin, Zhiheng Xi, Dongdong Zhang, Sophia Ananiadou, Tao Gui, Ruobing Xie, Hayden Kwok-Hay So, Hinrich Schütze, Xuanjing Huang, Qi Zhang, Ngai Wong. *Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models*. arXiv:2601.14004. https://arxiv.org/abs/2601.14004
4. Samuel Marks and Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*. arXiv:2310.06824. https://arxiv.org/abs/2310.06824
