# GPT-2-small 事实真假表征的可读性、可控性与局限

## 摘要

本项目围绕“大模型机制可解释性”课程要求，研究 GPT-2-small 在事实真假判断任务中是否存在可定位、可干预、可复现的 true/false 内部表征。我们构建了 528 条英文事实判断样本，覆盖 `capital`、`continent`、`element_symbol`、`book_author`、`landmark_country`、`science`、`math` 七个领域，并使用 TransformerLens 提取每层 residual stream 激活。

本文的核心结论经过刻意降调：我们没有证明存在一个跨领域稳定、可直接控制输出的全局 truth direction。更准确地说，GPT-2-small 在结构化的 capital fact verification 中存在强线性可读的 true/false 信息，layer 8 AUC 达到 0.953，layer 10 accuracy 达到 0.870；但混合领域上的 truth direction 明显更弱，PCA 二维投影不能清楚分离真假样本，probe-direction steering 只能移动内部 probe score，不能改善 true/false 输出判断。进一步的 probe-direction ablation 显示，移除已发现方向上的 score gap 后，重新训练 probe 仍能恢复 AUC 0.945 以上，说明该信息更像存在于冗余子空间中，而不是一个单一可控按钮。

因此，本项目支持的 thesis 是：

> GPT-2-small 在结构化 capital fact verification 中存在强线性可读的 true/false 信息；但该信息不是跨领域稳定的全局 truth direction，也不是可通过 naive global steering 直接改善输出的单一控制方向。

## 1. 研究问题

我们选择“事实真假判断”作为机制可解释性现象。给定句子：

```text
The capital of France is Paris.
```

我们关心模型内部是否包含与 true/false 标签相关的可读信息，以及这些信息是否能被因果干预或推理时 steering 控制。

具体问题如下：

1. GPT-2-small 哪些层的 residual stream 中存在可由 linear probe 读取的 true/false 信息？
2. 这种可读性是否跨事实领域和 prompt 形式稳定？
3. activation patching 能否为相关事实知识召回提供补充因果证据？
4. probe direction steering 是否能改变内部 probe score，并进一步改善输出行为？
5. 如果移除一个 probe direction，true/false 信息是否仍能从其它方向读出？

课程要求与本项目对应关系如下：

| 课程要求 | 本项目实现 |
|---|---|
| Locate | layer-wise linear probe、domain/prompt sweep、PCA、error analysis |
| Steer & Improve | probe-direction steering；结果显示 internal control without output improvement |
| 顶会/Arxiv 复现 | 以 Bao et al. 2025 为主复现对象，复现 truth direction probing、跨领域一致性分析和干预式验证思想 |

## 2. 背景与复现对象

### 2.1 Transformer 与 Hook 点

Transformer 的 residual stream 是每层之间传递和累积信息的主通道。对 GPT-2-small 而言，我们主要使用以下 TransformerLens hook 点：

```text
blocks.{layer}.hook_resid_post
blocks.{layer}.hook_attn_out
blocks.{layer}.hook_mlp_out
```

其中 `hook_resid_post` 用于读取每层输出后的 residual activation，`hook_attn_out` 和 `hook_mlp_out` 用于模块级 activation patching。实验默认使用 final token position 的 activation，因为该位置最接近模型在 prompt 之后做预测时可用的上下文表示。

### 2.2 主复现对象：Bao et al. 2025

本项目的主复现对象是 Bao et al. 2025 的 *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*。Marks and Tegmark 2023 的 *The Geometry of Truth* 作为背景工作，因为它提出了 true/false 激活空间线性结构这一核心视角，但其年份不属于课程要求的 2024-2026 范围。

我们复现 Bao et al. 2025 的思路，而不是逐像素复现某个 figure。对应关系如下：

| Bao et al. 2025 思路 | 本项目实现 | 结果 |
|---|---|---|
| 构造 true/false statement datasets | 528 条英文事实判断样本 | balanced true/false，覆盖 7 个领域 |
| layer-wise truth probing | 对每层 residual stream 训练 linear probe | capital layer 8 AUC 0.953 |
| consistency/generalization | 比较 domain 与 prompt sweep | mixed-domain 弱，capital 强 |
| geometry visualization | PCA 可视化 activation | 二维 PCA 不足以解释 probe 可分性 |
| intervention check | steering 与 ablation | 内部 score 可控，但输出未改善；信息不局限于单方向 |

这个复现结果更偏向“限定条件下复现与反例分析”：我们复现了结构化任务中的强线性可读性，同时观察到跨领域一致性和可控性都有限。

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

## 4. Locate: 线性可读性定位

### 4.1 分领域与 prompt sweep

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

### 4.2 Focused Capital Probe

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

### 4.3 PCA 辅助可视化

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

### 4.4 Error Analysis

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

这些错误提醒我们，probe 的高 AUC 不意味着每个事实判断都被稳健掌握。模型可能对部分国家、实体长度、tokenization 或训练语料频率敏感。

## 5. Activation Patching: Capital Recall 的补充因果实验

这一节必须明确限定：本项目的 activation patching 不是 true/false verification 的直接因果定位，而是对相关 capital fact knowledge recall 的补充因果实验。它回答的是“首都知识召回信息在模型哪些层/模块中起作用”，而不是直接回答“模型如何判断 statement 为 true 或 false”。

实验 prompt：

```text
clean:   The capital of France is
corrupt: The capital of Germany is
metric:  logit(" Paris") - logit(" Berlin")
```

我们在 corrupt prompt 前向传播时，把 clean prompt 的同层同位置 activation patch 进去，并观察目标首都 token logit difference 是否恢复。

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

解释：

- 在 capital recall 任务中，最后层 residual stream patching 可以恢复到 clean baseline，说明后层 residual stream 对目标首都 token 输出有直接因果作用。
- `attn_out` layer 11 的 mean recovery 很高，但 patched logit diff 仍为负，说明它可能贡献了强恢复信号，但单独 patch 后还不足以稳定翻转目标 logit。
- 该结果不能直接外推为 true/false verification 的 causal localization。报告中后续所有相关结论都只把它称作 supplementary causal evidence for capital recall。

## 6. Steering and Ablation: Internal Control without Output Improvement

### 6.1 Held-out Probe-direction Steering

原始版本的 steering 实验有一个口径问题：direction 和 threshold 用全量 capital 数据拟合，再在同一批数据上报告 probe-threshold accuracy，导致 `alpha=0` 的 accuracy 过高。现在脚本已改为 group split：

- train split: 106 rows，用于拟合 `v_probe` 和 `probe_score_threshold`；
- test split: 46 rows，用于评估 steering 后的 logit-sign accuracy 和 probe-threshold accuracy；
- threshold source: `train_midpoint`，即 train true/false projection mean 的中点。

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

更新后的结果：

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

现在 `alpha=0` 的 held-out probe-threshold accuracy 为 0.826，与 focused capital probe 的 layer 8 held-out accuracy 对齐。这个结果说明：

1. 沿 probe direction 加法可以线性移动内部 probe score；
2. 但所有样本几乎一起平移，所以阈值附近的分类边界很容易被整体推坏；
3. true/false logit-sign accuracy 始终为 0.500，说明 naive global steering 没有改善输出行为。

因此，本节不应被解读为 Improve 成功，而应被解读为：

> probe direction is internally steerable, but this steering fails to improve output behavior.

### 6.2 Probe-direction Ablation

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

这是本项目最关键的负结果之一。移除当前 probe direction 后，该方向上的 score gap 从 0.573 降到约 0，fixed direction accuracy 退化到 0.500；但如果在 ablated activation 上重新训练 probe，AUC 仍保持在 0.945 以上。

这说明我们发现的不是唯一 truth direction，而是一个线性可读方向。true/false 信息可能分布在多个相关方向或子空间中。该结论比“我找到了 truth direction”更稳，也更符合机制可解释性中对 distributed representation 的谨慎理解。

## 7. 综合讨论

本项目的实验链条可以概括为：

```text
strong capital linear readability
-> weak mixed-domain generalization
-> PCA does not reveal a simple 2D separation
-> related capital recall can be patched in late residual stream
-> steering moves internal score without output improvement
-> ablation removes one direction but not all readable information
-> true/false information is readable but not a single controllable direction
```

因此，本文最重要的结论不是“truth direction 可以直接 steering 成功”，而是：

> 可读性强，不代表单方向因果控制强；capital truth/false 信息更像冗余子空间现象，而不是一个全局稳定按钮。

这也解释了为什么 mixed-domain 结果弱而 capital 结果强。truth/false 不是脱离任务内容的抽象二值属性；它依赖事实类型、prompt 形式、实体分布和模型对相关知识的掌握程度。

## 8. 局限

1. 模型只使用 GPT-2-small，尚未扩展到 Qwen2.5-0.5B 或 Qwen2.5-1.5B。
2. activation patching 针对的是 related capital recall，不是 true/false verification 的直接 patching。
3. patching 粒度主要是 layer/module，没有做 attention head 或 neuron 级定位。
4. steering 使用全局固定方向，没有做输入条件化 steering 或多方向 subspace steering。
5. 数据集为人工构造的小规模事实判断集，虽然可复现，但仍可能存在模板、实体频率和 tokenization 偏差。

## 9. 结论

本项目完成了一个较完整的机制可解释性小闭环：先用 linear probe 定位 true/false 信息，再用 domain/prompt sweep 和 PCA 检验其稳定性与几何结构，随后用 capital recall patching、probe-direction steering 和 ablation 检验其因果与可控性边界。

最终结论如下：

1. GPT-2-small 在结构化 capital fact verification 中存在强线性可读的 true/false 信息，layer 8 AUC 为 0.953。
2. 这种信号并不构成跨领域稳定的全局 truth direction，mixed-domain separability 明显更弱。
3. PCA 不能清楚分离 true/false，说明高维 supervised direction 与低维高方差方向不同。
4. Capital recall patching 为相关事实知识召回提供补充因果证据，但不是 truth verification 的直接因果定位。
5. Probe-direction steering 可以移动内部 probe score，但 without output improvement。
6. Probe-direction ablation 表明，true/false 信息不局限于单一方向，更可能存在于 redundant subspace 中。

## 参考文献

1. Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Zhengwen Feng, Hao Peng, Jianwei Yin. *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*. arXiv:2506.00823. https://arxiv.org/abs/2506.00823
2. Leonard Bereska and Efstratios Gavves. *Mechanistic Interpretability for AI Safety -- A Review*. arXiv:2404.14082.
3. *Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in LLMs*. arXiv 2026.
4. Samuel Marks and Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*. arXiv:2310.06824. https://arxiv.org/abs/2310.06824
