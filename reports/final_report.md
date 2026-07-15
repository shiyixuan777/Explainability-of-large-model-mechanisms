# GPT-2-small 事实配对标签信号的线性可分性、补全兼容度与干预边界

## 摘要

本文研究一个容易被过度解释的问题：GPT-2-small 的 residual stream 中，是否存在可线性读出的事实配对标签相关信号？如果存在，它更像抽象事实表征，还是更像数据构造、实体熟悉度、补全兼容度等混合因素形成的 verification-associated direction？

实验从首都事实开始。原始 capital 数据上，final-token residual probe 的 layer 8 AUC 达到 0.953；但 bag-of-words surface baseline 的方向无关 AUC 也达到 0.933，说明原始高分受到明显词汇分布伪线索影响。为排除最直接的 unigram 边际频率线索，本文构造词汇平衡 capital 数据集：每个二国 block 中，两个国家名和两个首都名都同时出现在 true 与 false 句子里。此时 bag-of-words 和 numeric surface baseline 都降为 0.500，而 residual probe 仍在 layer 6 保留 0.809 AUC，多 seed mean AUC 为 0.813。

这说明简单词袋线索不是全部解释，但也不等于发现 truth direction。completion margin 实验显示，同一 held-out split 上，total logprob AUC 为 0.861；但按 completion token 数归一化后，avg-token AUC 降到 0.786，低于 residual probe 的 0.809，且 block bootstrap CI 高度重叠。因此，completion margin 更适合作为 completion-compatibility-related signal，而不是事实知识的直接度量。

在当前主 split 上，沿 balanced layer 6 learned verification-associated direction 做 prompt-final-only completion-margin steering，可将 held-out avg-token margin 推动约 +0.135。learned effect 超过全部已采样的 50 条 random directions 和 20 条 label-permutation directions；相应经验 p 值受 sampled null 数量限制，分辨率分别约为 0.020 和 0.048。进一步的 repeated group split steering 中，10/10 个 split 的 learned shift 均为正，并且均大于该 split 的 random/permutation control 均值；aggregate learned-minus-random mean 为 +0.125，learned-minus-permutation mean 为 +0.119。

但这个效果仍然有限。当前主 split 上 pairwise preference accuracy 没有改变，sign flip 为 0；repeated splits 中 baseline pairwise accuracy 为 0.700，steered 后为 0.725，平均只提升 +0.025。candidate-set rank 检查中，正确首都平均 rank 从 15.04 小幅改善到 14.13，top-1 accuracy 仅从 0.083 到 0.125，只改变 1 个 top candidate。decomposition 显示 correct completion avg-token logprob 上升 +0.280，false completion 也上升 +0.147；shared uplift 约为 +0.214，correct-over-wrong differential 为 +0.133。因此，本文最终支持的结论是：

> GPT-2-small 在词汇平衡首都事实中存在一个 fact-pair label signal；它与补全兼容度相关，并能弱影响 correct-vs-selected-wrong completion scoring margin，但尚不能稳定改变 held-out correct-vs-selected-wrong pairwise choice，也尚不能被解释为跨领域稳定、可直接控制输出的全局 truth direction。

## 1. 问题定义

本文把问题拆成四层：

1. residual activation 中是否存在可线性读出的 fact-pair label signal；
2. 这种可分性是否只是词汇、模板或数据构造伪线索；
3. 剩余信号是否与模型自身的补全偏好一致；
4. 沿这种信号方向干预是否会产生下游 score-level 或 choice-level 影响。

probe AUC 高只能说明“某个方向能读出标签”，并不能说明模型真的在用这个方向判断事实，更不能说明找到了可部署的 truth button。后续实验都围绕同一个目标展开：把强 probe 结果分解成数据线索、completion-compatibility-related signal、跨域泛化和干预边界。

## 2. 与 Bao et al. 的 Reproduction Fidelity

本文参考 Bao et al. 2025, *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*。该论文关心 truthfulness probes 是否能跨任务、逻辑变换、问答形式和知识源泛化，并提醒 truth direction 的解释需要经过迁移与行为检验。

本文不是完整复现 Bao et al. 的全部设置，而是做一个小模型受控复现与扩展：

| 类型 | 研究问题 | 本文对应实验 | 结论关系 |
|---|---|---|---|
| Bao et al. 对照 | hidden state 中能否训练 truthfulness probe | GPT-2-small final-token `resid_post` probe | 复现线性可读现象，但只限人工事实验证数据 |
| Bao et al. 对照 | truth direction 是否跨任务泛化 | domain transfer 与 direction cosine | 未提供统一方向证据；地理相关任务有局部共享结构 |
| Bao et al. 对照 | probe 能否用于下游行为 | completion-margin steering、output readout baseline | 能弱移动补全评分，不能改善 pairwise choice |
| 本文扩展 | 高 probe AUC 是否受数据伪线索影响 | surface baseline、balanced dataset、completion margin | 强化 confound 诊断，说明高 AUC 很容易来自数据结构 |

没有复现的部分包括原论文完整 logical transformation、QA、ICL、external knowledge 数据、多模型比较和 selective QA 应用。因此，本文的复现定位是：在 GPT-2-small 上复现 truth-direction generalization 的核心问题意识，并通过更强的 surface/balanced/steering 控制实验检查其解释边界。

## 3. 数据与设置

原始数据集 `data/facts.csv` 含 528 条英文事实验证样本，覆盖 capital、continent、element_symbol、book_author、landmark_country、science、math 七类。主线使用首都事实，因为它既能形成清晰的 correct-vs-wrong completion 评价，又容易构造词汇平衡对照。

词汇平衡数据集 `data/capital_balanced.csv` 使用二国二首都闭合 block：

```text
France  - Paris   true
France  - Berlin  false
Germany - Berlin  true
Germany - Paris   false
```

这样国家名和首都名在 true/false 标签中的边际频率完全平衡。该数据集共有 152 行、38 个 block；当前主 split 的 held-out 部分为 12 个 block / 48 行。数据中存在少量有定义或政治语境差异的事实，例如 `Israel -> Jerusalem`、`South Africa -> Pretoria`、`Bolivia -> Sucre`。本文后续用 ambiguous-fact sensitivity 删除相关 block 后重新划分，检查信号和干预效应是否定性保留。

## 4. Probe 到底读到了什么

原始 capital probe 很强：

| Layer | Accuracy | AUC |
|---:|---:|---:|
| 8 | 0.826 | 0.953 |
| 10 | 0.870 | 0.947 |
| 5 | 0.848 | 0.941 |

seed sensitivity 显示 layer 8 在六个 group split 上 mean AUC 为 0.899，范围 0.832-0.953，说明这不是单个 seed 的偶然结果。

但 surface baseline 暴露了核心 confound：

| Dataset | Baseline | Accuracy | AUC | Direction-agnostic AUC |
|---|---|---:|---:|---:|
| original capital | numeric_surface | 0.500 | 0.500 | 0.500 |
| original capital | bag-of-words | 0.065 | 0.067 | 0.933 |

这里的 direction-agnostic AUC 定义为 `max(AUC, 1-AUC)`，只用于诊断预测分数与标签之间是否存在强排序关系，不表示标签方向能从训练集稳定泛化到测试集。bag-of-words 的 AUC 方向几乎反转，但方向无关 AUC 仍为 0.933，说明原始数据标签与词汇分布之间存在强线性结构。这个结果足以说明原始 0.953 AUC 不能直接解释为抽象事实表征。

在词汇平衡数据上，surface baseline 被压到随机：

| Dataset | Baseline | Accuracy | AUC | Direction-agnostic AUC |
|---|---|---:|---:|---:|
| capital_balanced | numeric_surface | 0.500 | 0.500 | 0.500 |
| capital_balanced | bag-of-words | 0.500 | 0.500 | 0.500 |

此时 residual probe 仍有中等强度可分性：

| Layer | Accuracy | AUC |
|---:|---:|---:|
| 6 | 0.625 | 0.809 |
| 8 | 0.625 | 0.802 |
| 7 | 0.667 | 0.783 |

balanced probe seed sensitivity 中，layer 6 mean AUC 为 0.813，layer 8 mean AUC 为 0.804。也就是说，排除 unigram 边际频率后，中层 residual state 仍保留非平凡标签信号。但它仍可能来自 subject-object compatibility、实体熟悉度、句子概率或关系模板，而不是抽象 truth representation。

从定位角度看，本文完成了 layer-position level localization：标签信号在 balanced layer 6 的 prompt-final residual state 上可读，且该位置的单点干预足以移动后续 completion margin。当前仍未定位负责生成或传递该效应的具体 attention head、MLP 与下游 computation path。

## 5. Completion Margin：剩余信号是否对应模型补全偏好

为了连接 activation signal 与模型行为，本文用首都补全偏好作为行为侧指标：

```text
The capital of France is Paris
The capital of France is Berlin
```

定义两个 completion margin：

```text
completion_total_margin =
  log p(correct capital | prompt) - log p(false capital | prompt)

completion_avg_token_margin =
  mean_token_logp(correct capital | prompt) - mean_token_logp(false capital | prompt)
```

total logprob 更接近完整 completion 概率，但偏向短 completion；avg-token logprob 缓解长度差异，但改变了指标含义。held-out 结果如下：

| Analysis | Rows | Blocks | Accuracy | AUC | 95% Block Bootstrap CI |
|---|---:|---:|---:|---:|---:|
| completion_total | 48 | 12 | 0.750 | 0.861 | [0.753, 0.955] |
| completion_avg_token | 48 | 12 | 0.625 | 0.786 | [0.674, 0.891] |
| residual_probe | 48 | 12 | 0.625 | 0.809 | [0.708, 0.922] |

total logprob 看起来强于 residual probe，但 held-out 中有 24 行 correct/false capital token 数不同；归一化后，completion AUC 降到 0.786，低于 residual probe 的 0.809。三个指标的 block bootstrap CI 也高度重叠。因此，completion margin 支持“activation signal 与补全兼容度有关”，但不能说明 probe 主要由 completion preference 解释，也不能说明 completion baseline 显著强于 residual probe。

## 6. 是否跨领域稳定

原始多领域实验未提供统一 truth direction 证据。probe sweep 中，不同 domain 的最佳层和 AUC 差异很大；direction cosine 的平均跨域相似度为 0.077，整体接近 0。domain transfer 只有地理相关任务之间存在局部共享，例如 continent -> capital direction-agnostic AUC 为 0.766，landmark_country -> capital 为 0.735。

这应解释为 exploratory evidence：当前跨领域实验没有提供支持统一 truth direction 的证据，但也不能写成“已经证明跨领域方向不稳定”。不同 domain 的样本量、负样本构造、prompt 形式都不完全一致，仍可能存在模板和实体类型 confound。

## 7. 沿该方向干预是否产生下游效应

这里的核心问题不是证明模型在自然前向计算中“使用”这个方向，而是检查外部沿该方向干预时，是否会对下游 completion score 或 choice-level 指标产生可测影响。早期 true/false verbalizer 因 GPT-2-small 固定输出偏置而失效；对应的 oracle steering 能移动内部 probe score，但 logit-sign accuracy 仍为 0.5。truth verification patching 显示后层 residual state 对当前 logit 差异有局部影响，但 clean-corrupt denominator 很小，shuffled control 也有非零 shift。因此，这些结果只作为早期诊断，解释为什么主线转向原生 completion margin。

### 7.1 Completion-margin steering

主线干预使用 balanced layer 6 verification-associated direction 和 correct-vs-wrong capital completion margin。该方向先在 statement true/false prompt 上训练，再迁移到 bare completion prompt，因此这是 cross-format transfer。

当前主 split 的 prompt-final-only 结果如下：

| Direction | Alpha | Mean Delta Avg-Token Margin | 95% Block Bootstrap CI | Pairwise Accuracy |
|---|---:|---:|---:|---:|
| learned_probe | -4 | -0.130 | [-0.195, -0.068] | 0.625 |
| learned_probe | 0 | 0.000 | [0.000, 0.000] | 0.625 |
| learned_probe | +4 | +0.135 | [0.075, 0.198] | 0.625 |
| random_direction | +4 | -0.031 | [-0.043, -0.017] | 0.625 |
| label_permutation | +4 | -0.024 | [-0.036, -0.015] | 0.625 |

paired block bootstrap 直接比较同一批 held-out block：

| Comparison | Estimate | 95% paired block CI |
|---|---:|---:|
| learned - random | +0.166 | [0.096, 0.239] |
| learned - label permutation | +0.159 | [0.090, 0.232] |

sampled null distribution 设置为 prompt-final-only、alpha=+4、held-out countries：

| Null type | Directions | Null mean | Null 95% interval | Learned effect | Empirical p |
|---|---:|---:|---:|---:|---:|
| random direction | 50 | +0.009 | [-0.065, 0.091] | +0.135 | 0.020 |
| label permutation | 20 | +0.010 | [-0.081, 0.088] | +0.135 | 0.048 |

这里的 empirical p 不能被理解成精确显著性估计。更准确地说，learned effect 超过当前全部 50 条随机方向和全部 20 条乱标签方向；受 sampled null 数量限制，经验 p 值的分辨率分别约为 1/51 和 1/21。

### 7.2 Repeated split steering

为了检查单 split 限制，本文进一步做 10 个 repeated group splits。每个 split 训练一个 balanced layer 6 direction，测试 prompt-final-only alpha=+4，并采样 10 条 random directions 与 5 条 label-permutation directions。

| Metric | Baseline | Steered | Change |
|---|---:|---:|---:|
| mean pairwise accuracy | 0.700 | 0.725 | +0.025 |
| wrong -> correct flips | - | 6 | +6 |
| correct -> wrong flips | - | 0 | 0 |
| mean avg-token margin shift | - | - | +0.116 |

| Margin statistic across splits | Result |
|---|---:|
| learned delta mean | +0.116 |
| learned delta std | 0.023 |
| learned delta min / max | +0.085 / +0.150 |
| mean learned-minus-random | +0.125 |
| mean learned-minus-permutation | +0.119 |
| learned delta > 0 | 10/10 splits |
| learned > all sampled random directions | 10/10 splits |
| learned > all sampled permutation directions | 10/10 splits |

这比单 split 的证据更强：learned direction 的 completion-margin effect 在不同 train/test country split 上重复出现。但这些 repeated splits 共享同一个数据池，训练和测试国家集合会部分重叠，因此 10/10 为正主要说明结果不是某个单一 seed 或划分的偶然现象，不能等同于 10 个相互独立实验构成的正式假设检验。choice-level 改善仍然很弱：平均 pairwise accuracy 只提升 +0.025，6 个 sign flips 虽然都从 wrong -> correct，但总量很小，且仍未检验自由生成。

### 7.3 Position、decomposition 与 rank

position decomposition 结果如下：

| Position Mode | Learned Alpha +4 Delta Margin | Learned - Random CI | Learned - Permutation CI | Pairwise Accuracy |
|---|---:|---:|---:|---:|
| all positions | +0.133 | [0.094, 0.239] | [0.092, 0.233] | 0.625 |
| prompt-final-only | +0.135 | [0.096, 0.239] | [0.090, 0.232] | 0.625 |
| completion-internal-only | -0.002 | [-0.003, 0.004] | [-0.004, 0.008] | 0.625 |

这说明当前 effect 主要来自 prompt-final residual position，不是 completion 内部多位置注入的累积影响。

decomposition 进一步说明它不是纯粹事实纠错。当前主 split、alpha=+4 时，correct completion avg-token logprob 上升 +0.280，false completion 也上升 +0.147，margin 增加 +0.133。换句话说：

```text
shared uplift        = (0.280 + 0.147) / 2 = +0.214
differential uplift  =  0.280 - 0.147       = +0.133
```

共享 uplift 比 correct-over-wrong differential 更大。该方向更像一个较大的 capital-completion promotion component，叠加一个较小的配对兼容度分量。baseline 本来偏向 correct 的国家平均 shift 为 +0.144，baseline 本来偏向 wrong 的国家为 +0.114；二者差值 +0.030 的 block bootstrap CI 为 [-0.077, 0.136]，跨过 0。当前主 split 的 sign flip count 为 0。

candidate-set rank 检查把候选从一个 selected wrong capital 扩展到 76 个首都候选。steering 后正确首都平均 rank 从 15.04 改善到 14.13，10/24 个 held-out countries 的 rank 提升，没有 rank 变差；但 top-1 accuracy 只从 0.083 到 0.125，top candidate 只改变 1 次。因此它支持“候选评分有弱改善”，但不支持稳定 choice-level 改善。

unembedding projection baseline 也不能给出简单解释。对 held-out learned alpha=+4，静态 `direction @ W_U` 投影的 predicted-vs-observed 相关性为 -0.188，corr squared 仅 0.035。简单 unembedding projection 不足以解释 observed shift；差异需要由 final layernorm、layer 6 后续计算及上下文依赖的局部映射中的一种或多种因素解释。

### 7.4 Ablation 与 sensitivity

ablation 表明 learned direction 与可分性相关但不充分。原始 capital 数据上，单方向 ablation 能把 fixed-direction score gap 从 0.573 降到接近 0，但重新训练 probe 后 AUC 仍为 0.945。iterative ablation 中 learned removal 从 0.953 降到 0.807，而 random/permutation controls 下降更弱。

在 balanced dataset 上，balanced ablation 结果更弱但更可信：

| Setting | Key result |
|---|---|
| balanced single-direction ablation | strength=1 后 fixed-direction gap 约为 0，retrained AUC 仍为 0.786 |
| balanced iterative ablation | learned_iterative 16 directions 后 AUC 为 0.726，random control 为 0.793 |

移除一个方向后，模型 activation 中仍有其他可训练 probe 利用的结构；因此 learned direction 不是唯一充分机制。

争议事实 sensitivity 删除 `Ghana/South Africa`、`Paraguay/Bolivia`、`Israel/Jordan` 三个 block 后，数据变为 140 行、35 个 block，并用同一随机种子重新做 group split，在剩余训练数据上重新训练 probe 和 steering direction；删除后 held-out 为 11 个 block / 44 行。该重新划分实验中，layer 6 residual probe held-out AUC 为 0.864，completion avg-token AUC 为 0.913；prompt-final learned steering delta 为 +0.120，random control 为 -0.038，label-permutation control 为 -0.005。因此它说明结论对删除争议事实具有定性稳健性，但不用于和主 split 数值直接比较。

## 8. 分层结论与边界

本文最重要的结论应按层次理解：

| Level | Current result | Evidence strength |
|---|---|---|
| Readout | balanced residual probe AUC 约 0.81 | 较强 |
| Score | prompt-final steering 稳定提高 completion margin | 中等，且跨 repeated splits |
| Choice | pairwise / candidate top-1 改善有限 | 很弱 |
| Free generation | 未系统验证 | 未知 |

这个分层很关键：readout 成立不等于 score-level 因果成立；score margin 增加也不等于 factual choice 提高。

本文可以稳妥地说：

1. 原始 capital truth-direction 结果受到强 lexical structure 放大；
2. 词汇平衡后，bag-of-words 失效，但 residual probe 仍有约 0.81 AUC 的中等强度可分性；
3. completion margin 显示该信号与模型补全兼容度有关，但结论对 total/avg-token 定义敏感；
4. 当前跨领域实验没有提供支持统一 truth direction 的证据；
5. 在当前主 split 上，completion-margin steering 能在 prompt-final residual position 弱移动 scoring margin，并超过 sampled null controls；
6. repeated split steering 显示这种弱 margin shift 能跨多个 group splits 重复；
7. 该干预只能小幅改善候选 rank 和平均 pairwise accuracy，尚不能稳定改变 held-out correct-vs-selected-wrong pairwise choice 或候选集合 top-1 choice；
8. ablation 表明该方向相关但不充分，尚不能定位完整事实机制。

因此，balanced probe 剩余信号尚不能被解释为纯粹 truth representation。更稳妥的说法是：它是与事实配对、补全兼容度、completion 指标选择和数据标签共同相关的 activation-level signal。它能弱影响模型评分偏好，但还不能被称为事实纠错机制。

本文没有证明 GPT-2-small 存在可部署的 truth direction、跨领域稳定的唯一 truth direction，或一个已经定位的 truth circuit。更有价值的后续工作不是继续堆 probe，而是做更接近 Bao et al. 的 QA / logical transformation 复现，以及 final layernorm、后续层路径、head/MLP 级别的因果分解。
