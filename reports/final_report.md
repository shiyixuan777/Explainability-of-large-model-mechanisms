# GPT-2-small 事实配对标签信号的线性可分性、补全兼容度与干预边界

## 摘要

本文研究 GPT-2-small 在人工事实验证任务中，最后一个词元位置的残差流（residual stream）是否包含可线性读出的事实配对标签信号，以及该信号是否能通过推理时激活干预影响模型的后续补全评分。实验首先发现，原始首都事实数据上的残差流探针很强，第 8 层 AUC 为 0.953；但词袋表面特征基线的方向无关 AUC 也达到 0.933，说明原始高分受到明显词汇伪线索放大。为此，本文构造词汇平衡首都数据集，使每个国家名和首都名在 true/false 标签中的边际频率完全平衡。在该设置下，词袋基线和数值表面特征基线均降到 0.500，而第 6 层残差流探针仍保留 0.809 AUC，多随机种子的平均 AUC 为 0.813。

进一步实验显示，这一信号与模型补全兼容度相关，但不是直接的事实选择机制。补全得分差中，总 logprob AUC 为 0.861，但按词元数归一化后，平均词元 logprob AUC 降至 0.786，与残差流探针的 0.809 接近，且二者自助法区间高度重叠。沿词汇平衡数据第 6 层的标签相关方向，在提示词末位置的残差状态上进行干预，可将留出测试集的平均词元补全得分差推动约 +0.135，并超过当前采样的 50 条随机方向和 20 条标签置乱方向。10 个重复分组划分中，学习方向带来的变化均为正，均值为 +0.116。

这种效果主要停留在评分层面。当前主划分的配对偏好准确率没有改变；重复划分中，配对偏好准确率只从 0.700 提升到 0.725；候选集排名检查中，正确首都平均排名从 15.04 改善到 14.13，top-1 准确率仅从 0.083 到 0.125。因此，本文的结论是：GPT-2-small 在词汇平衡首都事实中存在一个可读且可弱干预的事实配对标签信号；它能影响“正确首都 vs 选定错误首都”的补全得分差，但尚不能解释为跨领域稳定、可直接控制输出的全局真值方向（truth direction），也尚未定位完整事实机制路径。

## 1. 引言

机制可解释性中，线性探针（linear probe）常被用来判断模型隐状态是否包含某类信息。但探针 AUC 高并不自动说明模型自然计算中使用了该方向，也不说明该方向能稳定控制模型行为。围绕真值方向的工作尤其容易被误读：一个能够区分 true/false 标签的方向，可能来自抽象事实表征，也可能来自数据构造、实体频率、模板形式、补全概率或标签泄漏。

本文选择一个小而可控的现象：GPT-2-small 对人工事实验证句子的 true/false 标签是否在残差流中形成可读信号。研究问题分为三层：

1. **读出层面**：残差激活中是否存在可线性读出的事实配对标签信号。
2. **定位层面**：该信号出现在哪些层和词元位置，并是否受词汇伪线索影响。
3. **干预层面**：沿该方向进行推理时干预，是否能改变模型后续补全评分或选择行为。

本文的贡献有三点。第一，展示原始首都事实探针的高 AUC 很大程度上被词汇结构放大，并用词汇平衡数据集拆除这一伪线索。第二，在平衡数据上仍观察到中等强度残差信号：事实验证提示词的第 6 层最后词元激活可读，且迁移到直接补全提示词的第 6 层提示词末位置后，可以移动补全得分差。第三，通过随机方向、标签置乱方向、位置分解、重复划分和候选集排名检查，说明该方向主要产生评分层面影响，选择层面改善较弱。

## 2. 相关工作与复现范围

本文参考 Bao et al. [1] 的 *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*。该工作关注真实性探针是否能跨任务、逻辑变换、问答形式和知识源泛化，并提醒真值方向的解释需要经过迁移与行为检验。本文不是完整复现其全部实验，而是在 GPT-2-small [2] 上做受控小模型复现与扩展。

方法上，本文使用 TransformerLens [3] 的 hook 机制提取残差流激活，并借鉴 logit lens [4]、激活修补/因果追踪（activation patching / causal tracing）[5,6]、真值几何与线性探针 [7,8] 以及激活工程（activation engineering）[9] 的基本流程。与完整回路（circuit）级机制分析相比 [10,11]，本文只做到层级读出、位置特定干预和方向级干预，还没有展开注意力头、MLP 或路径级机制分解。

| 类型 | 研究问题 | 本文对应实验 | 结论关系 |
|---|---|---|---|
| Bao et al. 对照 | 隐状态中能否训练真实性探针 | GPT-2-small 最后词元 `resid_post` 探针 | 复现线性可读现象，但仅限人工事实验证数据 |
| Bao et al. 对照 | 真值方向是否跨任务泛化 | 跨领域迁移与方向余弦相似度 | 未提供统一方向证据；地理相关任务有局部共享结构 |
| Bao et al. 对照 | 探针能否用于下游行为 | 补全得分差干预、输出读出基线 | 能弱移动补全评分，不能稳定改善配对选择 |
| 本文扩展 | 高探针 AUC 是否受数据伪线索影响 | 表面特征基线、词汇平衡数据、补全得分差 | 强化混杂诊断，说明高 AUC 容易来自数据结构 |

未复现的部分包括 Bao et al. 的完整逻辑变换、QA、ICL、外部知识数据、多模型比较和选择性问答应用。因此本文更接近一个面向课程项目的小模型机制分析：复现真值方向泛化的核心问题意识，并通过更强的表面特征、词汇平衡和激活干预控制实验检查解释边界。

## 3. 实验方法

### 3.1 数据与模型

原始数据集 `data/facts.csv` 包含 528 条英文事实验证样本，覆盖 capital、continent、element_symbol、book_author、landmark_country、science、math 七类。主线使用首都事实，因为它可以自然形成“正确首都 vs 错误首都”的补全评价，也便于构造词汇平衡对照。

词汇平衡数据集 `data/capital_balanced.csv` 使用二国二首都闭合 block。例如：

```text
France  - Paris   true
France  - Berlin  false
Germany - Berlin  true
Germany - Paris   false
```

这样国家名和首都名在 true/false 标签中的边际频率完全平衡。该数据集共有 152 行、38 个数据块。当前主划分的留出测试部分为 12 个数据块 / 48 行。数据中存在少量有定义或政治语境差异的事实，例如 `Israel -> Jerusalem`、`South Africa -> Pretoria`、`Bolivia -> Sucre`；continent 任务也使用了简化标签，例如把 Australia/Oceania 相关国家统一标为 `Australia`。本文用争议事实敏感性分析删除首都任务中的争议数据块后重新划分，检查主线结果是否定性保留。

### 3.2 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | GPT-2-small |
| 权重来源 | TransformerLens/Hugging Face 的 `gpt2-small` 预训练权重 |
| 框架 | TransformerLens；`HookedTransformer.from_pretrained(model_name, device="cpu", dtype=torch.float32)` |
| 主要激活 | 最后词元位置的 `resid_post` |
| 数据划分 | `GroupShuffleSplit(test_size=0.3, random_state=42)`；按 `pair_id` 分组 |
| 词汇平衡分组 | `capital_balanced` 中一个 `pair_id` 对应完整四行二国二首都数据块 |
| 线性探针 | `StandardScaler()` + `LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", fit_intercept=True, max_iter=2000, class_weight="balanced", random_state=42)` |
| 标准化 | scaler 只在训练集拟合，再应用到留出测试集 |
| 探针分数 | AUC 使用 `predict_proba(... )[:, 1]`；准确率使用 0.5 概率阈值 |
| 表面特征基线 | 数值基线使用字符数、词数、逗号/数字/大写字母/句点计数；BOW 基线使用 lowercase unigram+bigram `CountVectorizer` + logistic regression，并采用同样的分组划分 |
| 主要层 | 原始 capital 使用第 8 层；词汇平衡主线使用第 6 层 |
| 干预位置 | 提示词末位置的残差状态；另比较全位置和仅补全文本内部位置 |
| 干预方向 | 由训练集探针权重转换回原激活基底后做 L2 归一化 |
| alpha | 主报告关注 `alpha = ±4`，并展示 alpha sweep |
| 补全指标 | 正确首都与选定错误首都的总 logprob 得分差和平均词元得分差 |
| 自助法 | 以 `pair_id` 数据块为采样单位，主实验 2000 次；随机种子为 42 |
| 对照方向 | L2 归一化高斯随机方向、标签置乱探针方向 |
| 测试环境 | Windows 11、Python 3.13.7、PyTorch 2.12.1 CPU、TransformerLens 3.5.1；关键依赖见 `requirements.txt` |

第 6 层是在主划分的词汇平衡探针扫描中选定的，随后固定用于补全得分差干预、采样零分布对照、位置分解和重复划分实验。因此，主划分上的层特定结果带有探索性选择成分；重复划分实验用于检查固定该层后，效果是否仍能跨多个分组划分重复出现。alpha 扫描预先使用对称集合 `{-4, -2, -1, 0, 1, 2, 4}`，`alpha=4` 作为正向端点用于零分布对照比较。当前自助法/零分布置信区间没有把层选择的不确定性纳入区间。

### 3.3 指标

探针主要报告准确率、AUC，以及方向无关 AUC（direction-agnostic AUC）。方向无关 AUC 定义为 `max(AUC, 1-AUC)`，只用于诊断预测分数与标签之间是否存在强排序关系，不表示标签方向能从训练集稳定泛化到测试集。

补全得分差定义为：

```text
completion_total_margin =
  log p(correct capital | prompt) - log p(false capital | prompt)

completion_avg_token_margin =
  mean_token_logp(correct capital | prompt) - mean_token_logp(false capital | prompt)
```

总 logprob 更接近完整补全概率，但偏向短补全；平均词元 logprob 缓解长度差异，但改变了指标含义。因此本文同时报告两者，并将平均词元得分差作为激活干预主指标。

选择层面指标的统计单位如下。配对偏好准确率（pairwise preference accuracy）以留出测试国家为单位：若该国家正确首都的平均词元得分高于选定错误首都，则记为正确。数据块完全正确率（block exact accuracy）以四行二国二首都数据块为单位：只有数据块内两个国家级判断全部正确，才记为正确。符号翻转（sign flip）指激活干预前后的配对平均词元得分差符号发生改变；wrong -> correct 表示得分差从非正变为正，correct -> wrong 表示得分差从正变为非正。候选集排名（candidate-set rank）在 76 个预设首都候选中按平均词元得分排序，rank 1 表示模型给该候选最高分；top-1 准确率表示正确首都是否排在第一位。重复划分干预共包含 10 个划分，每个划分有 24 次留出国家级评价，因此汇总翻转统计的分母是 240 次评价事件；其中观察到 6 次 wrong -> correct 翻转和 0 次 correct -> wrong 翻转，这些事件不一定对应 6 个互不重复的国家。

标签置乱对照的生成方式也保持固定：只在训练划分内逐行随机置换标签，留出测试集、输入文本和分组不变；由于是置换，训练集正负标签数量保持不变。每条标签置乱方向都重新拟合 scaler 与 logistic 探针，再把探针权重转换回原激活基底并做 L2 归一化。固定 alpha 扫描中的标签置乱对照和采样零分布中的 20 条标签置乱方向使用同一流程。

## 4. 实验结果

### 4.1 词汇伪线索放大了原始探针

原始 capital 探针很强：第 8 层 AUC 为 0.953，第 10 层 AUC 为 0.947，第 5 层 AUC 为 0.941。随机种子敏感性检查显示，第 8 层在六个分组划分上的平均 AUC 为 0.899，范围为 0.832-0.953，说明结果不是单一随机种子的偶然现象。

但表面特征基线暴露了核心混杂因素。原始 capital 中，数值表面特征基线较弱，方向无关 AUC 为 0.549；词袋基线的普通 AUC 几乎反向，为 0.067，但方向无关 AUC 达到 0.933。这说明原始标签与词汇分布之间存在强线性结构，不能把原始残差流探针的 0.953 AUC 直接解释为抽象事实表征。

如图 1 所示，词袋基线的方向无关 AUC 接近原始残差流探针，说明原始数据的高探针 AUC 不能单独作为机制证据。

![表面特征基线](../figures/surface_baselines.png)

**图 1　原始数据上的表面特征基线。** BOW 的普通 AUC 发生方向反转，但方向无关 AUC 达到 0.933，暴露出明显词汇结构。

词汇平衡数据集拆除了最直接的 unigram 边际频率线索。在 `capital_balanced` 上，数值表面特征基线和词袋基线的准确率、AUC、方向无关 AUC 均为 0.500。

### 4.2 平衡数据上仍存在残差流可分性

去掉词汇边际频率线索后，残差流探针仍有中等强度可分性。`capital_balanced` 上第 6 层 AUC 为 0.809，第 8 层 AUC 为 0.802，第 7 层 AUC 为 0.783。多随机种子检查中，第 6 层平均 AUC 为 0.813，第 8 层平均 AUC 为 0.804。

如图 2 所示，词汇平衡后表面特征基线降到随机水平，但残差流探针在中层仍保留可分性。

![词汇平衡首都探针层扫描](../figures/probe_capital_balanced.png)

**图 2　词汇平衡首都数据上的层级探针。** 第 6、7、8 层的留出测试 AUC 仍高于随机，其中第 6 层是后续词汇平衡激活干预主线使用的层。

从定位角度看，本文完成的是两步证据，而不是完整统一路径定位：在事实验证提示词中，第 6 层最后词元残差激活的标签可读性较强；将该方向迁移到直接补全提示词的第 6 层提示词末位置进行单点干预，能够移动后续补全评分。该结果构成层级读出定位和跨格式位置特定干预证据，但尚不是统一计算路径的完整层-位置定位。当前仍未定位负责生成或传递该效应的具体注意力头、MLP 与下游计算路径。

跨领域方向一致性较弱。在原始多领域数据上，不同领域探针方向的平均跨领域余弦相似度仅为 0.077，多数跨领域迁移 AUC 接近随机水平。局部例外主要出现在语义相近的地理领域，例如 `continent -> capital` 的 AUC 为 0.766。该结果不支持统一的跨领域真值方向，但提示相关地理任务之间可能共享部分表征结构。由于这些实验使用原始而非词汇平衡多领域数据，本文只将其作为探索性边界证据。

### 4.3 剩余信号与补全兼容度相关但不等价

为了连接激活信号与模型行为，本文使用首都补全偏好作为行为侧指标。例如：

```text
The capital of France is Paris
The capital of France is Berlin
```

在同一留出测试划分上，补全总 logprob AUC 为 0.861，看起来高于残差流探针的 0.809；但留出测试集中有 24 行正确/错误首都的词元数不同。按词元数归一化后，补全平均词元 AUC 降到 0.786，低于残差流探针。三者的数据块自助法置信区间高度重叠。

| 分析方法 | 样本数 | 数据块数 | 准确率 | AUC | 95% 数据块自助法 CI |
|---|---:|---:|---:|---:|---:|
| completion_total | 48 | 12 | 0.750 | 0.861 | [0.753, 0.955] |
| completion_avg_token | 48 | 12 | 0.625 | 0.786 | [0.674, 0.891] |
| residual_probe | 48 | 12 | 0.625 | 0.809 | [0.708, 0.922] |

如图 3 所示，补全总 logprob 与平均词元 logprob 对结论强度的影响不同，因此本文后续以平均词元得分差作为更保守的干预指标。

![补全得分差基线](../figures/capital_knowledge_margin_summary.png)

**图 3　补全兼容度基线。** 总 logprob AUC 高于残差流探针，但长度归一化后的平均词元 AUC 下降，三者数据块自助法区间高度重叠。

这说明剩余激活信号与补全兼容度有关，但总 logprob / 平均词元 logprob 的定义会影响结论。本文因此把补全得分差视为行为侧线索，而不是事实知识的直接度量。

### 4.4 标签相关方向能弱移动补全评分

主线干预使用词汇平衡数据第 6 层的标签相关方向，并在直接补全提示词的提示词末残差状态上加上 `alpha * direction`。该方向先在 statement true/false 事实验证提示词上训练，再迁移到首都补全提示词，因此这是一个跨格式迁移设置。

当前主划分中，仅提示词末位置干预（代码参数 `prompt-final-only`）的结果如下。`alpha=+4` 时，学习方向将留出测试集的平均词元得分差推动 +0.135；`alpha=-4` 时，得分差变化为 -0.130。随机方向和标签置乱方向在 `alpha=+4` 时分别为 -0.030 和 -0.022。

如图 4 所示，学习方向的 alpha 曲线方向稳定：正向 alpha 提高平均词元得分差，负向 alpha 降低该得分差。

![补全得分差干预 alpha 曲线](../figures/completion_margin_steering_position_prompt_final_summary.png)

**图 4　补全得分差干预的 alpha 曲线。** 主线方向为词汇平衡数据第 6 层探针方向，干预位置为直接补全提示词的提示词末残差状态。

配对数据块自助法直接比较同一批留出测试数据块：学习方向 - 随机方向的估计值为 +0.165，95% 配对数据块 CI 为 [0.096, 0.239]；学习方向 - 标签置乱方向的估计值为 +0.158，CI 为 [0.090, 0.232]。

采样零分布设置为仅提示词末位置干预、`alpha=+4`、留出测试国家。学习方向效果超过全部 50 条随机方向和全部 20 条标签置乱方向；经验 p 值的分辨率分别约为 1/51 和 1/21，因此只能解释为采样对照下的强对照结果，而不是精确显著性估计。

如图 5 所示，学习方向的效果位于当前采样零分布对照之外。

![采样零分布](../figures/completion_margin_steering_null_distribution.png)

**图 5　采样零分布。** 零分布对照包含 50 条 L2 归一化高斯随机方向和 20 条标签置乱探针方向；经验 p 值受采样数量限制。

### 4.5 效果跨划分重复，但选择层面改善有限

为了检查单一划分限制，本文进一步做 10 个重复分组划分。每个划分训练一个词汇平衡数据第 6 层方向，测试仅提示词末位置干预（`prompt-final-only`）下的 `alpha=+4`，并采样 10 条随机方向与 5 条标签置乱方向。

| 指标 | 结果 |
|---|---:|
| 学习方向效果均值 | +0.116 |
| 学习方向效果标准差 | 0.023 |
| 学习方向效果最小值 / 最大值 | +0.085 / +0.150 |
| 学习方向 - 随机方向均值 | +0.125 |
| 学习方向 - 标签置乱方向均值 | +0.119 |
| 学习方向效果 > 0 | 10/10 个划分 |
| 学习方向 > 所有采样随机方向 | 10/10 个划分 |
| 学习方向 > 所有采样标签置乱方向 | 10/10 个划分 |

如图 6 所示，学习方向带来的得分差变化在 10 个重复分组划分中均为正，但这些划分共享同一数据池。

![重复划分干预](../figures/repeated_split_completion_steering_summary.png)

**图 6　重复分组划分下的激活干预。** 该图用于检查结果是否依赖单一划分，不应被解释为 10 个相互独立实验的正式假设检验。

这些重复划分共享同一个数据池，训练和测试国家集合会部分重叠，因此 10/10 为正主要说明结果不是某个单一随机种子或划分的偶然现象，不能等同于 10 个相互独立实验构成的正式假设检验。

选择层面改善明显更弱。当前主划分的配对偏好准确率没有改变，符号翻转为 0。重复划分中，基线配对偏好准确率为 0.700，干预后为 0.725，平均只提升 +0.025；在全部重复划分测试出现次数中，共观察到 6 次 wrong -> correct 翻转和 0 次 correct -> wrong 翻转。由于不同划分的测试集合可能重叠，这些是 6 次翻转评价事件，不一定对应 6 个不同国家。候选集排名检查把候选从一个选定错误首都扩展到 76 个首都候选：正确首都平均排名从 15.04 改善到 14.13，top-1 准确率只从 0.083 到 0.125。

### 4.6 效果主要来自提示词末位置

位置分解比较全位置干预、仅提示词末位置干预（`prompt-final-only`）和仅补全文本内部位置干预（`completion-internal-only`）。结果显示，仅提示词末位置干预几乎复现全位置干预效果，而仅补全文本内部位置干预近似为 0。

| 干预位置模式 | 学习方向在 `alpha=+4` 时的得分差变化 | 学习方向 - 随机方向 CI | 学习方向 - 标签置乱方向 CI | 配对偏好准确率 |
|---|---:|---:|---:|---:|
| 全位置干预（`all`） | +0.133 | [0.094, 0.239] | [0.092, 0.233] | 0.625 |
| 仅提示词末位置（`prompt-final-only`） | +0.135 | [0.096, 0.239] | [0.090, 0.232] | 0.625 |
| 仅补全文本内部位置（`completion-internal-only`） | -0.002 | [-0.003, 0.004] | [-0.004, 0.008] | 0.625 |

如图 7 所示，仅提示词末位置干预基本复现全位置干预效果，而仅补全文本内部位置干预接近 0。

![位置分解](../figures/completion_margin_steering_position_comparison.png)

**图 7　位置分解实验。** 当前干预效果主要来自直接补全提示词的第 6 层提示词末位置，而不是补全文本内部多位置注入。

提示词末位置分解进一步说明该方向不是纯粹事实纠错。当前主划分、`alpha=+4` 时，正确补全的平均词元 logprob 上升 +0.281，错误补全也上升 +0.146，得分差增加 +0.135。共同提升分量约为 +0.214，大于正确—错误差异分量 +0.135。该方向更像一个较大的“首都补全促进”分量，叠加一个较小的配对兼容度分量。

### 4.7 消融与敏感性分析

消融表明学习方向与可分性相关但不充分。原始 capital 数据上，单方向消融能把固定方向分数差从 0.573 降到接近 0，但重新训练探针后 AUC 仍为 0.945。迭代消融中，学习方向移除后 AUC 从 0.953 降到 0.807，而随机方向/标签置乱方向对照下降更弱。

在词汇平衡数据上，单方向消融后固定方向分数差接近 0，但重新训练探针后的 AUC 仍为 0.786；迭代移除 16 个学习方向后 AUC 为 0.726，随机方向对照为 0.793。这说明学习方向相关但不是唯一充分机制。需要强调的是，词汇平衡迭代消融目前只在主划分上运行，未进行重复划分或配对自助法稳健性检验，因此这里只将其作为方向冗余性的定性诊断，而不把学习方向和随机方向之间的差距解释为稳定效应。

争议事实敏感性分析删除 `Ghana/South Africa`、`Paraguay/Bolivia`、`Israel/Jordan` 三个数据块后，数据变为 140 行、35 个数据块，并用同一随机种子重新做分组划分，在剩余训练数据上重新训练探针和干预方向。删除后留出测试集为 11 个数据块 / 44 行；第 6 层残差流探针留出测试 AUC 为 0.864，补全平均词元 AUC 为 0.913；提示词末位置学习方向干预变化为 +0.120，随机方向对照为 -0.038，标签置乱方向对照为 -0.005。由于该实验重新划分了数据，它只说明主线结论对删除争议事实具有定性稳健性，不用于和主划分数值直接比较。

## 5. 讨论

本文结果可以按三层理解。第一，读出层面成立：词汇平衡后，词袋基线失效，但残差流探针仍有约 0.81 AUC。第二，评分层面有因果效应：沿第 6 层提示词末位置的标签相关方向进行干预，能稳定移动“正确首都 vs 选定错误首都”的补全得分差，并超过采样零分布对照。第三，选择层面证据较弱：配对偏好准确率和受限候选集 top-1 只显示很弱的选择层面改善；开放式自由生成尚未系统评估。

这一分层解释了为什么本文没有把该方向称为全局真值方向。探针读出不等于模型自然使用该方向；补全得分提升不等于选择行为提升；单方向消融后仍可重新训练出探针，也说明信号不是单一充分机制。更稳妥的表述是：GPT-2-small 在词汇平衡首都事实中存在一个与事实配对和补全兼容度相关的标签信号，该信号可以弱影响补全评分，但尚未构成完整事实判断机制。

当前工作的主要限制包括：模型只使用 GPT-2-small；主线数据是人工构造首都事实；负样本来自选定错误首都，而不是开放生成；注意力头、MLP 或路径级回路尚未定位；重复划分不是相互独立实验；采样零分布的经验 p 值受方向数量限制。干预强度 alpha 以 L2 归一化方向的欧氏尺度定义，尚未根据残差流 RMS、激活范数或训练分布中的投影标准差校准，因此当前结果能说明该方向具有干预效力，但不能充分判断 `alpha=4` 是否处在自然激活分布内；高斯随机方向对照做了 L2 范数匹配，但没有匹配残差激活的协方差结构。

此外，探针是在残差维度高于训练样本数的高维小样本条件下拟合的；虽然使用了 L2 正则化、分组划分和多随机种子检查，但本文未系统评估不同划分所得方向本身的几何稳定性。词汇平衡与线性 unigram/bigram BOW 基线主要排除了实体边际频率和线性 n-gram 线索，不能排除非线性的国家-首都交互、实体熟悉度或其他语义兼容度因素。后续工作应优先做更接近 Bao et al. 的 QA / 逻辑变换复现，以及 final layernorm、后续层、注意力头和 MLP 的路径级因果分解。

## 6. 结论

本文从一个高探针 AUC 的事实验证现象出发，展示了机制可解释性中一个常见风险：线性可读信号很容易被误读为抽象真值表征。通过词汇平衡、补全得分差、提示词末位置激活干预、随机/标签置乱对照、重复划分和消融，本文将结论收束为一个较窄但更可靠的表述：GPT-2-small 在词汇平衡首都事实中存在一个可读、可弱干预的事实配对标签信号，它主要影响补全评分，而不是稳定的事实选择行为。

对课程项目要求而言，本文按照定位（Locate）、干预（Steer）与改进（Improve）的行动式机制可解释性框架 [12] 完成了 Locate 与 Steer，并对 Improve 进行了系统检验：干预能够稳定移动补全评分，但只产生很弱的选择层面改善。论文复现/扩展部分围绕 Bao et al. [1] 的真值方向泛化问题意识，在小模型上复现线性可读现象并补充了混杂因素诊断与控制实验。

## 7. 参考文献

1. Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Zhengwen Feng, Hao Peng, and Jianwei Yin. 2025. *Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks*. arXiv:2506.00823.
2. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. *Language Models are Unsupervised Multitask Learners*. OpenAI technical report.
3. Neel Nanda and Joseph Bloom. 2022. *TransformerLens: A Library for Mechanistic Interpretability of Generative Language Models*. Software library.
4. nostalgebraist. 2020. *Interpreting GPT: The Logit Lens*.
5. Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. 2020. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*. NeurIPS.
6. Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. *Locating and Editing Factual Associations in GPT*. NeurIPS.
7. Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. 2022. *Discovering Latent Knowledge in Language Models Without Supervision*. arXiv:2212.03827.
8. Samuel Marks and Max Tegmark. 2024. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*. Conference on Language Modeling. arXiv:2310.06824.
9. Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2024. *Steering Language Models With Activation Engineering*. arXiv:2308.10248.
10. Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2021. *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread.
11. Daking Rai, Yilun Zhou, Shi Feng, Abulhair Saparov, and Ziyu Yao. 2024. *A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models*. arXiv:2407.02646.
12. Hengyuan Zhang, Zhihao Zhang, Mingyang Wang, Zunhai Su, Yiwei Wang, Qianli Wang, Shuzhou Yuan, Ercong Nie, Xufeng Duan, Feijiang Han, Qibo Xue, Zeping Yu, Chenming Shang, Xiao Liang, Jing Xiong, Hui Shen, Chaofan Tao, Zhengwu Liu, Senjie Jin, Zhiheng Xi, Dongdong Zhang, Sophia Ananiadou, Tao Gui, Ruobing Xie, Hayden Kwok-Hay So, Hinrich Schütze, Xuanjing Huang, Qi Zhang, and Ngai Wong. 2026. *Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models*. arXiv:2601.14004.
