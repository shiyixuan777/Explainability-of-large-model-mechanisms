# Presentation Outline

建议做 8-10 页 slides。每页只放一个核心信息，重点展示图和结论，不要把报告全文搬进去。

## Slide 1: 项目题目

题目：基于 Truth Direction 的 GPT-2-small 事实判断机制定位与推理时干预研究

要点：
- 课程方向：大模型机制可解释性
- 模型：GPT-2-small
- 框架：TransformerLens
- 现象：true/false factual statement representation

## Slide 2: 研究问题

要点：
- GPT-2-small 哪些层包含 true/false 信息？
- 这些信息是否跨事实领域稳定？
- 找到的 direction 是否能被 steering 或 ablation 控制？
- Activation patching 是否能给出更强的因果证据？

## Slide 3: 数据集与实验流程

放图/表：
- 数据集领域分布表
- 528 条英文事实判断样本，true/false 各 264 条

要点：
- 使用 group split，避免同一个 `pair_id` 同时进入训练和测试。
- prompt 形式包括 `statement_is`、`answer`、`question`。

## Slide 4: Locate 结果：分领域 Probe Sweep

推荐图：
- `figures/probe_sweep_summary.png`

要点：
- 混合领域信号较弱。
- capital 领域最稳定，`answer` prompt 第 8 层 AUC=0.953。
- truth direction 更像领域相关结构，而不是所有事实共享的单一方向。

## Slide 5: Focused Capital Probe 与 PCA

推荐图：
- `figures/probe_capital_answer.png`
- `figures/pca_capital_layer8.png`

要点：
- 第 8 层 AUC 最高，第 10 层 accuracy 最高。
- PCA 二维图没有完全分离 true/false，说明 supervised high-dimensional direction 比无监督主成分更关键。

## Slide 6: 错误样本分析

推荐表：
- `figures/error_analysis_capital_layer8_errors.csv` 中挑 4-6 个例子

要点：
- test split 46 条，38 正确，8 错误。
- AUC 高但固定 0.5 阈值仍有偏置。
- 这说明 probe 证明的是“可线性读取”，不是“完美事实判断器”。

## Slide 7: Activation Patching 因果定位

推荐图：
- `figures/activation_patching_capital_recall.png`

要点：
- 最后层 residual stream patching 可以恢复目标首都 logit。
- `attn_out` 对恢复有强贡献，但单独 patch 后平均 logit diff 未翻正。
- `mlp_out` 单独贡献较弱。

## Slide 8: Steering 与 Ablation

推荐图：
- `figures/steering_capital_probe_layer8_probe_accuracy.png`
- `figures/ablation_capital_probe_layer8_score_gap.png`

要点：
- Probe-direction steering 能控制内部 score，但没有提升输出层 true/false 判断。
- Ablation 能移除单一 probe direction 上的 gap。
- 重新训练 probe 仍保持高 AUC，说明信息是冗余/分布式的。

## Slide 9: 对论文复现的对应

要点：
- 复现 truth/false 线性结构思想：linear probe + truth direction。
- 复现可视化思想：PCA。
- 复现因果干预思想：activation patching、steering、ablation。
- 拓展点：分领域 sweep 和负结果分析。

## Slide 10: 结论与局限

结论：
- GPT-2-small 在 capital fact verification 中存在强线性 truth/false 表征。
- 这种结构不稳定地跨所有事实领域泛化。
- 可读性不等于直接可控性。

局限：
- 当前模型较小，未扩展到 Qwen2.5。
- 当前 patching 是 layer/module 级，尚未做 head-level patching。
- Steering 是 global direction，未来可以做输入条件化 steering 或多方向子空间干预。

