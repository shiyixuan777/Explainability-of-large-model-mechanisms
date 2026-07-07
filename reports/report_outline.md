# Report Outline

## 1. Introduction

- 研究目标：解释模型如何表示事实判断中的 truth/false 信息
- 为什么选择事实判断：现象明确、数据易构造、可做因果干预
- 与机制可解释性三步法的关系：locate, steer, improve

## 2. Background

- Transformer residual stream
- Hook 机制
- Logit Lens
- Activation Patching
- Steering Vector / Vector Arithmetic

## 3. Reproduction Target

- 复现 truth/false 激活具有线性结构的核心结论
- 检验该方向是否能通过推理时干预改变输出

## 4. Experimental Setup

- 模型：GPT-2-small；扩展可用 Qwen2.5-0.5B
- 数据集：英文和中文事实判断陈述
- Hook 点：`blocks.{layer}.hook_resid_post`
- 指标：accuracy, AUC, logit difference

## 5. Locate Results

- 每层 probe 结果
- 哪些层开始形成 truth/false 可分结构
- 失败案例分析

## 6. Steering Results

- truth direction 构造方式
- alpha 扫描结果
- 过强干预带来的副作用

## 7. Extension

- 中英文迁移实验
- 或不同模型对比

## 8. Discussion

- truth direction 是否稳定？
- 结果是否具有因果解释力？
- 与原论文结果的一致和不同之处
- 局限和未来改进

## 9. Conclusion

- 总结定位结果
- 总结干预效果
- 总结自己的分析和想法
