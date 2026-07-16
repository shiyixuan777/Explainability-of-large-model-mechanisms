# 结果汇总

本文件由 `python -m scripts.summarize_results` 根据 CSV 结果自动生成。
它用于作为报告结果的紧凑表格索引。

生成时间：2026-07-16T10:22:17
生成时 Git commit：3db24d6
生成时工作树是否有未提交修改：no
源目录：项目根目录
生成脚本：`scripts/summarize_results.py`

`direction_agnostic_auc = max(AUC, 1 - AUC)`。它诊断预测分数与标签之间是否存在强排序关系，而不关心方向符号；它不表示训练时学到的标签方向一定能作为分类器稳定泛化。

`learned_percentile = 1.0` 表示在当前采样集合中没有零分布方向超过学习方向效果；它不是总体分位数估计。`mean_rank_delta > 0` 表示正确候选向 rank 1 移动。重复划分中的翻转计数是跨重叠划分的评价事件，不一定对应互不重复的国家。

## 核心结果索引

| 结论 | 关键结果 |
| --- | --- |
| 原始数据存在词汇混杂 | 第 8 层残差流 AUC 0.953；BOW 方向无关 AUC 0.933 |
| 词汇平衡后仍可读出 | 第 6 层 AUC 0.809 |
| 评分层面干预 | 提示词末位置变化 0.135 |
| 重复划分稳定性 | 10/10 个划分为正；均值 0.116 |
| 选择层面效果 | 配对准确率变化 0.025；错误转正确事件 6 |
| 候选集 top-1 | 0.083 -> 0.125 |
| 机制边界 | 单方向消融后重训 AUC 0.786 |

## 原始数据表面特征基线

| 领域 | 基线 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- | --- |
| all | numeric_surface | 0.481 | 0.453 | 0.547 |
| all | bag_of_words | 0.281 | 0.192 | 0.808 |
| capital | numeric_surface | 0.478 | 0.451 | 0.549 |
| capital | bag_of_words | 0.174 | 0.067 | 0.933 |

## 探针扫描：按方向无关 AUC 排序的最高设置

| 领域 | 提示词 | 层 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- | --- | --- |
| capital | answer | 8 | 0.826 | 0.953 | 0.953 |
| capital | statement_is | 6 | 0.804 | 0.940 | 0.940 |
| capital | question | 7 | 0.804 | 0.932 | 0.932 |
| continent | statement_is | 11 | 0.808 | 0.846 | 0.846 |
| element_symbol | answer | 3 | 0.208 | 0.181 | 0.819 |
| landmark_country | statement_is | 9 | 0.778 | 0.815 | 0.815 |
| element_symbol | statement_is | 3 | 0.292 | 0.208 | 0.792 |
| book_author | answer | 11 | 0.278 | 0.210 | 0.790 |
| continent | answer | 5 | 0.654 | 0.787 | 0.787 |
| book_author | question | 11 | 0.278 | 0.235 | 0.765 |

## 首都任务重点探针

按 AUC 排序的最高层：

| 层 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- |
| 8 | 0.826 | 0.953 | 0.953 |
| 10 | 0.870 | 0.947 | 0.947 |
| 6 | 0.826 | 0.943 | 0.943 |
| 9 | 0.804 | 0.941 | 0.941 |
| 5 | 0.848 | 0.941 | 0.941 |

按准确率排序的最高层：

| 层 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- |
| 10 | 0.870 | 0.947 | 0.947 |
| 5 | 0.848 | 0.941 | 0.941 |
| 7 | 0.848 | 0.938 | 0.938 |
| 6 | 0.826 | 0.943 | 0.943 |
| 8 | 0.826 | 0.953 | 0.953 |

## 探针随机种子敏感性

| 层 | 平均准确率 | 准确率标准差 | 平均 AUC | AUC 标准差 | 最小 AUC | 最大 AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.790 | 0.040 | 0.899 | 0.039 | 0.832 | 0.953 |
| 5 | 0.779 | 0.069 | 0.878 | 0.046 | 0.822 | 0.941 |
| 10 | 0.797 | 0.045 | 0.873 | 0.047 | 0.824 | 0.947 |
| 11 | 0.754 | 0.038 | 0.848 | 0.038 | 0.800 | 0.902 |

## 词汇平衡首都探针

| 层 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- |
| 6 | 0.625 | 0.809 | 0.809 |
| 8 | 0.625 | 0.802 | 0.802 |
| 7 | 0.667 | 0.783 | 0.783 |
| 10 | 0.667 | 0.778 | 0.778 |
| 9 | 0.625 | 0.771 | 0.771 |
| 4 | 0.625 | 0.759 | 0.759 |

## 词汇平衡表面特征基线

| 领域 | 基线 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- | --- |
| capital_balanced | numeric_surface | 0.500 | 0.500 | 0.500 |
| capital_balanced | bag_of_words | 0.500 | 0.500 | 0.500 |

## 词汇平衡探针随机种子敏感性

| 层 | 平均准确率 | 平均 AUC | 最小 AUC | 最大 AUC |
| --- | --- | --- | --- | --- |
| 6 | 0.740 | 0.813 | 0.750 | 0.845 |
| 8 | 0.733 | 0.804 | 0.781 | 0.825 |
| 10 | 0.722 | 0.782 | 0.773 | 0.795 |
| 11 | 0.715 | 0.755 | 0.717 | 0.807 |
| 4 | 0.677 | 0.752 | 0.695 | 0.786 |

## 首都补全得分差基线

`grouping_margin_mean` 是用于定义或汇总该行分组的得分差列均值；对于 `residual_probe` 行，它不是平均探针分数。

`heldout_high_avg_token_margin` 和 `heldout_low_avg_token_margin` 是按平均词元得分差事后划分的探索性子集，不用于确认性结论。

| 分析 | 分组 | 样本数 | 数据块数 | 准确率 | AUC | AUC CI 下界 | AUC CI 上界 | 方向无关 AUC | 分组得分差列 | 分组得分差均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| completion_total | heldout_rows | 48 | 12 | 0.750 | 0.861 | 0.753 | 0.955 | 0.861 | completion_total_margin | 2.821 |
| completion_avg_token | heldout_rows | 48 | 12 | 0.625 | 0.786 | 0.674 | 0.891 | 0.786 | completion_avg_token_margin | 2.079 |
| residual_probe | heldout_rows | 48 | 12 | 0.625 | 0.809 | 0.708 | 0.922 | 0.809 | completion_avg_token_margin | 2.079 |
| completion_total | heldout_high_avg_token_margin | 24 | 11 | 0.833 | 0.944 | 0.750 | 1.000 | 0.944 | completion_total_margin | 4.289 |
| completion_avg_token | heldout_high_avg_token_margin | 24 | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | completion_avg_token_margin | 4.925 |
| residual_probe | heldout_high_avg_token_margin | 24 | 11 | 0.583 | 0.764 | 0.579 | 0.929 | 0.764 | completion_avg_token_margin | 4.925 |
| completion_total | heldout_low_avg_token_margin | 24 | 11 | 0.667 | 0.750 | 0.437 | 0.970 | 0.750 | completion_total_margin | 1.353 |
| completion_avg_token | heldout_low_avg_token_margin | 24 | 11 | 0.250 | 0.271 | 0.000 | 0.563 | 0.729 | completion_avg_token_margin | -0.767 |
| residual_probe | heldout_low_avg_token_margin | 24 | 11 | 0.667 | 0.847 | 0.681 | 1.000 | 0.847 | completion_avg_token_margin | -0.767 |

## 探索性与补充诊断

### 激活 PCA

| 层 | PC1 解释方差 | PC2 解释方差 | 样本数 |
| --- | --- | --- | --- |
| 8 | 0.620 | 0.117 | 152 |

### 输出读出基线

| 领域 | 标签词 | 提示词 | 示例数 | 准确率 | AUC | 预测 true 比例 | 平均 logit 差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | yes_no | answer_yes_no | 0 | 0.502 | 0.542 | 0.009 | -1.995 |
| all | correct_incorrect | statement_correct | 0 | 0.500 | 0.509 | 1.000 | 2.056 |
| all | lower_true_false | statement_is | 0 | 0.500 | 0.499 | 1.000 | 1.578 |
| all | title_true_false | answer_True_False | 2 | 0.500 | 0.494 | 1.000 | 1.505 |
| capital | yes_no | answer_yes_no | 2 | 0.507 | 0.560 | 0.007 | -1.976 |
| capital | correct_incorrect | statement_correct | 0 | 0.500 | 0.495 | 1.000 | 2.052 |
| capital | lower_true_false | answer_true_false | 2 | 0.500 | 0.486 | 1.000 | 1.607 |
| capital | title_true_false | answer_True_False | 0 | 0.500 | 0.530 | 1.000 | 1.510 |

### 跨领域方向迁移

| 源领域 | 目标领域 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- | --- |
| continent | capital | 0.638 | 0.766 | 0.766 |
| landmark_country | capital | 0.539 | 0.735 | 0.735 |
| landmark_country | continent | 0.628 | 0.654 | 0.654 |
| science | capital | 0.526 | 0.648 | 0.648 |
| capital | continent | 0.488 | 0.633 | 0.633 |
| continent | landmark_country | 0.517 | 0.612 | 0.612 |
| capital | landmark_country | 0.550 | 0.574 | 0.574 |
| capital | science | 0.520 | 0.573 | 0.573 |
| science | book_author | 0.567 | 0.572 | 0.572 |
| capital | element_symbol | 0.512 | 0.571 | 0.571 |

### 领域方向余弦相似度汇总

| 平均跨领域余弦 | 最小跨领域余弦 | 最大跨领域余弦 |
| --- | --- | --- |
| 0.077 | -0.043 | 0.301 |

### 错误分析

| 测试样本数 | 正确数 | 错误数 | 准确率 |
| --- | --- | --- | --- |
| 46 | 38 | 8 | 0.826 |

误分类样本：

| 陈述 | 标签 | 预测 | true 概率 |
| --- | --- | --- | --- |
| The capital of Laos is Vientiane. | true | false | 0.003 |
| The capital of Canada is Amman. | false | true | 0.964 |
| The capital of Chile is Santiago. | true | false | 0.179 |
| The capital of India is New Delhi. | true | false | 0.217 |
| The capital of Morocco is Rabat. | true | false | 0.219 |
| The capital of Nigeria is Mexico City. | false | true | 0.688 |
| The capital of Kenya is Nairobi. | true | false | 0.347 |
| The capital of Nepal is Kathmandu. | true | false | 0.472 |

### 激活修补：各组件最佳层

| 组件 | 层 | 平均恢复率 | 恢复率中位数 | 修补后 logit 差 |
| --- | --- | --- | --- | --- |
| attn_out | 11 | 1.762 | 1.077 | -0.672 |
| resid_post | 11 | 1.000 | 1.000 | 0.264 |
| mlp_out | 7 | 0.388 | 0.074 | -1.022 |

### 事实验证激活修补

| 组件 | 层 | 平均恢复率 | 恢复率中位数 | 修补后 logit 差 | 平均绝对 logit 变化 | 平均绝对分母 |
| --- | --- | --- | --- | --- | --- | --- |
| resid_post | 11 | 1.000 | 1.000 | 1.547 | 0.076 | 0.076 |
| resid_post | 10 | 0.816 | 0.814 | 1.540 | 0.068 | 0.076 |
| resid_pre | 11 | 0.816 | 0.814 | 1.540 | 0.068 | 0.076 |
| resid_post | 9 | 0.607 | 0.622 | 1.538 | 0.055 | 0.076 |
| resid_pre | 10 | 0.607 | 0.622 | 1.538 | 0.055 | 0.076 |
| resid_post | 8 | 0.568 | 0.541 | 1.536 | 0.052 | 0.076 |
| resid_pre | 9 | 0.568 | 0.541 | 1.536 | 0.052 | 0.076 |
| mlp_out | 1 | 0.330 | 0.155 | 1.521 | 0.014 | 0.076 |

### 探针方向激活干预

| alpha | logit 符号准确率 | 留出探针阈值准确率 | 平均探针分数 | 划分 | 阈值来源 |
| --- | --- | --- | --- | --- | --- |
| -8.0 | 0.500 | 0.500 | -9.560 | group | train_midpoint |
| -4.0 | 0.500 | 0.522 | -5.560 | group | train_midpoint |
| -2.0 | 0.500 | 0.522 | -3.560 | group | train_midpoint |
| -1.0 | 0.500 | 0.522 | -2.560 | group | train_midpoint |
| 0.0 | 0.500 | 0.826 | -1.560 | group | train_midpoint |
| 1.0 | 0.500 | 0.500 | -0.560 | group | train_midpoint |
| 2.0 | 0.500 | 0.500 | 0.440 | group | train_midpoint |
| 4.0 | 0.500 | 0.500 | 2.440 | group | train_midpoint |
| 8.0 | 0.500 | 0.500 | 6.440 | group | train_midpoint |

### Oracle 条件干预

| alpha | logit 符号准确率 | 探针阈值准确率 | 平均正确 logit 得分差 | 平均正确探针得分差 | 模式 |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.500 | 0.826 | -0.025 | 0.286 | oracle_label_conditioned |
| 0.5 | 0.500 | 1.000 | -0.029 | 0.786 | oracle_label_conditioned |
| 1.0 | 0.500 | 1.000 | -0.033 | 1.286 | oracle_label_conditioned |
| 2.0 | 0.500 | 1.000 | -0.041 | 2.286 | oracle_label_conditioned |
| 4.0 | 0.500 | 1.000 | -0.057 | 4.286 | oracle_label_conditioned |
| 8.0 | 0.500 | 1.000 | -0.088 | 8.286 | oracle_label_conditioned |

## 词汇平衡主线干预结果

### 词汇平衡提示词末位置补全得分差干预

| 方向 | alpha | 平均词元得分差变化 | 变化 CI | 配对偏好准确率 | 数据块完全正确率 |
| --- | --- | --- | --- | --- | --- |
| learned_probe | -4.0 | -0.130 | [-0.191, -0.072] | 0.625 | 0.250 |
| learned_probe | -2.0 | -0.066 | [-0.096, -0.036] | 0.625 | 0.250 |
| learned_probe | 0.0 | 0.000 | [0.000, 0.000] | 0.625 | 0.250 |
| learned_probe | 2.0 | 0.067 | [0.038, 0.098] | 0.625 | 0.250 |
| learned_probe | 4.0 | 0.135 | [0.077, 0.198] | 0.625 | 0.250 |
| random_direction | -4.0 | 0.029 | [0.017, 0.041] | 0.625 | 0.250 |
| random_direction | -2.0 | 0.015 | [0.008, 0.021] | 0.625 | 0.250 |
| random_direction | 0.0 | 0.000 | [0.000, 0.000] | 0.625 | 0.250 |
| random_direction | 2.0 | -0.015 | [-0.021, -0.008] | 0.625 | 0.250 |
| random_direction | 4.0 | -0.030 | [-0.042, -0.017] | 0.625 | 0.250 |
| label_permutation | -4.0 | 0.023 | [0.013, 0.035] | 0.625 | 0.250 |
| label_permutation | -2.0 | 0.012 | [0.007, 0.018] | 0.625 | 0.250 |
| label_permutation | 0.0 | 0.000 | [0.000, 0.000] | 0.625 | 0.250 |
| label_permutation | 2.0 | -0.011 | [-0.017, -0.006] | 0.625 | 0.250 |
| label_permutation | 4.0 | -0.022 | [-0.034, -0.013] | 0.625 | 0.250 |

### 提示词末位置补全得分差干预分解

| 方向 | alpha | 正确补全 logprob 变化 | 错误补全 logprob 变化 | 得分差变化 | 得分差变化标准差 | 基线正确时变化 | 基线错误时变化 | 正确-错误变化差 | 变化差 CI | 符号翻转数 | 基线得分差与变化相关 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_permutation | -4.0 | 0.073 | 0.049 | 0.023 | 0.045 | 0.021 | 0.026 | -0.005 | [-0.038, 0.029] | 0 | 0.249 |
| label_permutation | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0 | nan |
| label_permutation | 4.0 | -0.072 | -0.050 | -0.022 | 0.046 | -0.021 | -0.025 | 0.004 | [-0.029, 0.038] | 0 | -0.278 |
| learned_probe | -4.0 | -0.287 | -0.157 | -0.130 | 0.131 | -0.126 | -0.136 | 0.010 | [-0.087, 0.117] | 0 | -0.030 |
| learned_probe | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0 | nan |
| learned_probe | 4.0 | 0.281 | 0.146 | 0.135 | 0.133 | 0.132 | 0.140 | -0.008 | [-0.110, 0.084] | 0 | 0.001 |
| random_direction | -4.0 | 0.004 | -0.025 | 0.029 | 0.050 | 0.029 | 0.030 | -0.002 | [-0.045, 0.039] | 0 | 0.281 |
| random_direction | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0 | nan |
| random_direction | 4.0 | -0.010 | 0.019 | -0.030 | 0.049 | -0.028 | -0.031 | 0.003 | [-0.038, 0.043] | 0 | -0.263 |

### 提示词末位置补全干预配对自助法

| 指标 | 比较 | 估计值 | CI | CI 单位 |
| --- | --- | --- | --- | --- |
| delta_avg_token_margin_alpha_4 | learned_probe_minus_random_direction | 0.165 | [0.096, 0.239] | pair_id_block |
| delta_avg_token_margin_alpha_4 | learned_probe_minus_label_permutation | 0.158 | [0.090, 0.232] | pair_id_block |
| slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_random_direction | 0.040 | [0.024, 0.058] | pair_id_block |
| slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_label_permutation | 0.039 | [0.023, 0.057] | pair_id_block |

### 补全干预位置分解

| 干预位置模式 | alpha | 平均词元得分差变化 | 配对偏好准确率 | 数据块完全正确率 |
| --- | --- | --- | --- | --- |
| all | -4.0 | -0.126 | 0.625 | 0.250 |
| all | 0.0 | 0.000 | 0.625 | 0.250 |
| all | 4.0 | 0.133 | 0.625 | 0.250 |
| prompt-final-only | -4.0 | -0.130 | 0.625 | 0.250 |
| prompt-final-only | 0.0 | 0.000 | 0.625 | 0.250 |
| prompt-final-only | 4.0 | 0.135 | 0.625 | 0.250 |
| completion-internal-only | -4.0 | 0.003 | 0.625 | 0.250 |
| completion-internal-only | 0.0 | 0.000 | 0.625 | 0.250 |
| completion-internal-only | 4.0 | -0.002 | 0.625 | 0.250 |

### 位置分解配对自助法

| 干预位置模式 | 指标 | 比较 | 估计值 | CI |
| --- | --- | --- | --- | --- |
| prompt-final-only | delta_avg_token_margin_alpha_4 | learned_probe_minus_random_direction | 0.165 | [0.096, 0.239] |
| prompt-final-only | delta_avg_token_margin_alpha_4 | learned_probe_minus_label_permutation | 0.158 | [0.090, 0.232] |
| prompt-final-only | slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_random_direction | 0.040 | [0.024, 0.058] |
| prompt-final-only | slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_label_permutation | 0.039 | [0.023, 0.057] |
| completion-internal-only | delta_avg_token_margin_alpha_4 | learned_probe_minus_random_direction | -0.000 | [-0.003, 0.004] |
| completion-internal-only | delta_avg_token_margin_alpha_4 | learned_probe_minus_label_permutation | 0.001 | [-0.004, 0.008] |
| completion-internal-only | slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_random_direction | 0.000 | [-0.001, 0.001] |
| completion-internal-only | slope_delta_avg_token_margin_-4_to_4 | learned_probe_minus_label_permutation | 0.000 | [-0.001, 0.002] |

### 补全干预零分布

| 对照类型 | 方向数 | 平均变化 | 零分布 95% 区间 | 学习方向效果 | 学习方向分位数 | 经验 p 值 |
| --- | --- | --- | --- | --- | --- | --- |
| label_permutation | 20 | 0.010 | [-0.081, 0.088] | 0.135 | 1.000 | 0.048 |
| learned_probe | 1 | 0.135 | [0.135, 0.135] | 0.135 | 1.000 |  |
| random_direction | 50 | 0.009 | [-0.065, 0.091] | 0.135 | 1.000 | 0.020 |

### 重复划分补全干预

| 范围 | 划分数 | 学习方向变化 | 学习方向变化标准差 | 学习方向变化范围 | 学习-随机均值 | 学习-置乱均值 | 强于全部随机方向的划分数 | 强于全部置乱方向的划分数 | 基线配对准确率 | 平均配对准确率 | 配对准确率变化 | 总符号翻转数 | 错误转正确次数 | 正确转错误次数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggregate | 10 | 0.116 | 0.023 | [0.085, 0.150] | 0.125 | 0.119 | 10 | 10 | 0.700 | 0.725 | 0.025 | 6 | 6 | 0 |

### 争议事实敏感性

| 分析 | 数据块数 | 留出数据块数 | AUC | 变化 | 配对准确率 | 符号翻转数 |
| --- | --- | --- | --- | --- | --- | --- |
| dataset | 35 |  |  |  |  |  |
| residual_probe |  | 11 | 0.864 |  |  |  |
| completion_total |  | 11 | 0.959 |  |  |  |
| completion_avg_token |  | 11 | 0.913 |  |  |  |
| prompt_final_steering:learned_probe |  | 11 |  | 0.120 | 0.864 | 1 |
| prompt_final_steering:random_direction |  | 11 |  | -0.038 | 0.818 | 0 |
| prompt_final_steering:label_permutation |  | 11 |  | -0.005 | 0.818 | 0 |

### 候选集排名干预

| 留出国家数 | 候选数 | 平均排名变化 | 排名改善数 | 排名变差数 | 基线 top-1 准确率 | 干预后 top-1 准确率 | top-1 改变数 | 选定配对得分差变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 76 | 0.917 | 10 | 0 | 0.083 | 0.125 | 1 | 0.135 |

### 全位置 unembedding 投影基线

| 方向 | 观测均值 | 预测均值 | 观测绝对均值 | 预测绝对均值 | 平均绝对残差 | 相关系数 | 决定系数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| label_permutation | -0.026 | 0.000 | 0.042 | 0.464 | 0.452 | 0.381 | 0.145 |
| learned_probe | 0.133 | 0.000 | 0.153 | 0.621 | 0.669 | -0.188 | 0.035 |
| random_direction | -0.032 | 0.000 | 0.047 | 0.598 | 0.577 | 0.627 | 0.393 |

### 探针方向消融

| 强度 | 固定方向分数差 | 固定方向准确率 | 重训探针 AUC |
| --- | --- | --- | --- |
| 0.00 | 0.573 | 0.826 | 0.953 |
| 0.25 | 0.430 | 0.500 | 0.955 |
| 0.50 | 0.286 | 0.500 | 0.958 |
| 0.75 | 0.143 | 0.500 | 0.951 |
| 1.00 | -0.000 | 0.500 | 0.945 |
| 1.25 | -0.143 | 0.500 | 0.953 |
| 1.50 | -0.286 | 0.478 | 0.962 |

### 词汇平衡探针方向消融

| 强度 | 固定方向分数差 | 固定方向准确率 | 重训探针 AUC |
| --- | --- | --- | --- |
| 0.00 | 0.174 | 0.604 | 0.809 |
| 0.25 | 0.130 | 0.500 | 0.811 |
| 0.50 | 0.087 | 0.500 | 0.809 |
| 0.75 | 0.043 | 0.500 | 0.788 |
| 1.00 | 0.000 | 0.500 | 0.786 |
| 1.25 | -0.043 | 0.500 | 0.792 |
| 1.50 | -0.087 | 0.500 | 0.793 |

### 迭代方向消融

| 对照 | 移除方向数 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- | --- |
| learned_iterative | 0 | 0.826 | 0.953 | 0.953 |
| learned_iterative | 1 | 0.826 | 0.945 | 0.945 |
| learned_iterative | 2 | 0.870 | 0.941 | 0.941 |
| learned_iterative | 4 | 0.848 | 0.911 | 0.911 |
| learned_iterative | 8 | 0.783 | 0.862 | 0.862 |
| learned_iterative | 12 | 0.739 | 0.832 | 0.832 |
| learned_iterative | 16 | 0.739 | 0.807 | 0.807 |
| random_direction | 0 | 0.826 | 0.953 | 0.953 |
| random_direction | 1 | 0.826 | 0.953 | 0.953 |
| random_direction | 2 | 0.826 | 0.953 | 0.953 |
| random_direction | 4 | 0.826 | 0.949 | 0.949 |
| random_direction | 8 | 0.826 | 0.953 | 0.953 |
| random_direction | 12 | 0.826 | 0.953 | 0.953 |
| random_direction | 16 | 0.826 | 0.951 | 0.951 |
| label_permutation | 0 | 0.826 | 0.953 | 0.953 |
| label_permutation | 1 | 0.826 | 0.953 | 0.953 |
| label_permutation | 2 | 0.826 | 0.953 | 0.953 |
| label_permutation | 4 | 0.826 | 0.953 | 0.953 |
| label_permutation | 8 | 0.826 | 0.945 | 0.945 |
| label_permutation | 12 | 0.848 | 0.945 | 0.945 |
| label_permutation | 16 | 0.848 | 0.953 | 0.953 |

### 词汇平衡迭代方向消融

| 对照 | 移除方向数 | 准确率 | AUC | 方向无关 AUC |
| --- | --- | --- | --- | --- |
| learned_iterative | 0 | 0.625 | 0.809 | 0.809 |
| learned_iterative | 1 | 0.604 | 0.786 | 0.786 |
| learned_iterative | 2 | 0.688 | 0.778 | 0.778 |
| learned_iterative | 4 | 0.646 | 0.755 | 0.755 |
| learned_iterative | 8 | 0.688 | 0.757 | 0.757 |
| learned_iterative | 12 | 0.625 | 0.738 | 0.738 |
| learned_iterative | 16 | 0.646 | 0.726 | 0.726 |
| random_direction | 0 | 0.625 | 0.809 | 0.809 |
| random_direction | 1 | 0.625 | 0.809 | 0.809 |
| random_direction | 2 | 0.604 | 0.806 | 0.806 |
| random_direction | 4 | 0.625 | 0.804 | 0.804 |
| random_direction | 8 | 0.604 | 0.806 | 0.806 |
| random_direction | 12 | 0.625 | 0.797 | 0.797 |
| random_direction | 16 | 0.646 | 0.793 | 0.793 |
| label_permutation | 0 | 0.625 | 0.809 | 0.809 |
| label_permutation | 1 | 0.625 | 0.812 | 0.812 |
| label_permutation | 2 | 0.625 | 0.814 | 0.814 |
| label_permutation | 4 | 0.646 | 0.793 | 0.793 |
| label_permutation | 8 | 0.646 | 0.773 | 0.773 |
| label_permutation | 12 | 0.646 | 0.760 | 0.760 |
| label_permutation | 16 | 0.667 | 0.783 | 0.783 |
