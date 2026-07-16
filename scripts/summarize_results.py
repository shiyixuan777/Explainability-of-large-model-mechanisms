from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--out", default="reports/results_summary.md")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


HEADER_LABELS = {
    "claim": "结论",
    "key_result": "关键结果",
    "domain": "领域",
    "baseline": "基线",
    "prompt": "提示词",
    "layer": "层",
    "accuracy": "准确率",
    "auc": "AUC",
    "direction_agnostic_auc": "方向无关 AUC",
    "mean_accuracy": "平均准确率",
    "std_accuracy": "准确率标准差",
    "mean_auc": "平均 AUC",
    "std_auc": "AUC 标准差",
    "min_auc": "最小 AUC",
    "max_auc": "最大 AUC",
    "analysis": "分析",
    "group": "分组",
    "rows": "样本数",
    "blocks": "数据块数",
    "auc_ci_low": "AUC CI 下界",
    "auc_ci_high": "AUC CI 上界",
    "grouping_margin_column": "分组得分差列",
    "grouping_margin_mean": "分组得分差均值",
    "pc1_explained_variance": "PC1 解释方差",
    "pc2_explained_variance": "PC2 解释方差",
    "verbalizer": "标签词",
    "shots": "示例数",
    "predicted_true_rate": "预测 true 比例",
    "mean_logit_margin": "平均 logit 差",
    "source": "源领域",
    "target": "目标领域",
    "mean_cross_domain_cosine": "平均跨领域余弦",
    "min_cross_domain_cosine": "最小跨领域余弦",
    "max_cross_domain_cosine": "最大跨领域余弦",
    "test_rows": "测试样本数",
    "correct": "正确数",
    "wrong": "错误数",
    "statement": "陈述",
    "label": "标签",
    "prediction": "预测",
    "prob_true": "true 概率",
    "component": "组件",
    "mean_recovery": "平均恢复率",
    "median_recovery": "恢复率中位数",
    "patched_logit_diff": "修补后 logit 差",
    "mean_abs_logit_shift": "平均绝对 logit 变化",
    "mean_abs_denominator": "平均绝对分母",
    "alpha": "alpha",
    "logit_sign_accuracy": "logit 符号准确率",
    "heldout_probe_threshold_accuracy": "留出探针阈值准确率",
    "probe_threshold_accuracy": "探针阈值准确率",
    "mean_probe_score": "平均探针分数",
    "split": "划分",
    "threshold_source": "阈值来源",
    "mean_logit_correct_margin": "平均正确 logit 得分差",
    "mean_probe_correct_margin": "平均正确探针得分差",
    "mode": "模式",
    "direction": "方向",
    "mean_delta_avg_token_margin": "平均词元得分差变化",
    "delta_ci": "变化 CI",
    "pairwise_avg_accuracy": "配对偏好准确率",
    "block_exact_accuracy": "数据块完全正确率",
    "delta_correct_logprob": "正确补全 logprob 变化",
    "delta_false_logprob": "错误补全 logprob 变化",
    "delta_margin": "得分差变化",
    "delta_margin_std": "得分差变化标准差",
    "baseline_correct_shift": "基线正确时变化",
    "baseline_wrong_shift": "基线错误时变化",
    "baseline_correct_minus_wrong": "正确-错误变化差",
    "baseline_diff_ci": "变化差 CI",
    "sign_flips": "符号翻转数",
    "baseline_delta_corr": "基线得分差与变化相关",
    "metric": "指标",
    "comparison": "比较",
    "estimate": "估计值",
    "ci": "CI",
    "ci_unit": "CI 单位",
    "position_mode": "干预位置模式",
    "control_type": "对照类型",
    "directions": "方向数",
    "mean_delta": "平均变化",
    "null_95_interval": "零分布 95% 区间",
    "learned_effect": "学习方向效果",
    "learned_percentile": "学习方向分位数",
    "empirical_p_ge_learned": "经验 p 值",
    "scope": "范围",
    "splits": "划分数",
    "learned_delta": "学习方向变化",
    "learned_delta_std": "学习方向变化标准差",
    "learned_delta_range": "学习方向变化范围",
    "learned_minus_random_mean": "学习-随机均值",
    "learned_minus_permutation_mean": "学习-置乱均值",
    "learned_gt_all_random_splits": "强于全部随机方向的划分数",
    "learned_gt_all_permutation_splits": "强于全部置乱方向的划分数",
    "baseline_pairwise_accuracy": "基线配对准确率",
    "mean_pairwise_accuracy": "平均配对准确率",
    "pairwise_accuracy_change": "配对准确率变化",
    "total_sign_flips": "总符号翻转数",
    "wrong_to_correct_flips": "错误转正确次数",
    "correct_to_wrong_flips": "正确转错误次数",
    "heldout_blocks": "留出数据块数",
    "delta": "变化",
    "pairwise_accuracy": "配对准确率",
    "heldout_countries": "留出国家数",
    "candidate_count": "候选数",
    "mean_rank_delta": "平均排名变化",
    "rank_improved_count": "排名改善数",
    "rank_worsened_count": "排名变差数",
    "baseline_top1_accuracy": "基线 top-1 准确率",
    "steered_top1_accuracy": "干预后 top-1 准确率",
    "top1_changed_count": "top-1 改变数",
    "selected_pair_margin_delta": "选定配对得分差变化",
    "observed_mean": "观测均值",
    "predicted_mean": "预测均值",
    "observed_abs_mean": "观测绝对均值",
    "predicted_abs_mean": "预测绝对均值",
    "mean_abs_residual": "平均绝对残差",
    "corr": "相关系数",
    "corr_squared": "决定系数",
    "strength": "强度",
    "fixed_direction_score_gap": "固定方向分数差",
    "fixed_direction_accuracy": "固定方向准确率",
    "retrained_probe_auc": "重训探针 AUC",
    "control": "对照",
    "directions_removed": "移除方向数",
}


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_无可用数据。_\n"
    headers = list(rows[0].keys())
    display_headers = [HEADER_LABELS.get(header, header) for header in headers]
    table = [
        "| " + " | ".join(display_headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(table)


def fmt(value: float | int | str, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def fmt_label(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()


def git_metadata() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unavailable"

    try:
        subprocess.check_call(
            ["git", "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty = "no"
    except Exception:
        dirty = "yes"
    return commit, dirty


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    out_path = Path(args.out)
    commit, dirty = git_metadata()

    lines: list[str] = [
        "# 结果汇总",
        "",
        "本文件由 `python -m scripts.summarize_results` 根据 CSV 结果自动生成。",
        "它用于作为报告结果的紧凑表格索引。",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"生成时 Git commit：{commit}",
        f"生成时工作树是否有未提交修改：{dirty}",
        "源目录：项目根目录",
        "生成脚本：`scripts/summarize_results.py`",
        "",
        "`direction_agnostic_auc = max(AUC, 1 - AUC)`。它诊断预测分数与标签之间是否存在强排序关系，而不关心方向符号；它不表示训练时学到的标签方向一定能作为分类器稳定泛化。",
        "",
        "`learned_percentile = 1.0` 表示在当前采样集合中没有零分布方向超过学习方向效果；它不是总体分位数估计。`mean_rank_delta > 0` 表示正确候选向 rank 1 移动。重复划分中的翻转计数是跨重叠划分的评价事件，不一定对应互不重复的国家。",
        "",
    ]

    core_rows: list[dict[str, object]] = []
    original_probe_for_core = read_csv(figures_dir / "probe_capital_answer.csv")
    surface_for_core = read_csv(figures_dir / "surface_baselines.csv")
    if original_probe_for_core is not None and surface_for_core is not None:
        layer8 = original_probe_for_core[original_probe_for_core["layer"] == 8]
        bow = surface_for_core[
            (surface_for_core["domain"] == "capital") & (surface_for_core["baseline"] == "bag_of_words")
        ]
        if not layer8.empty and not bow.empty:
            core_rows.append(
                {
                    "claim": "原始数据存在词汇混杂",
                    "key_result": (
                        f"第 8 层残差流 AUC {fmt(layer8.iloc[0]['auc'])}；"
                        f"BOW 方向无关 AUC {fmt(bow.iloc[0]['separability_auc'])}"
                    ),
                }
            )
    balanced_probe_for_core = read_csv(figures_dir / "probe_capital_balanced.csv")
    if balanced_probe_for_core is not None:
        layer6 = balanced_probe_for_core[balanced_probe_for_core["layer"] == 6]
        if not layer6.empty:
            core_rows.append({"claim": "词汇平衡后仍可读出", "key_result": f"第 6 层 AUC {fmt(layer6.iloc[0]['auc'])}"})
    prompt_final_for_core = read_csv(figures_dir / "completion_margin_steering_position_prompt_final_summary.csv")
    if prompt_final_for_core is not None:
        row = prompt_final_for_core[
            (prompt_final_for_core["split"] == "heldout_countries")
            & (prompt_final_for_core["direction_type"] == "learned_probe")
            & (prompt_final_for_core["alpha"] == 4.0)
        ]
        if not row.empty:
            core_rows.append(
                {
                    "claim": "评分层面干预",
                    "key_result": f"提示词末位置变化 {fmt(row.iloc[0]['mean_delta_avg_token_margin'])}",
                }
            )
    repeated_for_core = read_csv(figures_dir / "repeated_split_completion_steering_summary.csv")
    if repeated_for_core is not None:
        aggregate = repeated_for_core[repeated_for_core["seed"].astype(str) == "aggregate"]
        split_rows_for_core = repeated_for_core[repeated_for_core["seed"].astype(str) != "aggregate"]
        if not aggregate.empty:
            row = aggregate.iloc[0]
            positive_splits = int((split_rows_for_core["learned_delta"] > 0).sum()) if not split_rows_for_core.empty else 0
            split_count = len(split_rows_for_core)
            core_rows.append(
                {
                    "claim": "重复划分稳定性",
                    "key_result": f"{positive_splits}/{split_count} 个划分为正；均值 {fmt(row.learned_delta)}",
                }
            )
            core_rows.append(
                {
                    "claim": "选择层面效果",
                    "key_result": (
                        f"配对准确率变化 {fmt(row.pairwise_accuracy_change)}；"
                        f"错误转正确事件 {int(row.wrong_to_correct_flips)}"
                    ),
                }
            )
    rank_for_core = read_csv(figures_dir / "candidate_rank_steering_summary.csv")
    if rank_for_core is not None and not rank_for_core.empty:
        row = rank_for_core.iloc[0]
        core_rows.append(
            {
                "claim": "候选集 top-1",
                "key_result": f"{fmt(row.baseline_top1_accuracy)} -> {fmt(row.steered_top1_accuracy)}",
            }
        )
    balanced_ablation_for_core = read_csv(figures_dir / "ablation_capital_balanced_layer6.csv")
    if balanced_ablation_for_core is not None:
        strength1 = balanced_ablation_for_core[balanced_ablation_for_core["strength"] == 1.0]
        if not strength1.empty:
            core_rows.append(
                {
                    "claim": "机制边界",
                    "key_result": f"单方向消融后重训 AUC {fmt(strength1.iloc[0]['auc'])}",
                }
            )
    if core_rows:
        lines += ["## 核心结果索引", "", markdown_table(core_rows), ""]

    surface = read_csv(figures_dir / "surface_baselines.csv")
    if surface is not None:
        rows = [
            {
                "domain": row.domain,
                "baseline": row.baseline,
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in surface.itertuples(index=False)
        ]
        lines += ["## 原始数据表面特征基线", "", markdown_table(rows), ""]

    sweep = read_csv(figures_dir / "probe_sweep.csv")
    if sweep is not None:
        best = (
            sweep.sort_values("separability_auc", ascending=False)
            .groupby(["domain", "prompt"], as_index=False)
            .first()
            .sort_values("separability_auc", ascending=False)
            .head(10)
        )
        rows = [
            {
                "domain": row.domain,
                "prompt": row.prompt,
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in best.itertuples(index=False)
        ]
        lines += ["## 探针扫描：按方向无关 AUC 排序的最高设置", "", markdown_table(rows), ""]

    probe = read_csv(figures_dir / "probe_capital_answer.csv")
    if probe is not None:
        top_auc = probe.sort_values("auc", ascending=False).head(5)
        top_acc = probe.sort_values("accuracy", ascending=False).head(5)
        rows_auc = [
            {
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in top_auc.itertuples(index=False)
        ]
        rows_acc = [
            {
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in top_acc.itertuples(index=False)
        ]
        lines += [
            "## 首都任务重点探针",
            "",
            "按 AUC 排序的最高层：",
            "",
            markdown_table(rows_auc),
            "",
            "按准确率排序的最高层：",
            "",
            markdown_table(rows_acc),
            "",
        ]

    probe_seeds = read_csv(figures_dir / "probe_seed_sensitivity_capital.csv")
    if probe_seeds is not None:
        summary = (
            probe_seeds.groupby("layer", as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                mean_auc=("auc", "mean"),
                std_auc=("auc", "std"),
                min_auc=("auc", "min"),
                max_auc=("auc", "max"),
            )
            .sort_values("mean_auc", ascending=False)
        )
        rows = [
            {
                "layer": int(row.layer),
                "mean_accuracy": fmt(row.mean_accuracy),
                "std_accuracy": fmt(row.std_accuracy),
                "mean_auc": fmt(row.mean_auc),
                "std_auc": fmt(row.std_auc),
                "min_auc": fmt(row.min_auc),
                "max_auc": fmt(row.max_auc),
            }
            for row in summary.itertuples(index=False)
        ]
        lines += ["## 探针随机种子敏感性", "", markdown_table(rows), ""]

    balanced_probe = read_csv(figures_dir / "probe_capital_balanced.csv")
    balanced_surface = read_csv(figures_dir / "surface_baselines_capital_balanced.csv")
    balanced_seeds = read_csv(figures_dir / "probe_seed_sensitivity_capital_balanced.csv")
    if balanced_probe is not None:
        top_balanced = balanced_probe.sort_values("auc", ascending=False).head(6)
        rows = [
            {
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in top_balanced.itertuples(index=False)
        ]
        lines += ["## 词汇平衡首都探针", "", markdown_table(rows), ""]

    if balanced_surface is not None:
        rows = [
            {
                "domain": row.domain,
                "baseline": row.baseline,
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in balanced_surface.itertuples(index=False)
        ]
        lines += ["## 词汇平衡表面特征基线", "", markdown_table(rows), ""]

    if balanced_seeds is not None:
        summary = (
            balanced_seeds.groupby("layer", as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                mean_auc=("auc", "mean"),
                min_auc=("auc", "min"),
                max_auc=("auc", "max"),
            )
            .sort_values("mean_auc", ascending=False)
        )
        rows = [
            {
                "layer": int(row.layer),
                "mean_accuracy": fmt(row.mean_accuracy),
                "mean_auc": fmt(row.mean_auc),
                "min_auc": fmt(row.min_auc),
                "max_auc": fmt(row.max_auc),
            }
            for row in summary.itertuples(index=False)
        ]
        lines += ["## 词汇平衡探针随机种子敏感性", "", markdown_table(rows), ""]

    knowledge = read_csv(figures_dir / "capital_knowledge_margin_summary.csv")
    if knowledge is not None:
        selected = knowledge[
            knowledge["group"].isin(
                [
                    "heldout_rows",
                    "heldout_high_avg_token_margin",
                    "heldout_low_avg_token_margin",
                ]
            )
        ]
        rows = [
            {
                "analysis": row.analysis,
                "group": row.group,
                "rows": int(row.rows),
                "blocks": int(getattr(row, "blocks", 0)),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "auc_ci_low": fmt(getattr(row, "auc_ci_low", 0.0)),
                "auc_ci_high": fmt(getattr(row, "auc_ci_high", 0.0)),
                "direction_agnostic_auc": fmt(row.separability_auc),
                "grouping_margin_column": getattr(row, "margin_column", ""),
                "grouping_margin_mean": fmt(row.mean_margin),
            }
            for row in selected.itertuples(index=False)
        ]
        lines += [
            "## 首都补全得分差基线",
            "",
            "`grouping_margin_mean` 是用于定义或汇总该行分组的得分差列均值；对于 `residual_probe` 行，它不是平均探针分数。",
            "",
            "`heldout_high_avg_token_margin` 和 `heldout_low_avg_token_margin` 是按平均词元得分差事后划分的探索性子集，不用于确认性结论。",
            "",
            markdown_table(rows),
            "",
        ]

    lines += ["## 探索性与补充诊断", ""]

    pca = read_csv(figures_dir / "pca_capital_layer8.csv")
    if pca is not None:
        rows = [
            {
                "layer": int(pca["layer"].iloc[0]),
                "pc1_explained_variance": fmt(pca["pc1_explained_variance"].iloc[0]),
                "pc2_explained_variance": fmt(pca["pc2_explained_variance"].iloc[0]),
                "rows": len(pca),
            }
        ]
        lines += ["### 激活 PCA", "", markdown_table(rows), ""]

    readout = read_csv(figures_dir / "output_readout_baselines.csv")
    if readout is not None:
        usable = readout[readout["single_token_readout"] == True].copy()
        if not usable.empty:
            best = (
                usable.sort_values("accuracy_from_logit_sign", ascending=False)
                .groupby(["domain", "verbalizer"], as_index=False)
                .first()
                .sort_values(["domain", "accuracy_from_logit_sign"], ascending=[True, False])
            )
            rows = [
                {
                    "domain": row.domain,
                    "verbalizer": row.verbalizer,
                    "prompt": row.prompt_name,
                    "shots": int(row.shots),
                    "accuracy": fmt(row.accuracy_from_logit_sign),
                    "auc": fmt(row.auc),
                    "predicted_true_rate": fmt(row.predicted_true_rate),
                    "mean_logit_margin": fmt(row.mean_true_minus_false_logit_diff),
                }
                for row in best.itertuples(index=False)
            ]
            lines += ["### 输出读出基线", "", markdown_table(rows), ""]

    transfer = read_csv(figures_dir / "domain_transfer_layer8.csv")
    if transfer is not None:
        cross = transfer[transfer["source_domain"] != transfer["target_domain"]]
        if not cross.empty:
            best_cross = cross.sort_values("separability_auc", ascending=False).head(10)
            rows = [
                {
                    "source": row.source_domain,
                    "target": row.target_domain,
                    "accuracy": fmt(row.accuracy),
                    "auc": fmt(row.auc),
                    "direction_agnostic_auc": fmt(row.separability_auc),
                }
                for row in best_cross.itertuples(index=False)
            ]
            lines += ["### 跨领域方向迁移", "", markdown_table(rows), ""]

    cosine = read_csv(figures_dir / "domain_direction_cosine_layer8.csv")
    if cosine is not None:
        cross = cosine[cosine["source_domain"] != cosine["target_domain"]]
        if not cross.empty:
            rows = [
                {
                    "mean_cross_domain_cosine": fmt(cross["cosine_similarity"].mean()),
                    "min_cross_domain_cosine": fmt(cross["cosine_similarity"].min()),
                    "max_cross_domain_cosine": fmt(cross["cosine_similarity"].max()),
                }
            ]
            lines += ["### 领域方向余弦相似度汇总", "", markdown_table(rows), ""]

    errors = read_csv(figures_dir / "error_analysis_capital_layer8.csv")
    if errors is not None:
        total = len(errors)
        correct = int(errors["correct"].sum())
        wrong = total - correct
        rows = [
            {
                "test_rows": total,
                "correct": correct,
                "wrong": wrong,
                "accuracy": fmt(correct / total),
            }
        ]
        lines += ["### 错误分析", "", markdown_table(rows), ""]

        error_rows = errors.loc[~errors["correct"]].sort_values("confidence", ascending=False).head(8)
        examples = [
            {
                "statement": row.statement,
                "label": fmt_label(row.label_name),
                "prediction": fmt_label(row.predicted_name),
                "prob_true": fmt(row.prob_true),
            }
            for row in error_rows.itertuples(index=False)
        ]
        lines += ["误分类样本：", "", markdown_table(examples), ""]

    patching = read_csv(figures_dir / "activation_patching_capital_recall.csv")
    if patching is not None:
        best = (
            patching.sort_values("mean_recovery", ascending=False)
            .groupby("component", as_index=False)
            .first()
            .sort_values("mean_recovery", ascending=False)
        )
        rows = [
            {
                "component": row.component,
                "layer": int(row.layer),
                "mean_recovery": fmt(row.mean_recovery),
                "median_recovery": fmt(row.median_recovery),
                "patched_logit_diff": fmt(row.patched_logit_diff),
            }
            for row in best.itertuples(index=False)
        ]
        lines += ["### 激活修补：各组件最佳层", "", markdown_table(rows), ""]

    truth_patching = read_csv(figures_dir / "truth_verification_patching_resid.csv")
    if truth_patching is not None:
        if "control" in truth_patching.columns:
            truth_for_table = truth_patching[truth_patching["control"] == "matched_clean"]
        else:
            truth_for_table = truth_patching
        best = truth_for_table.sort_values("mean_recovery", ascending=False).head(8)
        rows = [
            {
                "component": getattr(row, "component", "resid_post"),
                "layer": int(row.layer),
                "mean_recovery": fmt(row.mean_recovery),
                "median_recovery": fmt(row.median_recovery),
                "patched_logit_diff": fmt(row.patched_true_minus_false_logit_diff),
                "mean_abs_logit_shift": fmt(row.mean_abs_logit_shift),
                "mean_abs_denominator": fmt(getattr(row, "mean_abs_clean_minus_corrupt_denominator", 0.0)),
            }
            for row in best.itertuples(index=False)
        ]
        lines += ["### 事实验证激活修补", "", markdown_table(rows), ""]

    steering = read_csv(figures_dir / "steering_capital_probe_layer8.csv")
    if steering is not None:
        rows = [
            {
                "alpha": fmt(row.alpha, digits=1),
                "logit_sign_accuracy": fmt(row.accuracy_from_logit_sign),
                "heldout_probe_threshold_accuracy": fmt(row.accuracy_from_probe_score_threshold),
                "mean_probe_score": fmt(row.mean_probe_score),
                "split": getattr(row, "split", ""),
                "threshold_source": getattr(row, "threshold_source", ""),
            }
            for row in steering.itertuples(index=False)
        ]
        lines += ["### 探针方向激活干预", "", markdown_table(rows), ""]

    oracle = read_csv(figures_dir / "oracle_steering_capital_probe_layer8.csv")
    if oracle is not None:
        rows = [
            {
                "alpha": fmt(row.alpha, digits=1),
                "logit_sign_accuracy": fmt(row.accuracy_from_logit_sign),
                "probe_threshold_accuracy": fmt(row.accuracy_from_probe_score_threshold),
                "mean_logit_correct_margin": fmt(row.mean_logit_correct_margin),
                "mean_probe_correct_margin": fmt(row.mean_probe_correct_margin),
                "mode": row.steering_mode,
            }
            for row in oracle.itertuples(index=False)
        ]
        lines += ["### Oracle 条件干预", "", markdown_table(rows), ""]

    lines += ["## 词汇平衡主线干预结果", ""]

    completion_steering = read_csv(figures_dir / "completion_margin_steering_position_prompt_final_summary.csv")
    if completion_steering is not None:
        heldout = completion_steering[
            (completion_steering["split"] == "heldout_countries")
            & (completion_steering["alpha"].isin([-4.0, -2.0, 0.0, 2.0, 4.0]))
        ]
        rows = [
            {
                "direction": row.direction_type,
                "alpha": fmt(row.alpha, digits=1),
                "mean_delta_avg_token_margin": fmt(row.mean_delta_avg_token_margin),
                "delta_ci": f"[{fmt(row.delta_avg_token_margin_ci_low)}, {fmt(row.delta_avg_token_margin_ci_high)}]",
                "pairwise_avg_accuracy": fmt(row.pairwise_avg_token_accuracy),
                "block_exact_accuracy": fmt(row.block_exact_avg_token_accuracy),
            }
            for row in heldout.itertuples(index=False)
        ]
        lines += ["### 词汇平衡提示词末位置补全得分差干预", "", markdown_table(rows), ""]

    completion_decomposition = read_csv(figures_dir / "completion_margin_steering_position_prompt_final_decomposition.csv")
    if completion_decomposition is not None:
        selected = completion_decomposition[
            (completion_decomposition["split"] == "heldout_countries")
            & (completion_decomposition["alpha"].isin([-4.0, 0.0, 4.0]))
        ]
        rows = [
            {
                "direction": row.direction_type,
                "alpha": fmt(row.alpha, digits=1),
                "delta_correct_logprob": fmt(row.mean_delta_correct_avg_token_logprob),
                "delta_false_logprob": fmt(row.mean_delta_false_avg_token_logprob),
                "delta_margin": fmt(row.mean_delta_avg_token_margin),
                "delta_margin_std": fmt(row.std_delta_avg_token_margin),
                "baseline_correct_shift": fmt(row.mean_delta_margin_when_baseline_prefers_correct),
                "baseline_wrong_shift": fmt(row.mean_delta_margin_when_baseline_prefers_false),
                "baseline_correct_minus_wrong": fmt(row.baseline_correct_minus_wrong_delta_margin),
                "baseline_diff_ci": (
                    f"[{fmt(row.baseline_correct_minus_wrong_delta_margin_ci_low)}, "
                    f"{fmt(row.baseline_correct_minus_wrong_delta_margin_ci_high)}]"
                ),
                "sign_flips": int(row.sign_flip_total),
                "baseline_delta_corr": fmt(row.corr_baseline_margin_delta_margin),
            }
            for row in selected.itertuples(index=False)
        ]
        lines += ["### 提示词末位置补全得分差干预分解", "", markdown_table(rows), ""]

    completion_paired = read_csv(figures_dir / "completion_margin_steering_position_prompt_final_paired_bootstrap.csv")
    if completion_paired is not None:
        heldout = completion_paired[completion_paired["split"] == "heldout_countries"]
        rows = [
            {
                "metric": row.metric,
                "comparison": row.comparison,
                "estimate": fmt(row.estimate),
                "ci": f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}]",
                "ci_unit": row.ci_unit,
            }
            for row in heldout.itertuples(index=False)
        ]
        lines += ["### 提示词末位置补全干预配对自助法", "", markdown_table(rows), ""]

    position_summary_paths = [
        figures_dir / "completion_margin_steering_summary.csv",
        figures_dir / "completion_margin_steering_position_prompt_final_summary.csv",
        figures_dir / "completion_margin_steering_position_completion_internal_summary.csv",
    ]
    if all(path.exists() for path in position_summary_paths):
        position_summary = pd.concat([pd.read_csv(path) for path in position_summary_paths], ignore_index=True)
        selected = position_summary[
            (position_summary["split"] == "heldout_countries")
            & (position_summary["direction_type"] == "learned_probe")
            & (position_summary["alpha"].isin([-4.0, 0.0, 4.0]))
        ]
        rows = [
            {
                "position_mode": row.position_mode,
                "alpha": fmt(row.alpha, digits=1),
                "mean_delta_avg_token_margin": fmt(row.mean_delta_avg_token_margin),
                "pairwise_avg_accuracy": fmt(row.pairwise_avg_token_accuracy),
                "block_exact_accuracy": fmt(row.block_exact_avg_token_accuracy),
            }
            for row in selected.itertuples(index=False)
        ]
        lines += ["### 补全干预位置分解", "", markdown_table(rows), ""]

    position_paired_paths = [
        figures_dir / "completion_margin_steering_position_prompt_final_paired_bootstrap.csv",
        figures_dir / "completion_margin_steering_position_completion_internal_paired_bootstrap.csv",
    ]
    if all(path.exists() for path in position_paired_paths):
        paired_frames = []
        for path in position_paired_paths:
            frame = pd.read_csv(path)
            mode = "prompt-final-only" if "prompt_final" in path.name else "completion-internal-only"
            frame["position_mode"] = mode
            paired_frames.append(frame)
        position_paired = pd.concat(paired_frames, ignore_index=True)
        heldout = position_paired[position_paired["split"] == "heldout_countries"]
        rows = [
            {
                "position_mode": row.position_mode,
                "metric": row.metric,
                "comparison": row.comparison,
                "estimate": fmt(row.estimate),
                "ci": f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}]",
            }
            for row in heldout.itertuples(index=False)
        ]
        lines += ["### 位置分解配对自助法", "", markdown_table(rows), ""]

    null_summary = read_csv(figures_dir / "completion_margin_steering_null_summary.csv")
    if null_summary is not None:
        rows = [
            {
                "control_type": row.control_type,
                "directions": int(row.directions),
                "mean_delta": fmt(row.mean_delta),
                "null_95_interval": f"[{fmt(row.q025)}, {fmt(row.q975)}]",
                "learned_effect": fmt(row.learned_effect),
                "learned_percentile": fmt(row.learned_percentile),
                "empirical_p_ge_learned": fmt(row.empirical_p_ge_learned)
                if pd.notna(row.empirical_p_ge_learned)
                else "",
            }
            for row in null_summary.itertuples(index=False)
        ]
        lines += ["### 补全干预零分布", "", markdown_table(rows), ""]

    repeated = read_csv(figures_dir / "repeated_split_completion_steering_summary.csv")
    if repeated is not None:
        aggregate = repeated[repeated["seed"].astype(str) == "aggregate"]
        split_rows = repeated[repeated["seed"].astype(str) != "aggregate"]
        rows = []
        if not aggregate.empty:
            row = aggregate.iloc[0]
            rows.append(
                {
                    "scope": "aggregate",
                    "splits": len(split_rows),
                    "learned_delta": fmt(row.learned_delta),
                    "learned_delta_std": fmt(row.learned_delta_std),
                    "learned_delta_range": f"[{fmt(row.learned_delta_min)}, {fmt(row.learned_delta_max)}]",
                    "learned_minus_random_mean": fmt(row.learned_minus_random_mean),
                    "learned_minus_permutation_mean": fmt(row.learned_minus_permutation_mean),
                    "learned_gt_all_random_splits": int(row.learned_gt_all_random),
                    "learned_gt_all_permutation_splits": int(row.learned_gt_all_permutation),
                    "baseline_pairwise_accuracy": fmt(row.baseline_pairwise_accuracy),
                    "mean_pairwise_accuracy": fmt(row.learned_pairwise_accuracy),
                    "pairwise_accuracy_change": fmt(row.pairwise_accuracy_change),
                    "total_sign_flips": int(row.learned_sign_flips),
                    "wrong_to_correct_flips": int(row.wrong_to_correct_flips),
                    "correct_to_wrong_flips": int(row.correct_to_wrong_flips),
                }
            )
        lines += ["### 重复划分补全干预", "", markdown_table(rows), ""]

    ambiguous = read_csv(figures_dir / "ambiguous_fact_sensitivity_summary.csv")
    if ambiguous is not None:
        rows = []
        for row in ambiguous.itertuples(index=False):
            if row.analysis == "dataset":
                rows.append(
                    {
                        "analysis": row.analysis,
                        "blocks": int(row.blocks),
                        "heldout_blocks": "",
                        "auc": "",
                        "delta": "",
                        "pairwise_accuracy": "",
                        "sign_flips": "",
                    }
                )
            elif row.analysis == "prompt_final_steering":
                rows.append(
                    {
                        "analysis": f"{row.analysis}:{row.direction}",
                        "blocks": "",
                        "heldout_blocks": int(row.heldout_blocks),
                        "auc": "",
                        "delta": fmt(row.mean_delta_avg_token_margin),
                        "pairwise_accuracy": fmt(row.pairwise_avg_accuracy),
                        "sign_flips": int(row.sign_flips),
                    }
                )
            else:
                rows.append(
                    {
                        "analysis": row.analysis,
                        "blocks": "",
                        "heldout_blocks": int(row.heldout_blocks),
                        "auc": fmt(row.auc),
                        "delta": "",
                        "pairwise_accuracy": "",
                        "sign_flips": "",
                    }
                )
        lines += ["### 争议事实敏感性", "", markdown_table(rows), ""]

    rank = read_csv(figures_dir / "candidate_rank_steering_summary.csv")
    if rank is not None:
        rows = [
            {
                "heldout_countries": int(row.heldout_countries),
                "candidate_count": int(row.candidate_count),
                "mean_rank_delta": fmt(row.mean_rank_delta),
                "rank_improved_count": int(row.rank_improved_count),
                "rank_worsened_count": int(row.rank_worsened_count),
                "baseline_top1_accuracy": fmt(row.baseline_top1_accuracy),
                "steered_top1_accuracy": fmt(row.steered_top1_accuracy),
                "top1_changed_count": int(row.top1_changed_count),
                "selected_pair_margin_delta": fmt(row.mean_selected_pair_margin_delta),
            }
            for row in rank.itertuples(index=False)
        ]
        lines += ["### 候选集排名干预", "", markdown_table(rows), ""]

    projection_summary = read_csv(figures_dir / "unembedding_projection_baseline_summary.csv")
    if projection_summary is not None:
        selected = projection_summary[
            (projection_summary["split"] == "test")
            & (projection_summary["alpha"] == 4.0)
        ]
        rows = [
            {
                "direction": row.direction_type,
                "observed_mean": fmt(row.mean_observed_delta_avg_token_margin),
                "predicted_mean": fmt(row.mean_predicted_delta_avg_token_margin),
                "observed_abs_mean": fmt(row.mean_abs_observed_delta_avg_token_margin),
                "predicted_abs_mean": fmt(row.mean_abs_predicted_delta_avg_token_margin),
                "mean_abs_residual": fmt(row.mean_abs_residual),
                "corr": fmt(row.corr_predicted_observed),
                "corr_squared": fmt(row.corr_squared_predicted_observed),
            }
            for row in selected.itertuples(index=False)
        ]
        lines += ["### 全位置 unembedding 投影基线", "", markdown_table(rows), ""]

    ablation = read_csv(figures_dir / "ablation_capital_probe_layer8.csv")
    if ablation is not None:
        rows = [
            {
                "strength": fmt(row.strength, digits=2),
                "fixed_direction_score_gap": fmt(row.fixed_direction_score_gap),
                "fixed_direction_accuracy": fmt(row.fixed_direction_accuracy),
                "retrained_probe_auc": fmt(row.auc),
            }
            for row in ablation.itertuples(index=False)
        ]
        lines += ["### 探针方向消融", "", markdown_table(rows), ""]

    balanced_ablation = read_csv(figures_dir / "ablation_capital_balanced_layer6.csv")
    if balanced_ablation is not None:
        rows = [
            {
                "strength": fmt(row.strength, digits=2),
                "fixed_direction_score_gap": fmt(row.fixed_direction_score_gap),
                "fixed_direction_accuracy": fmt(row.fixed_direction_accuracy),
                "retrained_probe_auc": fmt(row.auc),
            }
            for row in balanced_ablation.itertuples(index=False)
        ]
        lines += ["### 词汇平衡探针方向消融", "", markdown_table(rows), ""]

    iterative_ablation = read_csv(figures_dir / "iterative_ablation_capital_layer8.csv")
    if iterative_ablation is not None:
        key_steps = iterative_ablation[
            iterative_ablation["directions_removed"].isin([0, 1, 2, 4, 8, 12, 16])
        ]
        rows = [
            {
                "control": row.control,
                "directions_removed": int(row.directions_removed),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in key_steps.itertuples(index=False)
        ]
        lines += ["### 迭代方向消融", "", markdown_table(rows), ""]

    balanced_iterative = read_csv(figures_dir / "iterative_ablation_capital_balanced_layer6.csv")
    if balanced_iterative is not None:
        key_steps = balanced_iterative[
            balanced_iterative["directions_removed"].isin([0, 1, 2, 4, 8, 12, 16])
        ]
        rows = [
            {
                "control": row.control,
                "directions_removed": int(row.directions_removed),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "direction_agnostic_auc": fmt(row.separability_auc),
            }
            for row in key_steps.itertuples(index=False)
        ]
        lines += ["### 词汇平衡迭代方向消融", "", markdown_table(rows), ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved results summary to {out_path}")


if __name__ == "__main__":
    main()
