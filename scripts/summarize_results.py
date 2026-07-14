from __future__ import annotations

import argparse
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


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No data available._\n"
    headers = list(rows[0].keys())
    table = [
        "| " + " | ".join(headers) + " |",
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


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    out_path = Path(args.out)

    lines: list[str] = [
        "# Results Summary",
        "",
        "This file is generated from CSV artifacts by `python -m scripts.summarize_results`.",
        "Use it as a consistency check for the report tables.",
        "",
    ]

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
                "separability_auc": fmt(row.separability_auc),
            }
            for row in best.itertuples(index=False)
        ]
        lines += ["## Probe Sweep: Top Settings", "", markdown_table(rows), ""]

    probe = read_csv(figures_dir / "probe_capital_answer.csv")
    if probe is not None:
        top_auc = probe.sort_values("auc", ascending=False).head(5)
        top_acc = probe.sort_values("accuracy", ascending=False).head(5)
        rows_auc = [
            {
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "separability_auc": fmt(row.separability_auc),
            }
            for row in top_auc.itertuples(index=False)
        ]
        rows_acc = [
            {
                "layer": int(row.layer),
                "accuracy": fmt(row.accuracy),
                "auc": fmt(row.auc),
                "separability_auc": fmt(row.separability_auc),
            }
            for row in top_acc.itertuples(index=False)
        ]
        lines += [
            "## Focused Capital Probe",
            "",
            "Top layers by AUC:",
            "",
            markdown_table(rows_auc),
            "",
            "Top layers by accuracy:",
            "",
            markdown_table(rows_acc),
            "",
        ]

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
        lines += ["## Activation PCA", "", markdown_table(rows), ""]

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
        lines += ["## Error Analysis", "", markdown_table(rows), ""]

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
        lines += ["Misclassified examples:", "", markdown_table(examples), ""]

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
        lines += ["## Activation Patching: Best Layer by Component", "", markdown_table(rows), ""]

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
        lines += ["## Probe-Direction Steering", "", markdown_table(rows), ""]

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
        lines += ["## Probe-Direction Ablation", "", markdown_table(rows), ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved results summary to {out_path}")


if __name__ == "__main__":
    main()
