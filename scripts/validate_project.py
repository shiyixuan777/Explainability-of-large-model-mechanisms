from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "README.md",
    "INSTALL.md",
    "requirements.txt",
    "data/facts.csv",
    "reports/final_report.md",
    "reports/project_report.md",
    "reports/results_summary.md",
    "reports/reproducibility_checklist.md",
    "reports/final_deliverable_checklist.md",
    "reports/presentation_outline.md",
    "figures/probe_sweep.csv",
    "figures/probe_sweep_summary.png",
    "figures/probe_capital_answer.csv",
    "figures/probe_capital_answer.png",
    "figures/pca_capital_layer8.csv",
    "figures/pca_capital_layer8.png",
    "figures/error_analysis_capital_layer8.csv",
    "figures/error_analysis_capital_layer8_errors.csv",
    "figures/activation_patching_capital_recall.csv",
    "figures/activation_patching_capital_recall.png",
    "figures/steering_capital_probe_layer8.csv",
    "figures/steering_capital_probe_layer8.png",
    "figures/steering_capital_probe_layer8_accuracy.png",
    "figures/steering_capital_probe_layer8_probe_accuracy.png",
    "figures/ablation_capital_probe_layer8.csv",
    "figures/ablation_capital_probe_layer8.png",
    "figures/ablation_capital_probe_layer8_score_gap.png",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"OK  {message}")


def read_csv(path: str) -> pd.DataFrame:
    full_path = Path(path)
    check(full_path.exists(), f"{path} exists")
    return pd.read_csv(full_path)


def validate_required_files() -> None:
    for file_name in REQUIRED_FILES:
        path = Path(file_name)
        check(path.exists(), f"{file_name} exists")
        check(path.stat().st_size > 0, f"{file_name} is non-empty")


def validate_dataset() -> None:
    data = read_csv("data/facts.csv")
    check(len(data) == 528, "dataset has 528 rows")
    counts = data["label"].value_counts().to_dict()
    check(counts.get(0) == 264 and counts.get(1) == 264, "dataset is label-balanced")
    check(data["domain"].nunique() >= 7, "dataset covers at least 7 domains")


def validate_probe_results() -> None:
    sweep = read_csv("figures/probe_sweep.csv")
    capital_answer = sweep[(sweep["domain"] == "capital") & (sweep["prompt"] == "answer")]
    best_capital_auc = float(capital_answer["auc"].max())
    check(best_capital_auc >= 0.94, "capital+answer probe sweep AUC >= 0.94")

    probe = read_csv("figures/probe_capital_answer.csv")
    layer8 = probe.loc[probe["layer"] == 8].iloc[0]
    layer10 = probe.loc[probe["layer"] == 10].iloc[0]
    check(float(layer8["auc"]) >= 0.94, "focused capital layer 8 AUC >= 0.94")
    check(float(layer10["accuracy"]) >= 0.85, "focused capital layer 10 accuracy >= 0.85")


def validate_visual_and_error_results() -> None:
    pca = read_csv("figures/pca_capital_layer8.csv")
    check(len(pca) == 152, "PCA contains all 152 capital rows")
    check(float(pca["pc1_explained_variance"].iloc[0]) > 0, "PCA has positive PC1 variance")

    errors = read_csv("figures/error_analysis_capital_layer8_errors.csv")
    check(len(errors) == 8, "error analysis has 8 misclassified examples")


def validate_interventions() -> None:
    patching = read_csv("figures/activation_patching_capital_recall.csv")
    resid11 = patching[(patching["component"] == "resid_post") & (patching["layer"] == 11)]
    check(not resid11.empty, "resid_post layer 11 patching row exists")
    check(float(resid11["mean_recovery"].iloc[0]) >= 0.99, "resid_post layer 11 recovery >= 0.99")

    steering = read_csv("figures/steering_capital_probe_layer8.csv")
    check((steering["accuracy_from_logit_sign"] == 0.5).all(), "steering logit-sign accuracy stays at 0.5")
    score_delta = float(steering["mean_probe_score"].iloc[-1] - steering["mean_probe_score"].iloc[0])
    alpha_delta = float(steering["alpha"].iloc[-1] - steering["alpha"].iloc[0])
    check(abs(score_delta - alpha_delta) < 0.1, "steering probe score moves approximately with alpha")

    ablation = read_csv("figures/ablation_capital_probe_layer8.csv")
    baseline_gap = float(ablation.loc[ablation["strength"] == 0, "fixed_direction_score_gap"].iloc[0])
    full_gap = float(ablation.loc[ablation["strength"] == 1.0, "fixed_direction_score_gap"].iloc[0])
    min_retrained_auc = float(ablation["auc"].min())
    check(baseline_gap > 0.5, "ablation baseline fixed-direction score gap > 0.5")
    check(abs(full_gap) < 0.01, "ablation strength=1 removes fixed-direction score gap")
    check(min_retrained_auc >= 0.94, "retrained probe AUC remains >= 0.94 after ablation")


def validate_report_docs() -> None:
    report_text = Path("reports/final_report.md").read_text(encoding="utf-8")
    report_text_lower = report_text.lower()
    required_terms = [
        ("Locate", "Locate"),
        ("Steering", "Steering"),
        ("Ablation", "Ablation"),
        ("Activation Patching", "Activation Patching"),
        ("References section", "参考文献"),
        ("Personal analysis section", "个人分析"),
    ]
    for label, term in required_terms:
        check(term.lower() in report_text_lower, f"final report mentions {label}")


def main() -> None:
    validate_required_files()
    validate_dataset()
    validate_probe_results()
    validate_visual_and_error_results()
    validate_interventions()
    validate_report_docs()
    print("\nProject validation passed.")


if __name__ == "__main__":
    main()
