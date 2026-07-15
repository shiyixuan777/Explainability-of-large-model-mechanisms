from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "data/facts.csv",
    "data/capital_balanced.csv",
    "reports/final_report.md",
    "reports/reproducibility_checklist.md",
    "reports/results_summary.md",
    "figures/surface_baselines.png",
    "figures/probe_capital_balanced.png",
    "figures/capital_knowledge_margin_summary.png",
    "figures/completion_margin_steering_position_prompt_final_summary.png",
    "figures/completion_margin_steering_null_distribution.png",
    "figures/repeated_split_completion_steering_summary.png",
    "figures/completion_margin_steering_position_comparison.png",
]


REQUIRED_CSV_COLUMNS = {
    "figures/probe_capital_balanced.csv": {"layer", "accuracy", "auc", "separability_auc"},
    "figures/completion_margin_steering_position_prompt_final_summary.csv": {
        "direction_type",
        "position_mode",
        "split",
        "alpha",
        "mean_delta_avg_token_margin",
    },
    "figures/completion_margin_steering_null_summary.csv": {
        "control_type",
        "directions",
        "learned_effect",
        "empirical_p_ge_learned",
    },
    "figures/repeated_split_completion_steering_summary.csv": {
        "seed",
        "learned_delta",
        "pairwise_accuracy_change",
        "wrong_to_correct_flips",
    },
}


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        fail("missing required files: " + ", ".join(missing))


def check_report_images() -> None:
    report = ROOT / "reports/final_report.md"
    text = report.read_text(encoding="utf-8")
    image_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    if len(image_links) < 7:
        fail(f"expected at least 7 report images, found {len(image_links)}")
    for link in image_links:
        target = (report.parent / link).resolve()
        if not target.exists():
            fail(f"report image link does not exist: {link}")


def check_balanced_dataset() -> None:
    data = pd.read_csv(ROOT / "data/capital_balanced.csv")
    if len(data) != 152:
        fail(f"capital_balanced.csv should have 152 rows, found {len(data)}")
    if data["pair_id"].nunique() != 38:
        fail(f"capital_balanced.csv should have 38 pair_id blocks, found {data['pair_id'].nunique()}")
    block_sizes = data.groupby("pair_id").size()
    bad_blocks = block_sizes[block_sizes != 4]
    if not bad_blocks.empty:
        fail("balanced pair_id blocks should all have four rows")
    counts = data["label"].value_counts().to_dict()
    if counts.get(0) != 76 or counts.get(1) != 76:
        fail(f"capital_balanced.csv should have 76 rows per label, found {counts}")


def check_csv_columns() -> None:
    for relative_path, required in REQUIRED_CSV_COLUMNS.items():
        path = ROOT / relative_path
        if not path.exists():
            fail(f"missing CSV: {relative_path}")
        columns = set(pd.read_csv(path, nrows=1).columns)
        missing = sorted(required - columns)
        if missing:
            fail(f"{relative_path} missing columns: {', '.join(missing)}")


def main() -> None:
    check_required_files()
    check_report_images()
    check_balanced_dataset()
    check_csv_columns()
    print("Project validation passed.")


if __name__ == "__main__":
    main()
