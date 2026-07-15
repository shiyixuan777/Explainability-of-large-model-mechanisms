from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.dataset import CAPITALS, _slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/capital_balanced.csv")
    parser.add_argument("--max-countries", type=int, default=76)
    return parser.parse_args()


def build_rows(max_countries: int) -> list[dict[str, object]]:
    selected = CAPITALS[:max_countries]
    if len(selected) < 2:
        raise ValueError("Need at least two country-capital pairs.")
    if len(selected) % 2 != 0:
        selected = selected[:-1]

    rows: list[dict[str, object]] = []
    for block_index in range(0, len(selected), 2):
        country_a, capital_a = selected[block_index]
        country_b, capital_b = selected[block_index + 1]
        pair_id = f"balanced_capital_{_slug(country_a)}_{_slug(country_b)}"
        examples = [
            (country_a, capital_a, 1),
            (country_a, capital_b, 0),
            (country_b, capital_b, 1),
            (country_b, capital_a, 0),
        ]
        for country, capital, label in examples:
            rows.append(
                {
                    "language": "en",
                    "domain": "capital_balanced",
                    "pair_id": pair_id,
                    "statement": f"The capital of {country} is {capital}.",
                    "label": label,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.max_countries)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "domain", "pair_id", "statement", "label"])
        writer.writeheader()
        writer.writerows(rows)

    true_count = sum(int(row["label"]) == 1 for row in rows)
    false_count = sum(int(row["label"]) == 0 for row in rows)
    print(f"Saved balanced capital dataset to {out_path}")
    print(f"Rows: {len(rows)}")
    print(f"True rows: {true_count}")
    print(f"False rows: {false_count}")
    print(f"Blocks: {len(set(str(row['pair_id']) for row in rows))}")


if __name__ == "__main__":
    main()
