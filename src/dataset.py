from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FactExample:
    language: str
    domain: str
    statement: str
    label: int
    pair_id: str


FACT_PAIRS: list[tuple[str, str, str, str, str]] = [
    ("en", "capital", "The capital of France is Paris.", "The capital of France is Rome.", "capital_france"),
    ("en", "capital", "The capital of Japan is Tokyo.", "The capital of Japan is Kyoto.", "capital_japan"),
    ("en", "capital", "The capital of Italy is Rome.", "The capital of Italy is Milan.", "capital_italy"),
    ("en", "capital", "The capital of Canada is Ottawa.", "The capital of Canada is Toronto.", "capital_canada"),
    ("en", "geography", "The Nile is a river in Africa.", "The Nile is a mountain in Europe.", "nile"),
    ("en", "science", "Water freezes at 0 degrees Celsius.", "Water freezes at 100 degrees Celsius.", "water_freezing"),
    ("en", "science", "The Earth orbits the Sun.", "The Sun orbits the Earth.", "earth_sun"),
    ("en", "science", "Humans need oxygen to survive.", "Humans need helium to survive.", "oxygen"),
    ("en", "math", "Two plus two equals four.", "Two plus two equals five.", "two_plus_two"),
    ("en", "history", "The Great Wall is in China.", "The Great Wall is in Brazil.", "great_wall"),
    ("zh", "capital", "法国的首都是巴黎。", "法国的首都是罗马。", "capital_france_zh"),
    ("zh", "capital", "日本的首都是东京。", "日本的首都是京都。", "capital_japan_zh"),
    ("zh", "capital", "意大利的首都是罗马。", "意大利的首都是米兰。", "capital_italy_zh"),
    ("zh", "capital", "加拿大的首都是渥太华。", "加拿大的首都是多伦多。", "capital_canada_zh"),
    ("zh", "geography", "尼罗河位于非洲。", "尼罗河位于欧洲的山脉中。", "nile_zh"),
    ("zh", "science", "水在 0 摄氏度会结冰。", "水在 100 摄氏度会结冰。", "water_freezing_zh"),
    ("zh", "science", "地球围绕太阳运行。", "太阳围绕地球运行。", "earth_sun_zh"),
    ("zh", "science", "人类需要氧气生存。", "人类需要氦气生存。", "oxygen_zh"),
    ("zh", "math", "二加二等于四。", "二加二等于五。", "two_plus_two_zh"),
    ("zh", "history", "长城位于中国。", "长城位于巴西。", "great_wall_zh"),
]


def build_fact_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for language, domain, true_statement, false_statement, pair_id in FACT_PAIRS:
        rows.append(
            {
                "language": language,
                "domain": domain,
                "pair_id": pair_id,
                "statement": true_statement,
                "label": 1,
            }
        )
        rows.append(
            {
                "language": language,
                "domain": domain,
                "pair_id": pair_id,
                "statement": false_statement,
                "label": 0,
            }
        )
    return rows


def save_default_dataset(path: str | Path = "data/facts.csv") -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_fact_rows()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "domain", "pair_id", "statement", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def load_dataset(path: str | Path):
    import pandas as pd

    data = pd.read_csv(path)
    expected = {"language", "domain", "pair_id", "statement", "label"}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return data
