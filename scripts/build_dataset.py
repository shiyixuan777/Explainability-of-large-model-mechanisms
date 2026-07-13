from __future__ import annotations

from src.dataset import load_dataset, save_default_dataset


def main() -> None:
    out_path = save_default_dataset()
    data = load_dataset(out_path)
    print(f"Saved dataset to {out_path}")
    print(f"Rows: {len(data)}")
    print("Label counts:")
    print(data["label"].value_counts().sort_index().to_string())
    print("Domain counts:")
    print(data["domain"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
