from __future__ import annotations

from src.dataset import save_default_dataset


def main() -> None:
    out_path = save_default_dataset()
    print(f"Saved starter dataset to {out_path}")


if __name__ == "__main__":
    main()
