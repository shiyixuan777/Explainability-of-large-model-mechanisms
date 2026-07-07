from __future__ import annotations

import importlib.util
import platform
import sys


REQUIRED = [
    "torch",
    "transformers",
    "transformer_lens",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "tqdm",
    "yaml",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print()

    missing: list[str] = []
    for module in REQUIRED:
        ok = importlib.util.find_spec(module) is not None
        print(f"{module:18s} {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(module)

    if missing:
        print("\nMissing packages detected. Run:")
        print("pip install -r requirements.txt")
    else:
        print("\nEnvironment looks ready.")


if __name__ == "__main__":
    main()
