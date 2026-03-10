"""Normalize account balance CSV so the first value becomes 1."""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "mars.csv"
DEFAULT_OUTPUT = BASE_DIR / "mars_normalized.csv"


def normalize_csv(input_path: Path, output_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        raise ValueError("No numeric columns found to normalize.")

    for col in numeric_cols:
        first_value = float(df[col].iloc[0])
        if first_value == 0:
            raise ValueError(f"First value of column '{col}' is 0, cannot scale to 1.")
        df[col] = df[col].astype(float) / first_value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale each numeric column proportionally so its first value becomes 1."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input CSV path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_path = normalize_csv(args.input, args.output)
    print(f"[OK] Normalized CSV saved to: {saved_path}")


if __name__ == "__main__":
    main()
