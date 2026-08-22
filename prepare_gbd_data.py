"""Convert an IHME GBD MND export into data used directly by the dashboard.

Example:
    uv run python prepare_gbd_data.py /path/to/IHME-GBD_2023_DATA.csv
"""

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path("data/mnd_burden.csv")
REQUIRED_COLUMNS = {
    "measure_name", "location_name", "sex_name", "age_name", "cause_name",
    "metric_name", "year", "val",
}


def normalize_mnd_burden(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep MND count estimates and expose measure, sex, age, location, and year."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Not an expected GBD Results export; missing: {', '.join(sorted(missing))}")

    selected = frame[
        (frame["metric_name"].str.casefold() == "number")
        & (frame["cause_name"].str.casefold() == "motor neuron disease")
    ].copy()
    if selected.empty:
        raise ValueError("No MND count estimates were found in this GBD export.")

    return pd.DataFrame({
        "region": selected["location_name"],
        "age_group": selected["age_name"],
        "sex": selected["sex_name"],
        "year": selected["year"],
        "measure": selected["measure_name"],
        "value": selected["val"],
        "source": "IHME GBD 2023 Results export — Motor neuron disease, Number metric",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an IHME GBD MND export for the Streamlit dashboard")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    prepared = normalize_mnd_burden(pd.read_csv(args.input_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(f"Wrote {len(prepared):,} MND burden records to {args.output}")


if __name__ == "__main__":
    main()
