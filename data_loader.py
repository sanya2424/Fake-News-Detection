"""Dataset loading utilities for the Fake and Real News dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.config import (
    FAKE_CSV_PATH,
    LABEL_TO_ID,
    RAW_DATA_DIR,
    TRUE_CSV_PATH,
    create_project_directories,
)


def _extract_zip_if_present(zip_path: Path) -> None:
    """Extract a CSV ZIP file into the raw data folder when it exists."""
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(RAW_DATA_DIR)


def ensure_dataset_available() -> None:
    """Check that Fake.csv and True.csv exist, extracting local ZIPs if needed.

    The user supplied the dataset ZIPs in Downloads for this build. This helper
    also supports future runs where only data/raw/Fake.csv.zip and
    data/raw/True.csv.zip are present.
    """
    create_project_directories()

    if not FAKE_CSV_PATH.exists():
        _extract_zip_if_present(RAW_DATA_DIR / "Fake.csv.zip")

    if not TRUE_CSV_PATH.exists():
        _extract_zip_if_present(RAW_DATA_DIR / "True.csv.zip")

    missing_files = [
        str(path)
        for path in [FAKE_CSV_PATH, TRUE_CSV_PATH]
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Dataset files are missing. Place Fake.csv and True.csv, or their "
            "ZIP versions, inside data/raw. Missing: " + ", ".join(missing_files)
        )


def load_raw_dataset() -> pd.DataFrame:
    """Load fake and real news CSV files and return one labelled DataFrame."""
    ensure_dataset_available()

    fake_df = pd.read_csv(FAKE_CSV_PATH)
    true_df = pd.read_csv(TRUE_CSV_PATH)

    fake_df["label"] = LABEL_TO_ID["Fake"]
    fake_df["label_name"] = "Fake"
    true_df["label"] = LABEL_TO_ID["Real"]
    true_df["label_name"] = "Real"

    combined_df = pd.concat([fake_df, true_df], ignore_index=True)

    # Keep the public dataset columns plus labels. Extra columns are preserved
    # only when they are useful for EDA or reproducibility.
    expected_columns = ["title", "text", "subject", "date", "label", "label_name"]
    available_columns = [
        column for column in expected_columns if column in combined_df.columns
    ]
    return combined_df[available_columns]
