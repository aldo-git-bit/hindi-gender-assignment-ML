"""Paths and dataset loading."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "hindi_noun_paradigms.csv"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42


def load_dataset(path=DATA_PATH):
    """Load the Hindi noun paradigm dataset (one row per noun lemma)."""
    return pd.read_csv(path)
