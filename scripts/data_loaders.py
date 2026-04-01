"""Load tabular and transcript-style datasets for persona experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _aggregate_stats(df: pd.DataFrame) -> str:
    stats: dict[str, Any] = {}
    for col in df.select_dtypes(include="number").columns:
        stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min": df[col].min(),
            "max": df[col].max(),
        }
    for col in df.select_dtypes(include="object").columns:
        s = df[col].astype(str)
        lens = s.str.len()
        if lens.median() > 120:
            stats[col] = {
                "avg_length": round(float(lens.mean()), 1),
                "min_length": int(lens.min()),
                "max_length": int(lens.max()),
            }
        else:
            stats[col] = s.value_counts().head(5).to_dict()
    return json.dumps(stats, indent=2, default=str)


def load_csv_dataset(path: Path, sample_max_rows: int) -> dict[str, Any]:
    # Survey exports may use legacy encodings; avoid hard failures on odd bytes.
    try:
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    except TypeError:
        df = pd.read_csv(path, encoding="latin-1")
    sample = df.head(sample_max_rows)
    return {
        "kind": "csv",
        "total_rows": len(df),
        "data_sample": sample.to_string(index=False),
        "data_stats": _aggregate_stats(df),
    }


def load_transcript_dataset(
    paths: list[Path],
    id_column: str,
    text_column: str,
    max_chars_per_transcript: int,
    max_rows_per_file: int,
) -> dict[str, Any]:
    parts: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        if text_column not in df.columns:
            raise ValueError(f"Missing column {text_column!r} in {p}")
        cohort = p.stem
        df = df.head(max_rows_per_file).copy()
        df["_cohort"] = cohort
        df[text_column] = (
            df[text_column].astype(str).str.slice(0, max_chars_per_transcript)
        )
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    display_cols = [c for c in (id_column, text_column, "_cohort") if c in full.columns]
    if not display_cols:
        display_cols = list(full.columns)
    sample = full[display_cols].rename(columns={"_cohort": "cohort"})
    return {
        "kind": "transcript_csv",
        "total_rows": len(full),
        "data_sample": sample.to_string(index=False),
        "data_stats": _aggregate_stats(full),
    }
