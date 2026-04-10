from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from utils import ensure_directory, write_json

CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
REPLACEMENT_MAP = {"U": "C", "B": "D", "Z": "E", "X": "A"}
VALID_POLICIES = {"strict", "replace"}


def normalize_protein_sequence(seq: Any) -> str | None:
    if seq is None or pd.isna(seq):
        return None
    normalized = re.sub(r"[^A-Za-z]", "", str(seq).upper())
    return normalized or None


def is_valid_protein_sequence(seq: str | None, allowed_chars: set[str] | None = None) -> bool:
    if not seq:
        return False
    allowed = allowed_chars or CANONICAL_AMINO_ACIDS
    return set(seq).issubset(allowed)


def clean_protein_sequence(seq: Any, policy: str = "replace") -> dict[str, Any]:
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unsupported sequence cleaning policy: {policy}")

    if seq is None or pd.isna(seq):
        return {
            "cleaned_sequence": None,
            "status": "dropped",
            "reason": "null_sequence",
            "invalid_chars": [],
            "removed_non_letters": [],
            "was_normalized": False,
            "was_repaired": False,
        }

    raw_sequence = str(seq)
    uppercase_sequence = raw_sequence.upper()
    removed_non_letters = re.findall(r"[^A-Z]", uppercase_sequence)
    normalized_sequence = normalize_protein_sequence(raw_sequence)
    if not normalized_sequence:
        return {
            "cleaned_sequence": None,
            "status": "dropped",
            "reason": "empty_after_normalization",
            "invalid_chars": [],
            "removed_non_letters": removed_non_letters,
            "was_normalized": True,
            "was_repaired": False,
        }

    invalid_chars = sorted({char for char in normalized_sequence if char not in CANONICAL_AMINO_ACIDS})
    was_normalized = normalized_sequence != raw_sequence
    if not invalid_chars:
        return {
            "cleaned_sequence": normalized_sequence,
            "status": "normalized_only" if was_normalized else "valid_as_is",
            "reason": None,
            "invalid_chars": [],
            "removed_non_letters": removed_non_letters,
            "was_normalized": was_normalized,
            "was_repaired": False,
        }

    if policy == "strict":
        return {
            "cleaned_sequence": None,
            "status": "dropped",
            "reason": f"invalid_residues:{''.join(invalid_chars)}",
            "invalid_chars": invalid_chars,
            "removed_non_letters": removed_non_letters,
            "was_normalized": was_normalized,
            "was_repaired": False,
        }

    repaired_sequence = "".join(REPLACEMENT_MAP.get(char, char) for char in normalized_sequence)
    invalid_after_repair = sorted({char for char in repaired_sequence if char not in CANONICAL_AMINO_ACIDS})
    if invalid_after_repair:
        return {
            "cleaned_sequence": None,
            "status": "dropped",
            "reason": f"unrepairable_residues:{''.join(invalid_after_repair)}",
            "invalid_chars": invalid_chars,
            "removed_non_letters": removed_non_letters,
            "was_normalized": was_normalized,
            "was_repaired": True,
        }

    return {
        "cleaned_sequence": repaired_sequence,
        "status": "repaired",
        "reason": None,
        "invalid_chars": invalid_chars,
        "removed_non_letters": removed_non_letters,
        "was_normalized": was_normalized,
        "was_repaired": True,
    }


def preprocess_sequences(
    df: pd.DataFrame,
    seq_columns: list[str],
    policy: str = "replace",
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for column in seq_columns:
        if column not in df.columns:
            raise KeyError(f"Sequence column {column} not found in DataFrame.")

    records: list[dict[str, Any]] = []
    invalid_character_counter: Counter[str] = Counter()

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        for seq_column in seq_columns:
            result = clean_protein_sequence(row_dict.get(seq_column), policy=policy)
            invalid_character_counter.update(result["invalid_chars"])
            invalid_character_counter.update(result["removed_non_letters"])
            record = {
                "source_file": row_dict.get("source_file"),
                "ppi_source_label": row_dict.get("ppi_source_label"),
                "source_row_index": row_dict.get("source_row_index"),
                "source_column": seq_column,
                "raw_sequence": row_dict.get(seq_column),
                "cleaned_sequence": result["cleaned_sequence"],
                "status": result["status"],
                "reason": result["reason"],
                "invalid_chars": "".join(result["invalid_chars"]),
                "removed_non_letters": "".join(result["removed_non_letters"]),
                "was_normalized": result["was_normalized"],
                "was_repaired": result["was_repaired"],
                "is_retained": result["cleaned_sequence"] is not None,
            }
            records.append(record)

    log_df = pd.DataFrame(records)
    retained_df = log_df[log_df["is_retained"]].copy()
    retained_unique_df = retained_df.drop_duplicates(subset=["cleaned_sequence"]).copy()
    retained_unique_df = retained_unique_df.rename(columns={"raw_sequence": "example_raw_sequence"})
    retained_unique_df = retained_unique_df[
        [
            "cleaned_sequence",
            "example_raw_sequence",
            "source_file",
            "ppi_source_label",
            "source_column",
        ]
    ]

    occurrence_counts = (
        retained_df.groupby("cleaned_sequence")
        .size()
        .reset_index(name="occurrence_count")
    )
    retained_unique_df = retained_unique_df.merge(occurrence_counts, on="cleaned_sequence", how="left")
    retained_unique_df.insert(0, "protein_id", range(1, len(retained_unique_df) + 1))

    failed_examples = (
        log_df.loc[~log_df["is_retained"], ["raw_sequence", "reason"]]
        .dropna(subset=["raw_sequence"])
        .drop_duplicates()
        .head(10)
        .to_dict(orient="records")
    )

    summary = {
        "policy": policy,
        "replacement_map": REPLACEMENT_MAP if policy == "replace" else {},
        "total_sequences_processed": int(len(log_df)),
        "valid_as_is": int((log_df["status"] == "valid_as_is").sum()),
        "normalized_only": int((log_df["status"] == "normalized_only").sum()),
        "repaired": int((log_df["status"] == "repaired").sum()),
        "dropped": int((log_df["status"] == "dropped").sum()),
        "retained_total_count": int(retained_df["cleaned_sequence"].notna().sum()),
        "retained_unique_count": int(len(retained_unique_df)),
        "duplicate_raw_sequences": int(log_df["raw_sequence"].dropna().duplicated().sum()),
        "duplicate_cleaned_sequences": int(retained_df["cleaned_sequence"].dropna().duplicated().sum()),
        "invalid_character_frequency": dict(sorted(invalid_character_counter.items())),
        "failed_examples": failed_examples,
    }

    if output_dir is not None:
        output_root = ensure_directory(output_dir)
        log_df.to_csv(output_root / "sequence_processing_log.csv", index=False)
        retained_unique_df.to_csv(output_root / "cleaned_sequences.csv", index=False)
        write_json(output_root / "sequence_preprocessing_summary.json", summary)

    return retained_unique_df, log_df, summary
