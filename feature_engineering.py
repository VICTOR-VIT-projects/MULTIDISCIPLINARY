from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sequence_cleaner import CANONICAL_AMINO_ACIDS
from utils import ensure_directory, write_json


def _charge_category_at_ph(pI: float, ph: float = 7.4) -> str:
    if pI > ph:
        return "Positive"
    if pI < ph:
        return "Negative"
    return "Neutral"


def get_protein_features(sequence: str) -> dict[str, Any]:
    if not sequence:
        raise ValueError("Protein sequence is empty.")

    analyzed = ProteinAnalysis(sequence)
    aa_counts = analyzed.count_amino_acids()
    sequence_length = len(sequence)
    pI = float(analyzed.isoelectric_point())
    helix_fraction, turn_fraction, sheet_fraction = analyzed.secondary_structure_fraction()

    try:
        flexibility_values = analyzed.flexibility()
        flexibility_mean = float(np.mean(flexibility_values)) if flexibility_values else np.nan
    except Exception:
        flexibility_mean = np.nan

    features: dict[str, Any] = {
        "sequence_length": int(sequence_length),
        "pI": pI,
        "GRAVY": float(analyzed.gravy()),
        "molecular_weight": float(analyzed.molecular_weight()),
        "charge_category_pH_7_4": _charge_category_at_ph(pI, ph=7.4),
        "aromaticity": float(analyzed.aromaticity()),
        "instability_index": float(analyzed.instability_index()),
        "flexibility_mean": flexibility_mean,
        "helix_fraction": float(helix_fraction),
        "turn_fraction": float(turn_fraction),
        "sheet_fraction": float(sheet_fraction),
    }
    for amino_acid in sorted(CANONICAL_AMINO_ACIDS):
        features[f"aa_frac_{amino_acid}"] = float(aa_counts.get(amino_acid, 0) / sequence_length)
    return features


def safe_feature_extraction(sequence: str) -> dict[str, Any]:
    try:
        features = get_protein_features(sequence)
        return {
            "cleaned_sequence": sequence,
            "extraction_status": "success",
            "failure_reason": None,
            **features,
        }
    except Exception as exc:
        return {
            "cleaned_sequence": sequence,
            "extraction_status": "failed",
            "failure_reason": str(exc),
        }


def extract_protein_feature_table(
    sequences: Iterable[str],
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    unique_sequences = list(dict.fromkeys(sequence for sequence in sequences if sequence))
    extraction_records = [safe_feature_extraction(sequence) for sequence in unique_sequences]
    feature_log_df = pd.DataFrame(extraction_records)

    success_df = feature_log_df[feature_log_df["extraction_status"] == "success"].copy()
    success_df.insert(0, "protein_id", range(1, len(success_df) + 1))

    failure_df = feature_log_df[feature_log_df["extraction_status"] != "success"].copy()
    summary = {
        "attempted_sequences": int(len(unique_sequences)),
        "successful_extractions": int(len(success_df)),
        "failed_extractions": int(len(failure_df)),
    }

    if output_dir is not None:
        output_root = ensure_directory(output_dir)
        success_df.to_csv(output_root / "protein_feature_table.csv", index=False)
        failure_df.to_csv(output_root / "protein_feature_failures.csv", index=False)
        write_json(output_root / "feature_extraction_summary.json", summary)

    return success_df, feature_log_df, summary
