from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from data_loader import ColumnValidationError, detect_sequence_columns, validate_columns
from dataset_builder import build_interaction_dataset, generate_interaction_label
from feature_engineering import safe_feature_extraction
from model_benchmark import benchmark_models
from sequence_cleaner import clean_protein_sequence, preprocess_sequences


def test_validate_columns_reports_missing_columns() -> None:
    df = pd.DataFrame({"NPs": ["A"], "surfcharge": [1.0]})
    with pytest.raises(ColumnValidationError):
        validate_columns(df, ["NPs", "surfcharge", "class"], dataset_name="toxicity_df")


def test_detect_sequence_columns_prefers_exact_pair() -> None:
    df = pd.DataFrame(
        {
            "protein_sequences_1": ["AAAA"],
            "protein_sequences_2": ["KKKK"],
            "notes": ["example"],
        }
    )
    detection = detect_sequence_columns(df)
    assert detection.columns == ["protein_sequences_1", "protein_sequences_2"]


def test_detect_sequence_columns_raises_on_ambiguous_candidates() -> None:
    df = pd.DataFrame(
        {
            "seq_alpha": ["AAAAA"],
            "seq_beta": ["CCCCC"],
            "seq_gamma": ["DDDDD"],
        }
    )
    with pytest.raises(ColumnValidationError):
        detect_sequence_columns(df)


def test_clean_protein_sequence_replace_policy_repairs_supported_residues() -> None:
    result = clean_protein_sequence(" aUbzX\n", policy="replace")
    assert result["cleaned_sequence"] == "ACDEA"
    assert result["status"] == "repaired"


def test_clean_protein_sequence_strict_policy_drops_invalid_residues() -> None:
    result = clean_protein_sequence("AAAU", policy="strict")
    assert result["cleaned_sequence"] is None
    assert result["status"] == "dropped"


def test_preprocess_sequences_counts_duplicates() -> None:
    df = pd.DataFrame(
        {
            "protein_sequences_1": ["AAAA", "AAAA", "CCCC"],
            "protein_sequences_2": ["DDDD", "DDDD", "EEEE"],
            "source_file": ["a.csv", "a.csv", "a.csv"],
            "ppi_source_label": ["positive", "positive", "positive"],
            "source_row_index": [0, 1, 2],
        }
    )
    cleaned_df, _log_df, summary = preprocess_sequences(df, ["protein_sequences_1", "protein_sequences_2"])
    assert cleaned_df["cleaned_sequence"].nunique() == 4
    assert summary["duplicate_cleaned_sequences"] > 0


def test_safe_feature_extraction_captures_failures() -> None:
    result = safe_feature_extraction("")
    assert result["extraction_status"] == "failed"
    assert result["failure_reason"]


def test_build_interaction_dataset_and_label_distribution() -> None:
    toxicity_df = pd.DataFrame(
        {
            "NPs": ["A", "B"],
            "coresize": [1.0, 2.0],
            "hydrosize": [1.1, 2.2],
            "surfcharge": [2.0, -1.0],
            "surfarea": [10.0, 20.0],
            "Ec": [0.1, 0.2],
            "Expotime": [24, 24],
            "dosage": [1.0, 2.0],
            "e": [0.1, 0.2],
            "NOxygen": [1, 2],
            "class": ["nonToxic", "toxic"],
        }
    )
    protein_features_df = pd.DataFrame(
        {
            "protein_id": [1, 2],
            "cleaned_sequence": ["DDDDDDDD", "KKKKKKKK"],
            "extraction_status": ["success", "success"],
            "failure_reason": [None, None],
            "pI": [4.0, 10.0],
            "GRAVY": [0.1, 0.2],
            "molecular_weight": [1.0, 2.0],
            "charge_category_pH_7_4": ["Negative", "Positive"],
            "sequence_length": [8, 8],
        }
    )
    interaction_df, summary = build_interaction_dataset(toxicity_df, protein_features_df)
    assert len(interaction_df) == 4
    assert summary["actual_merged_size"] == 4
    assert generate_interaction_label(interaction_df.iloc[0]) in {0, 1}


def test_benchmark_models_trains_four_models(tmp_path: Path) -> None:
    X_train = pd.DataFrame(
        {
            "NPs": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "class": ["nonToxic", "toxic", "nonToxic", "toxic", "nonToxic", "toxic", "nonToxic", "toxic"],
            "charge_category_pH_7_4": ["Negative", "Positive", "Negative", "Positive", "Negative", "Positive", "Negative", "Positive"],
            "surfcharge": [2, -1, 2, -1, 2, -1, 2, -1],
            "pI": [4, 10, 4, 10, 4, 10, 4, 10],
            "GRAVY": [0.1] * 8,
        }
    )
    y_train = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    results_df, _models = benchmark_models(
        X_train=X_train,
        X_test=X_train.copy(),
        y_train=y_train,
        y_test=y_train.copy(),
        preprocessor={
            "categorical_features": ["NPs", "class", "charge_category_pH_7_4"],
            "numerical_features": ["surfcharge", "pI", "GRAVY"],
        },
        output_dir=tmp_path,
        random_state=42,
    )
    assert results_df["model_name"].nunique() == 4


def test_main_smoke_run_exports_artifacts(tmp_path: Path) -> None:
    toxicity_df = pd.DataFrame(
        {
            "NPs": ["A", "A", "B", "B", "C", "C"],
            "coresize": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            "hydrosize": [10.0, 10.5, 20.0, 20.5, 30.0, 30.5],
            "surfcharge": [2.0, 2.5, -1.0, -1.5, 3.0, -2.0],
            "surfarea": [11.0, 11.5, 21.0, 21.5, 31.0, 31.5],
            "Ec": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            "Expotime": [24, 24, 24, 24, 24, 24],
            "dosage": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "e": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            "NOxygen": [1, 1, 2, 2, 3, 3],
            "class": ["nonToxic", "nonToxic", "toxic", "toxic", "nonToxic", "toxic"],
        }
    )
    ppi_dir = tmp_path / "ppi"
    ppi_dir.mkdir()
    positive_df = pd.DataFrame(
        {
            "protein_sequences_1": ["DDDDDDDDDD", "KKKKKKKKKK", "DDDDDDDDDD", "KKKKKKKKKK"],
            "protein_sequences_2": ["KKKKKKKKKK", "DDDDDDDDDD", "EEEEEEEEEE", "RRRRRRRRRR"],
        }
    )
    negative_df = pd.DataFrame(
        {
            "protein_sequences_1": ["AAAAAAAAXX", "CCCCCCCCCC", "DDDDDDDDDD", "KKKKKKKKKK"],
            "protein_sequences_2": ["GGGGGGGGGG", "VVVVVVVVVV", "KKKKKKKKKK", "DDDDDDDDDD"],
        }
    )
    toxicity_path = tmp_path / "nanotox_dataset.csv"
    positive_path = ppi_dir / "positive_protein_sequences.csv"
    negative_path = ppi_dir / "negative_protein_sequences.csv"
    toxicity_df.to_csv(toxicity_path, index=False)
    positive_df.to_csv(positive_path, index=False)
    negative_df.to_csv(negative_path, index=False)

    output_dir = tmp_path / "artifacts"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            str(repo_root / ".venv" / "bin" / "python"),
            str(repo_root / "main.py"),
            "--toxicity-path",
            str(toxicity_path),
            "--ppi-path",
            str(ppi_dir),
            "--output-dir",
            str(output_dir),
            "--benchmark-subset-size",
            "40",
            "--cv-folds",
            "2",
            "--skip-full-linear-baseline",
        ],
        cwd=repo_root,
        check=True,
    )

    final_summary = json.loads((output_dir / "reports" / "final_summary.json").read_text())
    assert "heuristic synthetic pseudo-labels" in final_summary["disclaimer"]
    assert (output_dir / "benchmarks" / "model_comparison.csv").exists()
    assert (output_dir / "sequence_reports" / "sequence_processing_log.csv").exists()
    assert (output_dir / "feature_reports" / "protein_feature_table.csv").exists()
