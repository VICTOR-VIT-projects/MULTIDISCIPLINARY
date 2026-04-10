from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils import ENVIRONMENTAL_CONSTANTS, SYNTHETIC_LABEL_DISCLAIMER, ensure_directory, write_json

PROTEIN_NON_MODEL_COLUMNS = {
    "cleaned_sequence",
    "extraction_status",
    "failure_reason",
    "example_raw_sequence",
    "source_file",
    "ppi_source_label",
    "source_column",
    "occurrence_count",
}


def generate_interaction_label(data: pd.DataFrame | pd.Series) -> pd.Series | int:
    if isinstance(data, pd.DataFrame):
        return ((data["surfcharge"] > 0) & (data["pI"] < 6)).astype("int8")
    if isinstance(data, pd.Series):
        return int(data["surfcharge"] > 0 and data["pI"] < 6)
    raise TypeError("generate_interaction_label expects a pandas DataFrame or Series.")


def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    optimized = df.copy()
    for column in optimized.select_dtypes(include=["float64"]).columns:
        optimized[column] = pd.to_numeric(optimized[column], downcast="float")
    for column in optimized.select_dtypes(include=["int64"]).columns:
        optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
    for column in optimized.select_dtypes(include=["object", "string"]).columns:
        unique_ratio = optimized[column].nunique(dropna=False) / max(len(optimized), 1)
        if unique_ratio < 0.5:
            optimized[column] = optimized[column].astype("category")
    return optimized


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    class_counts = df["Interaction"].value_counts(dropna=False).sort_index()
    positive_count = int(class_counts.get(1, 0))
    negative_count = int(class_counts.get(0, 0))
    total_rows = int(len(df))
    return {
        "rows": total_rows,
        "columns": int(df.shape[1]),
        "unique_nanoparticle_rows": int(df[["NPs", "coresize", "hydrosize", "surfcharge", "surfarea", "Ec", "Expotime", "dosage", "e", "NOxygen", "class"]].drop_duplicates().shape[0]),
        "unique_nanoparticle_names": int(df["NPs"].nunique()),
        "unique_proteins": int(df["protein_id"].nunique()),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "positive_percentage": float(positive_count / total_rows) if total_rows else 0.0,
        "negative_percentage": float(negative_count / total_rows) if total_rows else 0.0,
        "null_counts": {column: int(value) for column, value in df.isnull().sum().items()},
        "label_disclaimer": SYNTHETIC_LABEL_DISCLAIMER,
    }


def build_interaction_dataset(
    toxicity_df: pd.DataFrame,
    protein_features_df: pd.DataFrame,
    environmental_constants: dict[str, float] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    successful_features = protein_features_df[protein_features_df["extraction_status"] == "success"].copy()
    if successful_features.empty:
        raise ValueError("No successful protein feature rows are available for dataset assembly.")

    protein_model_columns = [
        column for column in successful_features.columns if column not in PROTEIN_NON_MODEL_COLUMNS
    ]
    protein_feature_table = successful_features[protein_model_columns].copy()
    interaction_df = toxicity_df.merge(protein_feature_table, how="cross")

    constants = environmental_constants or ENVIRONMENTAL_CONSTANTS
    for column, value in constants.items():
        interaction_df[column] = value

    interaction_df["Interaction"] = generate_interaction_label(interaction_df)
    interaction_df = _downcast_dataframe(interaction_df)

    summary = summarize_dataset(interaction_df)
    summary.update(
        {
            "toxicity_rows": int(len(toxicity_df)),
            "protein_feature_rows": int(len(protein_feature_table)),
            "expected_cartesian_size": int(len(toxicity_df) * len(protein_feature_table)),
            "actual_merged_size": int(len(interaction_df)),
        }
    )

    if output_dir is not None:
        output_root = ensure_directory(output_dir)
        write_json(output_root / "interaction_dataset_summary.json", summary)

    return interaction_df, summary
