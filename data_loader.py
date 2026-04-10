from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from utils import (
    DEFAULT_PPI_DATASET,
    DEFAULT_TOXICITY_DATASET,
    dataframe_preview_records,
    ensure_directory,
    write_json,
)

TOXICITY_REQUIRED_COLUMNS = [
    "NPs",
    "coresize",
    "hydrosize",
    "surfcharge",
    "surfarea",
    "Ec",
    "Expotime",
    "dosage",
    "e",
    "NOxygen",
    "class",
]


class ColumnValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SequenceDetectionResult:
    columns: list[str]
    reason: str
    candidate_scores: dict[str, int]


def _resolve_dataset_source(path: str | Path | None, dataset_name: str, filename: str | None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()

    cache_dataset = DEFAULT_TOXICITY_DATASET if dataset_name == "toxicity" else DEFAULT_PPI_DATASET
    cache_path = _resolve_cached_kagglehub_dataset(cache_dataset)
    if cache_path is not None:
        return cache_path

    import kagglehub

    if dataset_name == "toxicity":
        return Path(kagglehub.dataset_download(DEFAULT_TOXICITY_DATASET))
    if dataset_name == "ppi":
        return Path(kagglehub.dataset_download(DEFAULT_PPI_DATASET))
    raise ValueError(f"Unknown dataset name: {dataset_name}")


def _resolve_cached_kagglehub_dataset(dataset_slug: str) -> Path | None:
    owner, dataset_name = dataset_slug.split("/", maxsplit=1)
    versions_dir = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / owner
        / dataset_name
        / "versions"
    )
    if not versions_dir.exists():
        return None

    version_paths = sorted(
        (path for path in versions_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not version_paths:
        return None
    return version_paths[-1]


def _resolve_csv_path(path: Path, preferred_filename: str | None = None) -> Path:
    if path.is_file():
        return path

    csv_files = sorted(candidate for candidate in path.glob("*.csv") if candidate.is_file())
    if preferred_filename is not None:
        preferred = path / preferred_filename
        if preferred.exists():
            return preferred
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {path}")
    raise FileNotFoundError(
        f"Multiple CSV files found under {path}. "
        f"Provide an explicit file path. Found: {[file.name for file in csv_files]}"
    )


def _near_match_map(columns: list[str], required_cols: list[str]) -> dict[str, list[str]]:
    lowered_to_actual = {column.lower(): column for column in columns}
    lowered_columns = list(lowered_to_actual)
    suggestions: dict[str, list[str]] = {}
    for required in required_cols:
        matches = difflib.get_close_matches(required.lower(), lowered_columns, n=3, cutoff=0.5)
        suggestions[required] = [lowered_to_actual[match] for match in matches]
    return suggestions


def validate_columns(df: pd.DataFrame, required_cols: list[str], dataset_name: str) -> None:
    missing_columns = [column for column in required_cols if column not in df.columns]
    if not missing_columns:
        return

    suggestions = _near_match_map(df.columns.tolist(), missing_columns)
    suggestion_text = ", ".join(
        f"{column}: {matches}" for column, matches in suggestions.items() if matches
    )
    error_message = (
        f"{dataset_name} is missing required columns: {missing_columns}. "
        f"Available columns: {df.columns.tolist()}."
    )
    if suggestion_text:
        error_message += f" Near matches: {suggestion_text}."
    raise ColumnValidationError(error_message)


def report_dataframe_health(
    df: pd.DataFrame,
    dataset_name: str,
    required_cols: list[str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    columns = df.columns.tolist()
    missing_required = [column for column in (required_cols or []) if column not in columns]
    report = {
        "dataset_name": dataset_name,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": columns,
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {column: int(value) for column, value in df.isnull().sum().items()},
        "duplicate_row_count": int(df.duplicated().sum()),
        "preview": dataframe_preview_records(df),
        "missing_required_columns": missing_required,
        "suspicious_columns": _near_match_map(columns, required_cols or []),
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def detect_sequence_columns(df: pd.DataFrame) -> SequenceDetectionResult:
    exact_pair = ["protein_sequences_1", "protein_sequences_2"]
    if all(column in df.columns for column in exact_pair):
        return SequenceDetectionResult(
            columns=exact_pair,
            reason="Selected exact protein pair columns protein_sequences_1 and protein_sequences_2.",
            candidate_scores={column: 100 for column in exact_pair},
        )

    candidate_scores: dict[str, int] = {}
    string_like_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]
    pattern = re.compile(r"(sequence|seq|protein)", re.IGNORECASE)
    exact_preferences = {
        "sequence": 95,
        "protein_sequence": 95,
        "seq": 90,
        "aa_sequence": 90,
        "protein_seq": 90,
    }

    for column in string_like_columns:
        score = 0
        column_lower = column.lower()
        if column_lower in exact_preferences:
            score = exact_preferences[column_lower]
        if pattern.search(column):
            score = max(score, 75)
        non_null_ratio = float(df[column].notna().mean()) if len(df) else 0.0
        if non_null_ratio > 0.5:
            score += 10
        sample = df[column].dropna().astype(str).head(20)
        if not sample.empty and sample.str.len().mean() > 15:
            score += 5
        if score > 0:
            candidate_scores[column] = score

    if not candidate_scores:
        raise ColumnValidationError(
            f"No plausible protein sequence columns found. Available columns: {df.columns.tolist()}"
        )

    ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    top_score = ranked[0][1]
    top_group = [column for column, score in ranked if score == top_score]
    if len(top_group) > 2:
        raise ColumnValidationError(
            f"Ambiguous sequence columns detected with equal score {top_score}: {top_group}"
        )

    selected_columns = [column for column, _score in ranked[: min(2, len(ranked))]]
    reason = (
        "Selected highest scoring sequence columns based on exact-name preference, "
        f"name pattern matching, non-null ratio, and sequence-like length. Scores: {ranked[:5]}"
    )
    return SequenceDetectionResult(
        columns=selected_columns,
        reason=reason,
        candidate_scores=dict(ranked),
    )


def load_toxicity_data(
    path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = _resolve_dataset_source(path, dataset_name="toxicity", filename="nanotox_dataset.csv")
    csv_path = _resolve_csv_path(source, preferred_filename="nanotox_dataset.csv")
    toxicity_df = pd.read_csv(csv_path)
    validate_columns(toxicity_df, TOXICITY_REQUIRED_COLUMNS, dataset_name="toxicity_df")

    report_path = None
    if output_dir is not None:
        report_path = ensure_directory(output_dir) / "toxicity_health_report.json"
    health_report = report_dataframe_health(
        toxicity_df,
        dataset_name="toxicity_df",
        required_cols=TOXICITY_REQUIRED_COLUMNS,
        output_path=report_path,
    )
    metadata = {"source_path": str(csv_path), "health_report": health_report}
    return toxicity_df, metadata


def load_ppi_data(
    path_or_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = _resolve_dataset_source(path_or_dir, dataset_name="ppi", filename=None)
    if source.is_file():
        csv_files = [source]
    else:
        csv_files = sorted(candidate for candidate in source.glob("*.csv") if candidate.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No PPI CSV files found under {source}")

    combined_frames: list[pd.DataFrame] = []
    file_reports: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []
    output_root = ensure_directory(output_dir) if output_dir is not None else None

    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        try:
            detection = detect_sequence_columns(frame)
            validate_columns(frame, detection.columns, dataset_name=csv_file.name)
        except ColumnValidationError as exc:
            skipped_files.append({"file": str(csv_file), "reason": str(exc)})
            if source.is_file():
                raise
            continue

        frame = frame.copy()
        frame["source_file"] = csv_file.name
        frame["ppi_source_label"] = csv_file.stem
        frame["source_row_index"] = range(len(frame))
        combined_frames.append(frame)

        report_path = None
        if output_root is not None:
            file_report_dir = ensure_directory(output_root / "ppi_file_reports")
            report_path = file_report_dir / f"{csv_file.stem}_health_report.json"
        file_report = report_dataframe_health(
            frame,
            dataset_name=csv_file.name,
            required_cols=detection.columns,
            output_path=report_path,
        )
        file_reports.append(
            {
                "file": str(csv_file),
                "selected_sequence_columns": detection.columns,
                "selection_reason": detection.reason,
                "health_report": file_report,
            }
        )

    if not combined_frames:
        raise ColumnValidationError(
            f"No valid PPI CSV files could be loaded from {source}. Skipped files: {skipped_files}"
        )

    combined_df = pd.concat(combined_frames, ignore_index=True, sort=False)
    combined_detection = detect_sequence_columns(combined_df)

    combined_report_path = None
    if output_root is not None:
        combined_report_path = output_root / "ppi_combined_health_report.json"
    combined_health_report = report_dataframe_health(
        combined_df,
        dataset_name="ppi_combined_df",
        required_cols=combined_detection.columns,
        output_path=combined_report_path,
    )

    metadata = {
        "source_path": str(source),
        "files_loaded": [str(path) for path in csv_files],
        "file_reports": file_reports,
        "skipped_files": skipped_files,
        "selected_sequence_columns": combined_detection.columns,
        "selection_reason": combined_detection.reason,
        "candidate_scores": combined_detection.candidate_scores,
        "health_report": combined_health_report,
    }
    if output_root is not None:
        write_json(output_root / "ppi_loader_metadata.json", metadata)
    return combined_df, metadata
