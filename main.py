from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import TOXICITY_REQUIRED_COLUMNS, detect_sequence_columns, load_ppi_data, load_toxicity_data
from dataset_builder import build_interaction_dataset
from feature_engineering import extract_protein_feature_table
from model_benchmark import benchmark_models, cross_validate_models
from sequence_cleaner import preprocess_sequences
from utils import (
    ENVIRONMENTAL_CONSTANTS,
    SYNTHETIC_LABEL_DISCLAIMER,
    ensure_directory,
    get_environment_versions,
    set_global_seed,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protein-nanoparticle ML pipeline")
    parser.add_argument("--toxicity-path", default=None, help="Path to toxicity CSV or directory.")
    parser.add_argument("--ppi-path", default=None, help="Path to PPI CSV or directory.")
    parser.add_argument("--output-dir", default="artifacts", help="Artifact output directory.")
    parser.add_argument(
        "--sequence-policy",
        choices=["strict", "replace"],
        default="replace",
        help="Protein sequence cleaning policy.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--benchmark-subset-size",
        type=int,
        default=200000,
        help="Maximum rows for full benchmark suite.",
    )
    parser.add_argument("--inspection-sample-size", type=int, default=1000, help="Rows in sample CSV.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds.")
    parser.add_argument(
        "--skip-full-linear-baseline",
        action="store_true",
        help="Skip the optional full-data LinearSVC baseline.",
    )
    return parser.parse_args()


def _select_model_columns(interaction_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    target_column = "Interaction"
    trace_columns = ["protein_id"]
    categorical_features = [
        column
        for column in interaction_df.columns
        if str(interaction_df[column].dtype) in {"category", "object"}
        and column not in {target_column}
    ]
    numerical_features = [
        column
        for column in interaction_df.columns
        if pd.api.types.is_numeric_dtype(interaction_df[column])
        and column not in {target_column, *trace_columns}
    ]
    feature_columns = [*categorical_features, *numerical_features]
    return feature_columns, categorical_features, numerical_features


def _deterministic_subset(df: pd.DataFrame, label_column: str, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    indices = train_test_split(
        df.index,
        train_size=max_rows,
        random_state=seed,
        stratify=df[label_column],
    )[0]
    subset_df = df.loc[sorted(indices)].copy()
    return subset_df


def _save_dataset_artifacts(
    interaction_df: pd.DataFrame,
    successful_features_df: pd.DataFrame,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    dataset_dir = ensure_directory(output_dir / "datasets")
    parquet_path = dataset_dir / "interaction_dataset.parquet"
    interaction_df.to_parquet(parquet_path, index=False)

    sample_df = interaction_df.sample(min(sample_size, len(interaction_df)), random_state=seed)
    sample_df = sample_df.merge(
        successful_features_df[["protein_id", "cleaned_sequence"]],
        on="protein_id",
        how="left",
    )
    sample_path = dataset_dir / "interaction_dataset_sample.csv"
    sample_df.to_csv(sample_path, index=False)
    return {
        "interaction_dataset_parquet": str(parquet_path),
        "interaction_dataset_sample_csv": str(sample_path),
    }


def _run_full_linear_baseline(
    interaction_df: pd.DataFrame,
    categorical_features: list[str],
    numerical_features: list[str],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    from model_benchmark import evaluate_single_model

    X = interaction_df[categorical_features + numerical_features]
    y = interaction_df["Interaction"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    metrics, _pipeline = evaluate_single_model(
        model_name="LinearSVC",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor={
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
        },
        output_dir=output_dir / "full_linear_baseline",
        random_state=seed,
    )
    return metrics


def _available_memory_bytes() -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size * available_pages)


def _full_linear_baseline_feasibility(interaction_df: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    dataset_bytes = int(interaction_df.memory_usage(deep=True).sum())
    available_bytes = _available_memory_bytes()
    estimated_peak_bytes = int(dataset_bytes * 2.5)
    row_count = int(len(interaction_df))
    row_count_limit = 1_000_000
    feasible = row_count <= row_count_limit and estimated_peak_bytes <= int(available_bytes * 0.7)
    diagnostics = {
        "row_count": row_count,
        "row_count_limit": row_count_limit,
        "dataset_bytes": dataset_bytes,
        "available_memory_bytes": available_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "threshold_bytes": int(available_bytes * 0.7),
    }
    return feasible, diagnostics


def main() -> None:
    args = parse_args()
    output_root = ensure_directory(args.output_dir)
    report_dir = ensure_directory(output_root / "reports")
    health_dir = ensure_directory(output_root / "health_reports")
    sequence_dir = ensure_directory(output_root / "sequence_reports")
    feature_dir = ensure_directory(output_root / "feature_reports")
    benchmark_dir = ensure_directory(output_root / "benchmarks")

    set_global_seed(args.random_seed)
    environment_versions = get_environment_versions()
    write_json(report_dir / "environment_versions.json", environment_versions)

    toxicity_df, toxicity_meta = load_toxicity_data(args.toxicity_path, output_dir=health_dir)
    ppi_df, ppi_meta = load_ppi_data(args.ppi_path, output_dir=health_dir)

    sequence_detection = detect_sequence_columns(ppi_df)
    sequence_selection_report = {
        "selected_columns": sequence_detection.columns,
        "reason": sequence_detection.reason,
        "candidate_scores": sequence_detection.candidate_scores,
    }
    write_json(report_dir / "sequence_column_selection.json", sequence_selection_report)

    cleaned_sequences_df, sequence_log_df, sequence_summary = preprocess_sequences(
        ppi_df,
        seq_columns=sequence_detection.columns,
        policy=args.sequence_policy,
        output_dir=sequence_dir,
    )
    assert len(cleaned_sequences_df) > 0, "No cleaned protein sequences were retained."

    feature_table_df, feature_log_df, feature_summary = extract_protein_feature_table(
        cleaned_sequences_df["cleaned_sequence"],
        output_dir=feature_dir,
    )
    assert len(feature_table_df) > 0, "Protein feature extraction produced no successful rows."

    merged_feature_df = feature_table_df.merge(
        cleaned_sequences_df[["protein_id", "cleaned_sequence", "example_raw_sequence", "occurrence_count"]],
        on=["protein_id", "cleaned_sequence"],
        how="left",
    )
    merged_feature_df.to_csv(feature_dir / "protein_feature_table_enriched.csv", index=False)

    interaction_df, interaction_summary = build_interaction_dataset(
        toxicity_df=toxicity_df,
        protein_features_df=merged_feature_df,
        environmental_constants=ENVIRONMENTAL_CONSTANTS,
        output_dir=report_dir,
    )
    assert len(interaction_df) > 0, "Interaction dataset is empty."
    assert interaction_df["Interaction"].nunique() >= 2, "Interaction target does not contain two classes."

    dataset_artifacts = _save_dataset_artifacts(
        interaction_df=interaction_df,
        successful_features_df=merged_feature_df,
        output_dir=output_root,
        sample_size=args.inspection_sample_size,
        seed=args.random_seed,
    )

    feature_columns, categorical_features, numerical_features = _select_model_columns(interaction_df)
    model_ready_df = interaction_df[feature_columns + ["Interaction", "protein_id"]].copy()
    model_ready_path = output_root / "datasets" / "model_ready_dataset.csv"
    model_ready_df.head(args.inspection_sample_size).to_csv(model_ready_path, index=False)

    benchmark_subset_df = _deterministic_subset(
        interaction_df,
        label_column="Interaction",
        max_rows=args.benchmark_subset_size,
        seed=args.random_seed,
    )
    benchmark_subset_df.to_csv(output_root / "datasets" / "benchmark_subset.csv", index=False)

    X_subset = benchmark_subset_df[feature_columns]
    y_subset = benchmark_subset_df["Interaction"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset,
        y_subset,
        test_size=0.2,
        random_state=args.random_seed,
        stratify=y_subset,
    )
    preprocessor_config = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
    }
    test_metrics_df, _fitted_models = benchmark_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor_config,
        output_dir=benchmark_dir,
        random_state=args.random_seed,
    )
    assert test_metrics_df["model_name"].nunique() >= 4, "Fewer than four models were benchmarked."
    cv_metrics_df = cross_validate_models(
        X=X_subset,
        y=y_subset,
        preprocessor=preprocessor_config,
        cv=args.cv_folds,
        random_state=args.random_seed,
        output_dir=benchmark_dir,
    )

    comparison_df = test_metrics_df.merge(cv_metrics_df, on="model_name", how="left", suffixes=("_test", "_cv"))
    comparison_df.to_csv(benchmark_dir / "model_comparison.csv", index=False)

    full_linear_metrics = None
    if not args.skip_full_linear_baseline:
        feasible, diagnostics = _full_linear_baseline_feasibility(interaction_df)
        if feasible:
            try:
                full_linear_metrics = _run_full_linear_baseline(
                    interaction_df=interaction_df,
                    categorical_features=categorical_features,
                    numerical_features=numerical_features,
                    output_dir=benchmark_dir,
                    seed=args.random_seed,
                )
            except Exception as exc:
                full_linear_metrics = {
                    "status": "failed",
                    "failure_reason": str(exc),
                    "feasibility_diagnostics": diagnostics,
                }
        else:
            full_linear_metrics = {
                "status": "skipped",
                "failure_reason": (
                    "Skipped optional full-data LinearSVC baseline because the "
                    "estimated peak memory exceeded the configured safety threshold."
                ),
                "feasibility_diagnostics": diagnostics,
            }
        write_json(report_dir / "full_linear_baseline.json", {"full_linear_baseline": full_linear_metrics})

    final_summary = {
        "disclaimer": SYNTHETIC_LABEL_DISCLAIMER,
        "toxicity_schema": toxicity_meta["health_report"],
        "ppi_schema": ppi_meta["health_report"],
        "required_toxicity_columns": TOXICITY_REQUIRED_COLUMNS,
        "selected_sequence_columns": sequence_selection_report,
        "sequence_cleaning_summary": sequence_summary,
        "feature_extraction_summary": feature_summary,
        "dataset_assembly_summary": interaction_summary,
        "benchmark_results": test_metrics_df.to_dict(orient="records"),
        "cross_validation_results": cv_metrics_df.to_dict(orient="records"),
        "full_linear_baseline": full_linear_metrics,
        "saved_artifacts": {
            **dataset_artifacts,
            "environment_versions": str(report_dir / "environment_versions.json"),
            "sequence_log": str(sequence_dir / "sequence_processing_log.csv"),
            "sequence_summary": str(sequence_dir / "sequence_preprocessing_summary.json"),
            "protein_feature_table": str(feature_dir / "protein_feature_table.csv"),
            "feature_failures": str(feature_dir / "protein_feature_failures.csv"),
            "metrics_csv": str(benchmark_dir / "model_comparison.csv"),
            "test_metrics_csv": str(benchmark_dir / "test_metrics.csv"),
            "cv_metrics_csv": str(benchmark_dir / "cv_metrics.csv"),
        },
    }
    write_json(report_dir / "final_summary.json", final_summary)

    best_rows = test_metrics_df[test_metrics_df["status"] == "success"].sort_values(by="f1", ascending=False)
    best_model_name = best_rows.iloc[0]["model_name"] if not best_rows.empty else "None"
    report_text = "\n".join(
        [
            "Protein-Nanoparticle Pipeline Final Summary",
            "",
            f"Disclaimer: {SYNTHETIC_LABEL_DISCLAIMER}",
            f"Sequence cleaning policy: {args.sequence_policy}",
            f"Retained unique cleaned sequences: {sequence_summary['retained_unique_count']}",
            f"Successful protein feature rows: {feature_summary['successful_extractions']}",
            f"Interaction dataset rows: {interaction_summary['rows']}",
            f"Benchmark subset rows: {len(benchmark_subset_df)}",
            f"Best benchmark model by test F1: {best_model_name}",
            f"Outputs written under: {output_root.resolve()}",
        ]
    )
    write_text(report_dir / "final_summary.txt", report_text)

    del interaction_df, benchmark_subset_df, X_subset, y_subset, X_train, X_test, y_train, y_test
    gc.collect()


if __name__ == "__main__":
    main()
