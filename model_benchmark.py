from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from utils import ensure_directory, write_json

matplotlib.use("Agg")


def build_preprocessor(
    categorical_features: list[str],
    numerical_features: list[str],
    dense_output: bool = False,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense_output else 1.0,
    )


def get_candidate_models(random_state: int = 42) -> dict[str, dict[str, Any]]:
    return {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                random_state=random_state,
            ),
            "dense_output": False,
        },
        "LinearSVC": {
            "estimator": LinearSVC(random_state=random_state, dual="auto", max_iter=5000),
            "dense_output": False,
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                n_estimators=120,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            "dense_output": True,
        },
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingClassifier(
                random_state=random_state,
                max_iter=200,
            ),
            "dense_output": True,
        },
    }


def _preprocessor_config(preprocessor: dict[str, Any]) -> tuple[list[str], list[str]]:
    categorical_features = list(preprocessor["categorical_features"])
    numerical_features = list(preprocessor["numerical_features"])
    return categorical_features, numerical_features


def _build_pipeline(model_name: str, preprocessor: dict[str, Any], random_state: int = 42) -> Pipeline:
    candidate_models = get_candidate_models(random_state=random_state)
    model_spec = candidate_models[model_name]
    categorical_features, numerical_features = _preprocessor_config(preprocessor)
    transformer = build_preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        dense_output=model_spec["dense_output"],
    )
    return Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("classifier", model_spec["estimator"]),
        ]
    )


def _score_output(model: Pipeline, X_test: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1], "predict_proba"
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test), "decision_function"
    return None, None


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str | Path,
) -> Path:
    matrix = confusion_matrix(y_true, y_pred)
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"{model_name} Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    output_file = Path(output_path)
    ensure_directory(output_file.parent)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file


def plot_roc_curve(
    y_true: pd.Series,
    y_score: np.ndarray,
    model_name: str,
    output_path: str | Path,
) -> tuple[Path, float]:
    fpr, tpr, _thresholds = roc_curve(y_true, y_score)
    score = roc_auc_score(y_true, y_score)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {score:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"{model_name} ROC Curve")
    axis.legend(loc="lower right")
    output_file = Path(output_path)
    ensure_directory(output_file.parent)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file, float(score)


def plot_pr_curve(
    y_true: pd.Series,
    y_score: np.ndarray,
    model_name: str,
    output_path: str | Path,
) -> tuple[Path, float]:
    precision, recall, _thresholds = precision_recall_curve(y_true, y_score)
    score = average_precision_score(y_true, y_score)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall, precision, label=f"PR-AUC = {score:.4f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"{model_name} Precision-Recall Curve")
    axis.legend(loc="lower left")
    output_file = Path(output_path)
    ensure_directory(output_file.parent)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file, float(score)


def evaluate_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: dict[str, Any],
    output_dir: str | Path,
    random_state: int = 42,
) -> tuple[dict[str, Any], Pipeline | None]:
    model_output_dir = ensure_directory(Path(output_dir) / model_name)
    try:
        pipeline = _build_pipeline(model_name, preprocessor=preprocessor, random_state=random_state)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score, score_method = _score_output(pipeline, X_test)

        metrics: dict[str, Any] = {
            "model_name": model_name,
            "status": "success",
            "score_method": score_method,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "failure_reason": None,
        }

        metrics["confusion_matrix_path"] = str(
            plot_confusion_matrix(y_test, y_pred, model_name, model_output_dir / "confusion_matrix.png")
        )
        if y_score is not None:
            roc_path, roc_auc = plot_roc_curve(y_test, y_score, model_name, model_output_dir / "roc_curve.png")
            pr_path, pr_auc = plot_pr_curve(y_test, y_score, model_name, model_output_dir / "pr_curve.png")
            metrics["roc_auc"] = roc_auc
            metrics["pr_auc"] = pr_auc
            metrics["roc_curve_path"] = str(roc_path)
            metrics["pr_curve_path"] = str(pr_path)
        else:
            metrics["roc_curve_path"] = None
            metrics["pr_curve_path"] = None
            metrics["failure_reason"] = "No probability or decision_function available for ROC/PR scoring."

        joblib.dump(pipeline, model_output_dir / "pipeline.joblib")
        write_json(model_output_dir / "metrics.json", metrics)
        return metrics, pipeline
    except Exception as exc:
        failure_metrics = {
            "model_name": model_name,
            "status": "failed",
            "score_method": None,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "failure_reason": str(exc),
            "confusion_matrix_path": None,
            "roc_curve_path": None,
            "pr_curve_path": None,
        }
        write_json(model_output_dir / "metrics.json", failure_metrics)
        return failure_metrics, None


def benchmark_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: dict[str, Any],
    output_dir: str | Path,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    benchmark_output_dir = ensure_directory(output_dir)
    records: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name in get_candidate_models(random_state=random_state):
        metrics, pipeline = evaluate_single_model(
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            output_dir=benchmark_output_dir,
            random_state=random_state,
        )
        records.append(metrics)
        if pipeline is not None:
            fitted_models[model_name] = pipeline
    results_df = pd.DataFrame(records).sort_values(by=["status", "f1"], ascending=[True, False])
    results_df.to_csv(benchmark_output_dir / "test_metrics.csv", index=False)
    return results_df, fitted_models


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: dict[str, Any],
    cv: int = 5,
    random_state: int = 42,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": make_scorer(
            roc_auc_score,
            response_method=("decision_function", "predict_proba"),
        ),
        "pr_auc": make_scorer(
            average_precision_score,
            response_method=("decision_function", "predict_proba"),
        ),
    }

    records: list[dict[str, Any]] = []
    for model_name in get_candidate_models(random_state=random_state):
        pipeline = _build_pipeline(model_name, preprocessor=preprocessor, random_state=random_state)
        try:
            cv_results = cross_validate(
                pipeline,
                X,
                y,
                cv=cv_splitter,
                scoring=scoring,
                error_score=np.nan,
                n_jobs=1,
                return_train_score=False,
            )
            record = {
                "model_name": model_name,
                "status": "success",
                "failure_reason": None,
            }
            for metric_name in scoring:
                values = cv_results[f"test_{metric_name}"]
                record[f"{metric_name}_mean"] = float(np.nanmean(values))
                record[f"{metric_name}_std"] = float(np.nanstd(values))
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "failure_reason": str(exc),
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "precision_mean": np.nan,
                    "precision_std": np.nan,
                    "recall_mean": np.nan,
                    "recall_std": np.nan,
                    "f1_mean": np.nan,
                    "f1_std": np.nan,
                    "roc_auc_mean": np.nan,
                    "roc_auc_std": np.nan,
                    "pr_auc_mean": np.nan,
                    "pr_auc_std": np.nan,
                }
            )

    cv_df = pd.DataFrame(records).sort_values(by=["status", "f1_mean"], ascending=[True, False])
    if output_dir is not None:
        ensure_directory(output_dir)
        cv_df.to_csv(Path(output_dir) / "cv_metrics.csv", index=False)
    return cv_df
