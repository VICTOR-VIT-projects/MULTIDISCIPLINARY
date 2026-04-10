from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TOXICITY_DATASET = "ucimachinelearning/nanoparticle-toxicity-dataset"
DEFAULT_PPI_DATASET = "spandansureja/ppi-dataset"

SYNTHETIC_LABEL_DISCLAIMER = (
    "Interaction labels in this project are heuristic synthetic pseudo-labels "
    "generated from nanoparticle surfcharge and protein pI. They are not "
    "experimentally validated biological or clinical ground truth."
)

ENVIRONMENTAL_CONSTANTS = {
    "Medium_pH": 7.4,
    "Ionic_Strength": 0.15,
    "Temperature_C": 25.0,
    "Temperature_K": 298.15,
}


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def write_text(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_environment_versions() -> dict[str, str]:
    import pandas as pd
    import sklearn
    from Bio import __version__ as biopython_version

    return {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "biopython": biopython_version,
    }


def dataframe_preview_records(df, n: int = 5) -> list[dict[str, Any]]:
    preview_df = df.head(n).replace({np.nan: None})
    return preview_df.to_dict(orient="records")
