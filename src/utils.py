"""Shared helpers for paths, labels, image discovery, and reports."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

try:
    from . import config
except ImportError:  # pragma: no cover - direct script execution
    import config  # type: ignore


def project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return config.PROJECT_ROOT / path


def relative_to_project(path_value: str | Path) -> str:
    """Return a stable project-relative path string when possible."""

    path = Path(path_value)
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_directories() -> None:
    """Create all output directories used by the pipeline."""

    for directory in [
        config.PROCESSED_DIR,
        config.FEATURES_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
        config.MODELS_DIR,
        "notebooks",
    ]:
        project_path(directory).mkdir(parents=True, exist_ok=True)


def normalize_text(value: str) -> str:
    """Normalize folder names for flexible label mapping."""

    value = value.lower().replace("tomato___", " ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def simplify_label(folder_name: str) -> str | None:
    """Map PlantVillage-style folder names to the five journal classes."""

    text = normalize_text(folder_name)
    tokens = set(text.split())

    if {"early", "blight"}.issubset(tokens):
        return "Early Blight"
    if {"late", "blight"}.issubset(tokens):
        return "Late Blight"
    if "healthy" in tokens or "health" in tokens:
        return "Healthy"
    if "mosaic" in tokens:
        return "Mosaic Virus"
    if "yellow" in tokens and "curl" in tokens:
        return "Yellow Leaf Curl Virus"
    return None


def slugify_label(label: str) -> str:
    """Convert a class label into a filesystem-friendly name."""

    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS


def iter_image_paths(dataset_dir: str | Path | None = None) -> Iterable[Path]:
    """Yield supported image files under the configured dataset directory."""

    root = project_path(dataset_dir or config.DATASET_DIR)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. "
            "Change DATASET_DIR in src/config.py or place the dataset in data/raw."
        )

    for path in sorted(root.rglob("*")):
        if path.is_file() and is_supported_image(path):
            yield path


def infer_original_split(path: Path) -> str:
    """Infer an existing dataset split name from path parts, if present."""

    split_aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    for part in path.parts:
        normalized = part.lower()
        if normalized in split_aliases:
            return split_aliases[normalized]
    return "unspecified"


def make_image_id(path_value: str | Path) -> str:
    """Create a deterministic id from a path."""

    value = str(path_value).replace("\\", "/")
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def target_class_order(labels: Iterable[str]) -> list[str]:
    """Return labels ordered by the journal class order, then alphabetically."""

    present = set(labels)
    ordered = [label for label in config.TARGET_CLASSES if label in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def write_markdown(path_value: str | Path, content: str) -> None:
    path = project_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")

