"""Extract explicit numeric image features for SVM and KNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern
from tqdm import tqdm

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:  # pragma: no cover - older scikit-image spelling
    from skimage.feature import greycomatrix as graycomatrix  # type: ignore
    from skimage.feature import greycoprops as graycoprops  # type: ignore

try:
    from . import config
    from .utils import ensure_directories, project_path
except ImportError:  # pragma: no cover - direct script execution
    import config  # type: ignore
    from utils import ensure_directories, project_path  # type: ignore


def normalized_hist(values: np.ndarray, bins: int, value_range: tuple[int, int]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_color_histograms_rgb(image_rgb: np.ndarray) -> tuple[list[float], list[str]]:
    features: list[float] = []
    names: list[str] = []
    channels = ("r", "g", "b")
    for channel_index, channel_name in enumerate(channels):
        hist = normalized_hist(image_rgb[:, :, channel_index], config.RGB_HIST_BINS, (0, 256))
        features.extend(hist.tolist())
        names.extend([f"rgb_{channel_name}_hist_bin_{idx:02d}" for idx in range(config.RGB_HIST_BINS)])
    return features, names


def extract_color_histograms_hsv(image_rgb: np.ndarray) -> tuple[list[float], list[str]]:
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    ranges = {"h": (0, 180), "s": (0, 256), "v": (0, 256)}
    features: list[float] = []
    names: list[str] = []
    for channel_index, channel_name in enumerate(("h", "s", "v")):
        hist = normalized_hist(image_hsv[:, :, channel_index], config.HSV_HIST_BINS, ranges[channel_name])
        features.extend(hist.tolist())
        names.extend([f"hsv_{channel_name}_hist_bin_{idx:02d}" for idx in range(config.HSV_HIST_BINS)])
    return features, names


def extract_glcm_features(gray: np.ndarray) -> tuple[list[float], list[str]]:
    quantized = np.floor(gray.astype(np.float64) / (256 / config.GLCM_LEVELS)).astype(np.uint8)
    quantized = np.clip(quantized, 0, config.GLCM_LEVELS - 1)
    matrix = graycomatrix(
        quantized,
        distances=config.GLCM_DISTANCES,
        angles=config.GLCM_ANGLES,
        levels=config.GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    features: list[float] = []
    names: list[str] = []
    for prop in config.GLCM_PROPERTIES:
        values = graycoprops(matrix, prop)
        for distance_idx, distance in enumerate(config.GLCM_DISTANCES):
            for angle_idx, angle in enumerate(config.GLCM_ANGLES):
                features.append(float(values[distance_idx, angle_idx]))
                names.append(f"glcm_{prop}_d{distance}_a{angle_idx}")
    return features, names


def extract_lbp_features(gray: np.ndarray) -> tuple[list[float], list[str]]:
    lbp = local_binary_pattern(gray, config.LBP_POINTS, config.LBP_RADIUS, method=config.LBP_METHOD)
    bins = config.LBP_POINTS + 2
    hist = normalized_hist(lbp, bins=bins, value_range=(0, bins))
    names = [f"lbp_uniform_bin_{idx:02d}" for idx in range(bins)]
    return hist.tolist(), names


def extract_features_from_image(path: str | Path) -> tuple[list[float], list[str]]:
    with Image.open(path) as image:
        image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    feature_parts: list[float] = []
    feature_names: list[str] = []
    for extractor in (
        extract_color_histograms_rgb,
        extract_color_histograms_hsv,
    ):
        values, names = extractor(image_rgb)
        feature_parts.extend(values)
        feature_names.extend(names)

    values, names = extract_glcm_features(gray)
    feature_parts.extend(values)
    feature_names.extend(names)

    values, names = extract_lbp_features(gray)
    feature_parts.extend(values)
    feature_names.extend(names)
    return feature_parts, feature_names


def extract_features(metadata_path: str | Path = "data/processed/dataset_metadata.csv") -> pd.DataFrame:
    ensure_directories()
    metadata_file = project_path(metadata_path)
    if not metadata_file.exists():
        raise FileNotFoundError("Metadata not found. Run `python main.py --prepare` first.")

    metadata_df = pd.read_csv(metadata_file)
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None

    for row in tqdm(metadata_df.to_dict("records"), desc="Extracting features"):
        image_path = project_path(str(row["processed_path"]))
        values, names = extract_features_from_image(image_path)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature name mismatch while extracting images.")

        feature_row = {"image_id": row["image_id"]}
        feature_row.update({name: value for name, value in zip(names, values)})
        feature_rows.append(feature_row)
        label_rows.append(
            {
                "image_id": row["image_id"],
                "source_path": row["source_path"],
                "processed_path": row["processed_path"],
                "label": row["label"],
                "split": row["split"],
            }
        )

    if feature_names is None:
        raise RuntimeError("No features extracted.")

    features_df = pd.DataFrame(feature_rows)
    labels_df = pd.DataFrame(label_rows)
    feature_names_df = pd.DataFrame({"feature_name": feature_names})

    features_df.to_csv(project_path("data/features/features.csv"), index=False)
    labels_df.to_csv(project_path("data/features/labels.csv"), index=False)
    feature_names_df.to_csv(project_path("data/features/feature_names.csv"), index=False)
    return features_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract image features for SVM and KNN.")
    parser.add_argument("--metadata-path", default="data/processed/dataset_metadata.csv")
    args = parser.parse_args()
    extract_features(args.metadata_path)


if __name__ == "__main__":
    main()

