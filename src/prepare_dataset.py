"""Prepare metadata, resized images, and stratified train/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from . import config
    from .utils import (
        ensure_directories,
        infer_original_split,
        iter_image_paths,
        make_image_id,
        project_path,
        relative_to_project,
        simplify_label,
        slugify_label,
        target_class_order,
    )
except ImportError:  # pragma: no cover - direct script execution
    import config  # type: ignore
    from utils import (  # type: ignore
        ensure_directories,
        infer_original_split,
        iter_image_paths,
        make_image_id,
        project_path,
        relative_to_project,
        simplify_label,
        slugify_label,
        target_class_order,
    )


def read_valid_target_records(dataset_dir: str | Path | None = None) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in iter_image_paths(dataset_dir or config.DATASET_DIR):
        label = simplify_label(path.parent.name)
        if label not in config.TARGET_CLASSES:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except (UnidentifiedImageError, OSError, ValueError):
            continue

        rel_path = relative_to_project(path)
        records.append(
            {
                "image_id": make_image_id(rel_path),
                "source_path": rel_path,
                "original_split": infer_original_split(path),
                "original_class": path.parent.name,
                "label": label,
                "source_width": width,
                "source_height": height,
            }
        )

    if not records:
        raise RuntimeError("No valid target-class images found. Check DATASET_DIR and class folder names.")

    return pd.DataFrame(records)


def limit_per_class(df: pd.DataFrame, max_images_per_class: int) -> pd.DataFrame:
    limited_parts = []
    for label in target_class_order(df["label"].unique()):
        class_df = df[df["label"] == label]
        if len(class_df) > max_images_per_class:
            class_df = class_df.sample(n=max_images_per_class, random_state=config.RANDOM_STATE)
        limited_parts.append(class_df)
    return pd.concat(limited_parts, ignore_index=True).sort_values(["label", "source_path"]).reset_index(drop=True)


def resize_and_save_images(df: pd.DataFrame) -> pd.DataFrame:
    output_records: list[dict[str, object]] = []
    processed_root = project_path(config.PROCESSED_DIR) / "images"
    processed_root.mkdir(parents=True, exist_ok=True)

    for record in tqdm(df.to_dict("records"), desc="Preparing images"):
        source_path = project_path(str(record["source_path"]))
        label = str(record["label"])
        image_id = str(record["image_id"])
        class_dir = processed_root / slugify_label(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        output_path = class_dir / f"{image_id}.png"

        with Image.open(source_path) as image:
            image = image.convert("RGB")
            image = image.resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)
            image.save(output_path)

        updated = dict(record)
        updated["processed_path"] = relative_to_project(output_path)
        updated["processed_width"] = config.IMAGE_SIZE[0]
        updated["processed_height"] = config.IMAGE_SIZE[1]
        updated["color_mode"] = "RGB"
        updated["pixel_value_range"] = "0-255; normalized during feature extraction/model scaling"
        output_records.append(updated)

    return pd.DataFrame(output_records)


def prepare_dataset(
    dataset_dir: str | Path | None = None,
    max_images_per_class: int = config.MAX_IMAGES_PER_CLASS,
    test_size: float = config.TEST_SIZE,
) -> pd.DataFrame:
    ensure_directories()
    df = read_valid_target_records(dataset_dir)
    df = limit_per_class(df, max_images_per_class=max_images_per_class)
    df = resize_and_save_images(df)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    metadata_df = pd.concat([train_df, test_df], ignore_index=True)
    metadata_df["label"] = pd.Categorical(metadata_df["label"], categories=config.TARGET_CLASSES, ordered=True)
    metadata_df = metadata_df.sort_values(["split", "label", "source_path"]).reset_index(drop=True)

    metadata_df.to_csv(project_path("data/processed/dataset_metadata.csv"), index=False)
    train_df.sort_values(["label", "source_path"]).to_csv(project_path("data/processed/train_metadata.csv"), index=False)
    test_df.sort_values(["label", "source_path"]).to_csv(project_path("data/processed/test_metadata.csv"), index=False)
    return metadata_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tomato leaf dataset metadata and images.")
    parser.add_argument("--dataset-dir", default=None, help="Override dataset directory.")
    parser.add_argument("--max-images-per-class", type=int, default=config.MAX_IMAGES_PER_CLASS)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    args = parser.parse_args()
    prepare_dataset(args.dataset_dir, args.max_images_per_class, args.test_size)


if __name__ == "__main__":
    main()

