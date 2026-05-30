"""Analyze local tomato leaf image dataset structure and image integrity."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

try:
    from . import config
    from .utils import (
        ensure_directories,
        infer_original_split,
        iter_image_paths,
        project_path,
        relative_to_project,
        simplify_label,
        target_class_order,
        write_markdown,
    )
except ImportError:  # pragma: no cover - direct script execution
    import config  # type: ignore
    from utils import (  # type: ignore
        ensure_directories,
        infer_original_split,
        iter_image_paths,
        project_path,
        relative_to_project,
        simplify_label,
        target_class_order,
        write_markdown,
    )


def inspect_image(path: Path) -> tuple[bool, int | None, int | None, str | None]:
    """Verify an image and return validity, width, height, and error message."""

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return True, width, height, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, None, None, str(exc)


def analyze_dataset(dataset_dir: str | Path | None = None) -> pd.DataFrame:
    ensure_directories()
    dataset_root = project_path(dataset_dir or config.DATASET_DIR)

    records: list[dict[str, object]] = []
    corrupt_records: list[dict[str, object]] = []
    image_paths = list(iter_image_paths(dataset_root))

    for path in tqdm(image_paths, desc="Analyzing images"):
        original_class = path.parent.name
        mapped_label = simplify_label(original_class)
        is_valid, width, height, error = inspect_image(path)
        record = {
            "path": relative_to_project(path),
            "original_split": infer_original_split(path),
            "original_class": original_class,
            "mapped_label": mapped_label or "Ignored/Unknown",
            "is_target_class": mapped_label in config.TARGET_CLASSES,
            "extension": path.suffix.lower(),
            "is_valid": is_valid,
            "width": width,
            "height": height,
            "error": error,
        }
        records.append(record)
        if not is_valid:
            corrupt_records.append(record)

    if not records:
        raise RuntimeError(f"No supported image files found under {dataset_root}")

    detail_df = pd.DataFrame(records)
    valid_df = detail_df[detail_df["is_valid"]].copy()

    split_counts = (
        detail_df.pivot_table(
            index="original_class",
            columns="original_split",
            values="path",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    grouped = (
        detail_df.groupby(["original_class", "mapped_label", "is_target_class"], dropna=False)
        .agg(
            image_count=("path", "count"),
            valid_images=("is_valid", "sum"),
            corrupt_images=("is_valid", lambda values: int((~values).sum())),
            extensions=("extension", lambda values: ", ".join(sorted(set(values)))),
        )
        .reset_index()
    )

    size_stats = (
        valid_df.groupby("original_class")
        .agg(
            mean_width=("width", "mean"),
            mean_height=("height", "mean"),
            min_width=("width", "min"),
            min_height=("height", "min"),
            max_width=("width", "max"),
            max_height=("height", "max"),
        )
        .reset_index()
    )

    summary_df = grouped.merge(size_stats, on="original_class", how="left")
    summary_df = summary_df.merge(split_counts, on="original_class", how="left")
    summary_df = summary_df.sort_values(["is_target_class", "mapped_label", "original_class"], ascending=[False, True, True])

    summary_path = project_path("reports/dataset_summary.csv")
    detail_path = project_path("reports/dataset_files_detail.csv")
    corrupt_path = project_path("reports/corrupt_images.csv")

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    pd.DataFrame(corrupt_records).to_csv(corrupt_path, index=False)

    write_dataset_markdown(summary_df, detail_df, dataset_root)
    return summary_df


def write_dataset_markdown(summary_df: pd.DataFrame, detail_df: pd.DataFrame, dataset_root: Path) -> None:
    total_images = len(detail_df)
    total_valid = int(detail_df["is_valid"].sum())
    total_corrupt = total_images - total_valid
    target_df = detail_df[detail_df["is_target_class"]]
    target_counts = target_df.groupby("mapped_label")["path"].count().to_dict()
    ignored_classes = sorted(
        detail_df.loc[~detail_df["is_target_class"], "original_class"].dropna().unique().tolist()
    )
    extension_counts = Counter(detail_df["extension"])

    class_rows = []
    for _, row in summary_df.iterrows():
        class_rows.append(
            "| {original_class} | {mapped_label} | {image_count} | {valid_images} | "
            "{corrupt_images} | {extensions} | {mean_width:.1f} x {mean_height:.1f} |".format(
                original_class=row["original_class"],
                mapped_label=row["mapped_label"],
                image_count=int(row["image_count"]),
                valid_images=int(row["valid_images"]),
                corrupt_images=int(row["corrupt_images"]),
                extensions=row["extensions"],
                mean_width=float(row["mean_width"]) if pd.notna(row["mean_width"]) else 0.0,
                mean_height=float(row["mean_height"]) if pd.notna(row["mean_height"]) else 0.0,
            )
        )

    target_lines = []
    for label in target_class_order(target_counts.keys()):
        target_lines.append(f"- {label}: {target_counts.get(label, 0)} gambar")

    ignored_lines = [f"- {name}" for name in ignored_classes] or ["- Tidak ada"]
    extension_lines = [f"- `{ext}`: {count}" for ext, count in sorted(extension_counts.items())]

    content = f"""# Analisis Dataset Lokal

Dataset dianalisis dari: `{relative_to_project(dataset_root)}`

## Ringkasan

- Total file gambar terdeteksi: {total_images}
- Total gambar valid: {total_valid}
- Total gambar rusak/corrupt: {total_corrupt}
- Target kelas jurnal: {len(config.TARGET_CLASSES)} kelas
- File detail per gambar: `reports/dataset_files_detail.csv`
- File ringkasan per kelas: `reports/dataset_summary.csv`
- File gambar rusak: `reports/corrupt_images.csv`

## Distribusi Kelas Target

{chr(10).join(target_lines)}

## Kelas Diabaikan

Folder berikut ditemukan di dataset lokal tetapi tidak termasuk lima kelas target jurnal:

{chr(10).join(ignored_lines)}

## Format File

{chr(10).join(extension_lines)}

## Ringkasan Per Folder

| Folder Asli | Label Mapping | Jumlah | Valid | Corrupt | Ekstensi | Ukuran Rata-Rata |
|---|---:|---:|---:|---:|---|---:|
{chr(10).join(class_rows)}

## Catatan

Dataset lokal memiliki struktur `train` dan `val`, tetapi pipeline replikasi ini akan mengambil maksimal
{config.MAX_IMAGES_PER_CLASS} gambar valid per kelas target lalu membuat split stratified 80/20 baru.
Langkah ini mengikuti rancangan jurnal: 1000 gambar per kelas, 800 training dan 200 testing per kelas.
"""

    write_markdown("reports/dataset_analysis.md", content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local tomato leaf dataset.")
    parser.add_argument("--dataset-dir", default=None, help="Override dataset directory.")
    args = parser.parse_args()
    analyze_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()

