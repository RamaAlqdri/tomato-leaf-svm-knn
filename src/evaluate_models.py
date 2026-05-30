"""Evaluate saved SVM and KNN models on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .utils import ensure_directories, project_path, target_class_order
except ImportError:  # pragma: no cover - direct script execution
    from utils import ensure_directories, project_path, target_class_order  # type: ignore


def load_test_features() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    features_path = project_path("data/features/features.csv")
    labels_path = project_path("data/features/labels.csv")
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Feature files not found. Run `python main.py --extract-features` first.")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    merged = labels_df.merge(features_df, on="image_id", how="inner")
    test_df = merged[merged["split"] == "test"].copy()
    if test_df.empty:
        raise RuntimeError("No test rows found in features/labels files.")

    feature_columns = [column for column in features_df.columns if column != "image_id"]
    labels = target_class_order(test_df["label"].unique())
    return test_df[feature_columns], test_df["label"], labels


def compute_auc(model, x_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = model.predict_proba(x_test)
        classes = list(model.classes_)
        return float(roc_auc_score(y_test, probabilities, labels=classes, multi_class="ovr", average="macro"))
    except Exception:
        return None


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list[str], output_path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_one_model(model_path: str | Path, model_name: str, x_test: pd.DataFrame, y_test: pd.Series, labels: list[str]) -> dict[str, float | None]:
    model = joblib.load(project_path(model_path))
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "auc_ovr_macro": compute_auc(model, x_test, y_test),
    }

    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    lines = [
        f"Evaluation: {model_name}",
        "",
        f"Accuracy       : {metrics['accuracy']:.4f}",
        f"Precision macro: {metrics['precision_macro']:.4f}",
        f"Recall macro   : {metrics['recall_macro']:.4f}",
        f"F1 macro       : {metrics['f1_macro']:.4f}",
        f"AUC OVR macro  : {metrics['auc_ovr_macro']:.4f}" if metrics["auc_ovr_macro"] is not None else "AUC OVR macro  : not available",
        "",
        "Classification report:",
        report,
    ]
    report_path = project_path(f"reports/evaluation_{model_name.lower()}.txt")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    save_confusion_matrix(
        y_test,
        pd.Series(y_pred),
        labels,
        project_path(f"reports/figures/confusion_matrix_{model_name.lower()}.png"),
        f"Confusion Matrix - {model_name}",
    )
    return metrics


def evaluate_models() -> pd.DataFrame:
    ensure_directories()
    x_test, y_test, labels = load_test_features()

    model_specs = [
        ("models/svm_model.pkl", "SVM"),
        ("models/knn_model.pkl", "KNN"),
    ]
    rows = []
    for model_path, model_name in model_specs:
        path = project_path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Train the model first.")
        metrics = evaluate_one_model(model_path, model_name, x_test, y_test, labels)
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(project_path("reports/test_evaluation_summary.csv"), index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved SVM and KNN models.")
    parser.parse_args()
    evaluate_models()


if __name__ == "__main__":
    main()
