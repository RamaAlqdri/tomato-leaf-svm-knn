"""Train and cross-validate an SVM classifier."""

from __future__ import annotations

import argparse
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from . import config
    from .utils import ensure_directories, project_path
except ImportError:  # pragma: no cover - direct script execution
    import config  # type: ignore
    from utils import ensure_directories, project_path  # type: ignore


def load_training_features() -> tuple[pd.DataFrame, pd.Series]:
    features_path = project_path("data/features/features.csv")
    labels_path = project_path("data/features/labels.csv")
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Feature files not found. Run `python main.py --extract-features` first.")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    merged = labels_df.merge(features_df, on="image_id", how="inner")
    train_df = merged[merged["split"] == "train"].copy()
    if train_df.empty:
        raise RuntimeError("No training rows found in features/labels files.")

    feature_columns = [column for column in features_df.columns if column != "image_id"]
    return train_df[feature_columns], train_df["label"]


def build_svm_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=config.SVM_KERNEL,
                    C=config.SVM_C,
                    gamma=config.SVM_GAMMA,
                    probability=True,
                    random_state=config.RANDOM_STATE,
                    cache_size=1000,
                ),
            ),
        ]
    )


def run_cross_validation(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "auc_ovr_macro": "roc_auc_ovr",
    }

    results: list[dict[str, float | int | str]] = []
    min_class_count = int(y.value_counts().min())
    for fold in config.CV_FOLDS:
        if fold > min_class_count:
            warnings.warn(f"Skipping {fold}-Fold CV because smallest class has only {min_class_count} samples.")
            continue
        cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=config.RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            cv_result = cross_validate(
                model,
                x,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=config.N_JOBS,
                error_score="raise",
                return_train_score=False,
            )

        row: dict[str, float | int | str] = {
            "model": "SVM",
            "kernel": config.SVM_KERNEL,
            "fold": fold,
        }
        for metric in scoring:
            scores = cv_result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(scores.mean())
            row[f"{metric}_std"] = float(scores.std())
        row["fit_time_mean"] = float(cv_result["fit_time"].mean())
        row["score_time_mean"] = float(cv_result["score_time"].mean())
        results.append(row)
    return pd.DataFrame(results)


def train_svm() -> pd.DataFrame:
    ensure_directories()
    x_train, y_train = load_training_features()
    model = build_svm_pipeline()
    results_df = run_cross_validation(model, x_train, y_train)
    results_df.to_csv(project_path("reports/svm_results.csv"), index=False)

    final_model = build_svm_pipeline()
    final_model.fit(x_train, y_train)
    joblib.dump(final_model, project_path("models/svm_model.pkl"))
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVM classifier.")
    parser.parse_args()
    train_svm()


if __name__ == "__main__":
    main()

