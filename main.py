"""Command-line entrypoint for the full tomato leaf SVM/KNN pipeline."""

from __future__ import annotations

import argparse

from src.analyze_dataset import analyze_dataset
from src.compare_results import compare_results
from src.evaluate_models import evaluate_models
from src.feature_extraction import extract_features
from src.prepare_dataset import prepare_dataset
from src.train_knn import train_knn
from src.train_svm import train_svm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tomato leaf disease classification with SVM and KNN.")
    parser.add_argument("--analyze-dataset", action="store_true", help="Analyze local dataset structure.")
    parser.add_argument("--prepare", action="store_true", help="Prepare resized images and train/test metadata.")
    parser.add_argument("--extract-features", action="store_true", help="Extract RGB, HSV, GLCM, and LBP features.")
    parser.add_argument("--train-svm", action="store_true", help="Train and cross-validate SVM.")
    parser.add_argument("--train-knn", action="store_true", help="Train and cross-validate KNN.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved models on test split.")
    parser.add_argument("--compare", action="store_true", help="Compare SVM and KNN results.")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all = args.all

    if run_all or args.analyze_dataset:
        analyze_dataset()
    if run_all or args.prepare:
        prepare_dataset()
    if run_all or args.extract_features:
        extract_features()
    if run_all or args.train_svm:
        train_svm()
    if run_all or args.train_knn:
        train_knn()
    if run_all or args.evaluate:
        evaluate_models()
    if run_all or args.compare:
        compare_results()

    if not any(vars(args).values()):
        print("No action selected. Use --help to see available pipeline steps.")


if __name__ == "__main__":
    main()

