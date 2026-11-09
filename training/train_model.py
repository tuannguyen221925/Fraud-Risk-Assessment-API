"""Train and package a credit card fraud detection model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "creditcard.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "fraud_detection_runs"

RAW_FEATURE_ORDER = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)
REQUIRED_COLUMNS = RAW_FEATURE_ORDER + ["Class"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the credit card fraud detection model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to a CSV containing transaction features and Class.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Training device. Use cuda only when a compatible GPU is available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent folder where this training run's artifacts will be saved.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Select and order only the columns used by this project.
    df = df[REQUIRED_COLUMNS].copy()

    # Fail early if values are not numeric or contain missing data.
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    labels = set(df["Class"].unique())
    if labels != {0, 1}:
        raise ValueError(
            f"Class must contain both 0 and 1; found: {labels}"
        )

    return df


def split_dataset(df: pd.DataFrame):
    X = df[RAW_FEATURE_ORDER]
    y = df["Class"].astype(int)

    # Reserve 20% as a final holdout test set.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Split the remaining 80% into 64% train and 16% validation overall.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        stratify=y_train_val,
        random_state=42,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_grid_search(y_train: pd.Series, device: str) -> GridSearchCV:
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["Amount", "Time"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    scale_pos_weight = (
        (y_train == 0).sum()
        / (y_train == 1).sum()
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    tree_method="hist",
                    device=device,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__n_estimators": [100, 200, 300],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=2,
    )


def select_threshold(
    y_val: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(0.05, 1.00, 0.01)
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0}

    for threshold_value in thresholds:
        predictions = (probabilities >= threshold_value).astype(int)
        metrics = {
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
        }

        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold_value)
            best_metrics = metrics

    return best_threshold, best_metrics


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data)

    print(f"Loaded dataset: {args.data}")
    print(f"Rows and columns: {df.shape}")
    print("Class counts:")
    print(df["Class"].value_counts().sort_index())
    print(f"Fraud rate: {df['Class'].mean() * 100:.4f}%")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(df)

    print(
        "\nSplit sizes: "
        f"train={len(X_train)}, "
        f"validation={len(X_val)}, "
        f"test={len(X_test)}"
    )

    grid = build_grid_search(y_train, args.device)
    print("\nStarting Grid Search...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("\nBest cross-validation F1:")
    print(grid.best_score_)
    print("\nBest parameters:")
    print(grid.best_params_)

    # Select a threshold using validation data only.
    val_probabilities = best_model.predict_proba(X_val)[:, 1]
    best_threshold, validation_metrics = select_threshold(
        y_val,
        val_probabilities,
    )

    print("\nBest validation threshold:")
    print(best_threshold)
    print("Validation metrics:")
    print(validation_metrics)

    # Use the untouched test set once for the final evaluation.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    test_precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    test_recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    test_f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )
    test_confusion_matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    print("\nFinal test results:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Normal", "Fraud"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted; [Normal, Fraud]):")
    print(test_confusion_matrix)

    # Give each run its own folder so retraining won't overwrite another run.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    joblib.dump(best_model, run_dir / "fraud_pipeline.pkl")

    threshold_info = {
        "best_threshold": best_threshold,
        "selection_metric": "f1",
        "selected_on": "validation",
        "validation_precision": float(validation_metrics["precision"]),
        "validation_recall": float(validation_metrics["recall"]),
        "validation_f1": float(validation_metrics["f1"]),
    }

    model_info = {
        "algorithm": "XGBoost Pipeline",
        "version": run_id,
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1),
        "threshold": best_threshold,
    }

    with open(run_dir / "best_threshold.json", "w", encoding="utf-8") as file:
        json.dump(threshold_info, file, indent=4)

    with open(run_dir / "model_info.json", "w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=4)

    pd.DataFrame(grid.cv_results_).to_csv(
        run_dir / "grid_results.csv",
        index=False,
    )

    pd.DataFrame([{
        **model_info,
        "confusion_matrix": test_confusion_matrix.tolist(),
    }]).to_csv(
        run_dir / "final_results.csv",
        index=False,
    )

    print(f"\nArtifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
