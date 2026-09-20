from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACT_DIR,
    DATA_PATH,
    FEATURES,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
    TEST_SIZE,
)


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError("Dataset missing. Run: python scripts/download_data.py")

    df = pd.read_csv(path)

    required_columns = [*FEATURES, TARGET]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    return X, y


def build_candidates() -> dict[str, Pipeline]:
    return {
        "dummy_median": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "linear_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[Pipeline, dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    candidates = build_candidates()
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_rmse: dict[str, float] = {}

    for name, candidate in candidates.items():
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        cv_rmse[name] = float(-np.mean(scores))

    selected_name = min(cv_rmse, key=cv_rmse.get)
    model = candidates[selected_name]
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics: dict[str, object] = {
        "selected_model": selected_name,
        "cv_rmse_by_model": {name: round(value, 4) for name, value in cv_rmse.items()},
        "test": {
            "mae": round(float(mean_absolute_error(y_test, pred)), 4),
            "rmse": round(
                float(np.sqrt(mean_squared_error(y_test, pred))),
                4,
            ),
            "r2": round(float(r2_score(y_test, pred)), 4),
        },
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
    }

    return model, metrics


def main() -> None:
    X, y = load_data()
    model, metrics = train_and_evaluate(X, y)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "selected_model": metrics["selected_model"],
    }

    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print("\nSaved artifacts/model.joblib")
    print("Saved reports/metrics.json")


if __name__ == "__main__":
    main()
