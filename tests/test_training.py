import numpy as np
import pandas as pd

from src.config import FEATURES
from src.train import build_candidates, train_and_evaluate


def make_training_data(rows: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)

    X = pd.DataFrame({feature: rng.normal(size=rows) for feature in FEATURES})

    X["CHAS"] = rng.integers(0, 2, size=rows)

    y = 20.0 + 2.5 * X["RM"] - 1.2 * X["LSTAT"] + rng.normal(scale=0.5, size=rows)

    return X, pd.Series(y, name="MEDV")


def test_candidates_are_available():
    candidates = build_candidates()

    assert {
        "dummy_median",
        "linear_regression",
        "ridge",
        "random_forest",
    }.issubset(candidates)


def test_train_and_evaluate_returns_metrics():
    X, y = make_training_data()

    model, metrics = train_and_evaluate(X, y)

    assert hasattr(model, "predict")
    assert metrics["selected_model"] in {
        "dummy_median",
        "linear_regression",
        "ridge",
        "random_forest",
    }
    assert "mae" in metrics["test"]
    assert "rmse" in metrics["test"]
    assert "r2" in metrics["test"]
    assert metrics["train_rows"] > 0
    assert metrics["test_rows"] > 0
