from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"


def load_model_bundle() -> tuple[Any | None, list[str]]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, []

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, list(metadata["features"])


def parse_payload(data: dict[str, Any], features: list[str]) -> dict[str, float]:
    missing = [feature for feature in features if feature not in data]
    extra = [key for key in data if key not in features]

    if missing:
        raise ValueError(f"Missing features: {missing}")
    if extra:
        raise ValueError(f"Unexpected features: {extra}")

    parsed: dict[str, float] = {}
    for feature in features:
        try:
            value = float(data[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric.") from exc

        if not math.isfinite(value):
            raise ValueError(f"{feature} must be finite.")

        parsed[feature] = value

    if parsed.get("CHAS") not in {0.0, 1.0}:
        raise ValueError("CHAS must be 0 or 1.")

    return parsed


def create_app(
    model_override: Any | None = None,
    features_override: list[str] | None = None,
) -> Flask:
    app = Flask(__name__)

    if model_override is not None:
        model = model_override
        features = features_override or []
    else:
        model, features = load_model_bundle()

    @app.get("/")
    def home():
        return render_template(
            "home.html",
            features=features,
            model_ready=model is not None,
            prediction_text=None,
        )

    @app.get("/health")
    def health():
        model_ready = model is not None
        return (
            jsonify(
                {
                    "status": "ok" if model_ready else "model_not_trained",
                    "model_ready": model_ready,
                }
            ),
            200 if model_ready else 503,
        )

    @app.post("/predict_api")
    def predict_api():
        if model is None:
            return (
                jsonify({"error": "Model artifact is unavailable. Run python -m src.train."}),
                503,
            )

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            return jsonify({"error": "Expected JSON body: {'data': {...}}"}), 400

        try:
            parsed = parse_payload(payload["data"], features)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        frame = pd.DataFrame([parsed], columns=features)
        prediction = float(model.predict(frame)[0])
        return jsonify(
            {
                "predicted_median_value_thousands_usd": round(
                    prediction,
                    3,
                )
            }
        )

    @app.post("/predict")
    def predict_form():
        if model is None:
            return (
                render_template(
                    "home.html",
                    features=features,
                    model_ready=False,
                    prediction_text="Model artifact is unavailable.",
                ),
                503,
            )

        try:
            parsed = parse_payload(dict(request.form), features)
        except ValueError as exc:
            return (
                render_template(
                    "home.html",
                    features=features,
                    model_ready=True,
                    prediction_text=f"Input error: {exc}",
                ),
                400,
            )

        frame = pd.DataFrame([parsed], columns=features)
        prediction = float(model.predict(frame)[0])
        prediction_text = f"Predicted historical MEDV value: {prediction:.2f} ($1,000 units)"

        return render_template(
            "home.html",
            features=features,
            model_ready=True,
            prediction_text=prediction_text,
        )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
