from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "HousingData.csv"

# Keras hosts the original Boston Housing arrays from StatLib.
SOURCE_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "boston_housing.npz"
)
SOURCE_SHA256 = "f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5"

SOURCE_FEATURES = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]
EXPECTED_ROWS = 506


def download_source() -> bytes:
    with urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Downloaded dataset failed the SHA-256 integrity check.")

    return payload


def build_dataframe(payload: bytes) -> pd.DataFrame:
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        features = archive["x"]
        target = archive["y"]

    if features.shape != (EXPECTED_ROWS, len(SOURCE_FEATURES)):
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    if target.shape != (EXPECTED_ROWS,):
        raise ValueError(f"Unexpected target shape: {target.shape}")

    frame = pd.DataFrame(features, columns=SOURCE_FEATURES)
    frame["MEDV"] = target
    return frame


def main() -> None:
    frame = build_dataframe(download_source())
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET, index=False)
    print(f"Saved {len(frame)} rows to {TARGET}")


if __name__ == "__main__":
    main()
