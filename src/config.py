from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "HousingData.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
TARGET = "MEDV"

# Historical B variable is deliberately excluded due to its racial construction.
FEATURES = [
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
    "LSTAT",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
