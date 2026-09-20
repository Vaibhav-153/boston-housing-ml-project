from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "HousingData.csv"
URL = "https://raw.githubusercontent.com/Vaibhav-153/boston-housing-ml-project/main/HousingData.csv"


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        TARGET.write_bytes(response.read())
    print(f"Saved dataset to {TARGET}")


if __name__ == "__main__":
    main()
