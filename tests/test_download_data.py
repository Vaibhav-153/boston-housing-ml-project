import numpy as np

from scripts.download_data import EXPECTED_ROWS, SOURCE_FEATURES, build_dataframe


def make_npz_payload(tmp_path):
    path = tmp_path / "boston.npz"
    x = np.zeros((EXPECTED_ROWS, len(SOURCE_FEATURES)), dtype=float)
    y = np.arange(EXPECTED_ROWS, dtype=float)
    np.savez(path, x=x, y=y)
    return path.read_bytes()


def test_build_dataframe_has_expected_schema(tmp_path):
    frame = build_dataframe(make_npz_payload(tmp_path))

    assert frame.shape == (EXPECTED_ROWS, len(SOURCE_FEATURES) + 1)
    assert frame.columns.tolist() == [*SOURCE_FEATURES, "MEDV"]
    assert frame["MEDV"].iloc[-1] == EXPECTED_ROWS - 1
