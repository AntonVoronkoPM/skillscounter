import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skillscounter import data


@pytest.fixture(scope="module")
def df():
    data = [
        {"text": "The Role", "target": 0, "tag": "text inside DIV"},
        {"text": "SQL knowledge and query optimization preferred", "target": 1, "tag": "LI"},
    ]
    df = pd.DataFrame(data * 10)
    return df

def test_preprocess(df):
    assert "feature" not in df.columns
    df = data.preprocess(df=df)
    assert "feature" in df.columns


def test_get_data_splits(df):
    df = data.preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
        X=df.feature.to_numpy(), y=df.target
    )
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) / float(len(df)) == pytest.approx(0.7, abs=0.05)  # 0.7 ± 0.05
    assert len(X_val) / float(len(df)) == pytest.approx(0.15, abs=0.05)  # 0.15 ± 0.05
    assert len(X_test) / float(len(df)) == pytest.approx(0.15, abs=0.05)  # 0.15 ± 0.05
