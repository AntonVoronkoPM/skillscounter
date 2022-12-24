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
    assert "text_tag" not in df.columns
    df = data.preprocess(df=df)
    assert "text_tag" in df.columns
