import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from skillscounter import utils


def test_json_to_dataframe():
    filepath = Path(config.BASE_DIR, "tests", "code", "7-14-20-raw.json")
    with open(filepath) as fp:
        df = utils.json_to_dataframe(json.load(fp))
    assert type(df) is pd.DataFrame
    assert df.iloc[0, 0] == "Front End Web Developer"


def test_load_json_from_url():
    tags_url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/tags.json"
    tags_dict = {}
    for item in utils.load_json_from_url(url=tags_url):
        key = item.pop("tag")
        tags_dict[key] = item
    assert "natural-language-processing" in tags_dict


def test_load_frames():
    filepath = Path(config.BASE_DIR, "tests", "code", "devs_maks.csv")
    df = utils.load_frames(filepath)
    assert type(df) is pd.DataFrame
    assert df.iloc[0, 0] == "Role Summary"


def test_save_and_load_dict():
    with tempfile.TemporaryDirectory() as dp:
        d = {"hello": "world"}
        fp = Path(dp, "d.json")
        utils.save_dict(d=d, filepath=fp)
        d = utils.load_dict(filepath=fp)
        assert d["hello"] == "world"


def test_set_seed():
    utils.set_seeds()
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    utils.set_seeds()
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    assert np.array_equal(a, x)
    assert np.array_equal(b, y)
