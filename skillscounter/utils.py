import json
import random
from urllib.request import urlopen

import numpy as np
import pandas as pd

from typing import Dict

from pandas import json_normalize

def json_to_dataframe(req: Dict) -> pd.DataFrame:

    """Function takes json and returns pandas DataFrame

    Args:
        req (Dict): json of the data source.

    Returns:
        pd.DataFrame: loaded pandas frame data.
    """

    first_level_normalizing = json_normalize(req)

    works_data = first_level_normalizing

    return works_data

def load_json_from_url(url: str) -> Dict:
    """Load JSON data from a URL.

    Args:
        url (str): URL of the data source.

    Returns:
        Dict: loaded JSON data.
    """
    data = json.loads(urlopen(url).read())
    return data


def load_frames(filepath: str) -> pd.DataFrame:
    """Load a dictionary from a dataframe's filepath.

    Args:
        filepath (str): location of file.

    Returns:
        pd.DataFrame: loaded pandas dataframe.
    """
    df = pd.read_csv(filepath)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool=False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): data to save.
        filepath (str): location of where to save the data.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_seeds(seed: int = 42) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int, optional): number to be used as the seed. Defaults to 42.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
