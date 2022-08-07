from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.

    Returns:
        pd.DataFrame: Dataframe with preprocessed data.
    """
    df["feature"] = df.text + " " + df.tag  # feature engineering
    return df


def get_data_splits(X: pd.Series, y: np.ndarray, train_size: float = 0.7) -> Tuple:
    """Generate balanced data splits.

    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.

    Returns:
        Tuple: data splits as Numpy arrays.
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test
