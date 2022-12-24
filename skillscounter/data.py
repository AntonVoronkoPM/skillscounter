import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.

    Returns:
        pd.DataFrame: Dataframe with preprocessed data.
    """
    df["text_tag"] = df.text + " " + df.tag  # feature engineering
    return df
