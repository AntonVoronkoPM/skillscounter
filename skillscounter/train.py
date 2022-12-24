from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from config import config
from skillscounter import data, utils


def objective(params):
    with mlflow.start_run():

        projects_fp = Path(config.DATA_DIR, "full_dataset.csv")
        df = utils.load_frames(filepath=projects_fp)
        df = data.preprocess(df)

        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=5,
            norm="l2",
            encoding="latin-1",
            ngram_range=(1, 2),
            stop_words="english",
        )

        classifier_type = params["type"]
        del params["type"]

        for feature in ({"title": "text"}, {"title": "text_tag"}):
            mlflow.set_tag("feature", feature["title"])
            mlflow.log_params(params)

            # Tf-idf
            X_train = vectorizer.fit_transform(df[feature["title"]].to_numpy())
            y_train = df.target

            if classifier_type == "svm":
                clf = LinearSVC(**params)
            elif classifier_type == "logreg":
                clf = LogisticRegression(**params)
            else:
                return 0
            accuracy = cross_val_score(clf, X_train, y_train).mean()
            mlflow.log_metric("accuracy", accuracy)

            mlflow.log_param("model", type(clf).__name__)
            mlflow.sklearn.log_model(clf, "model")

            # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
        return {"loss": accuracy, "status": STATUS_OK}


def train(args: Dict, df: pd.DataFrame) -> Dict:
    """Train model on data.

    Args:
        args (Dict): arguments to use for training.
        df (pd.DataFrame): data for training.

    Returns:
        Dict: artifacts from the run.
    """

    # Setup
    utils.set_seeds()
    df = data.preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(df["text_tag"].to_numpy(), df.target)

    # Model
    exec("mod = " + args.__dict__.pop("model"), globals())
    args.__dict__.pop("accuracy")

    model = make_pipeline(
        TfidfVectorizer(
            sublinear_tf=True,
            min_df=5,
            norm="l2",
            encoding="latin-1",
            ngram_range=(1, 2),
            stop_words="english",
        ),
        mod(**args.__dict__),
    )

    # Training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}

    return {"args": args, "model": model, "performance": performance}
