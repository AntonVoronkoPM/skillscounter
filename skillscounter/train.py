import json
from argparse import Namespace
from typing import Dict

import mlflow

# import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from config.config import logger
from skillscounter import data, utils
from skillscounter.data import get_data_splits


def train(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial = None) -> Dict:
    """Train model on data.

    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
        trial (optuna.trial._trial.Trial, optional): optimization trial. Defaults to None.

    Raises:
        optuna.TrialPruned: early stopping of trial if it's performing poorly.

    Returns:
        Dict: artifacts from the run.
    """

    # Setup
    utils.set_seeds()
    df = data.preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(
        X=df.feature.to_numpy(), y=df.target
    )

    # Tf-idf
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer, ngram_range=(2, args.ngram_max_range)
    )  # char n-grams
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # Model
    log_clf = LogisticRegression(random_state=0, tol=args.tol, C=args.C, solver=args.solver)
    svm_clf = LinearSVC(loss=args.loss, tol=args.tol, C=args.C)
    model = VotingClassifier(
        estimators=[
            ("lr", log_clf),
            ("svm", svm_clf),
        ],
        voting="hard",
    )

    # Training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(report)
    # logger.info(report[0])

    # Log
    if not trial:
        # mlflow.log_metrics({"classification_report": report})
        mlflow.log_metrics({"precision_0": report["0"]["precision"]})
        mlflow.log_metrics({"precision_1": report["1"]["precision"]})
        mlflow.log_metrics({"recall_0": report["0"]["recall"]})
        mlflow.log_metrics({"recall_1": report["1"]["recall"]})
        mlflow.log_metrics({"f1_0": report["0"]["f1-score"]})
        mlflow.log_metrics({"f1_1": report["1"]["f1-score"]})
        mlflow.log_metrics({"accuracy": report["accuracy"]})

    # Pruning (for optimization in next section)
    if trial:
        # trial.report(report, 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Evaluation
    metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}

    return {"args": args, "vectorizer": vectorizer, "model": model, "performance": performance}


def objective(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial) -> float:
    """Objective function for optimization trials.

    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
        trial (optuna.trial._trial.Trial, optional): optimization trial.

    Returns:
        float: metric value to be used for optimization.
    """
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
    args.tol = trial.suggest_loguniform("tol", 1e-2, 1e0)
    args.C = trial.suggest_uniform("C", 0.1, 10)
    args.solver = trial.suggest_categorical(
        "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    )

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]
    logger.info(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"]
