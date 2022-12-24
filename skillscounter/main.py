import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import joblib
import mlflow
import numpy as np
import typer
from hyperopt import Trials, fmin, hp, tpe
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from config import config
from config.config import logger
from skillscounter import train, utils

warnings.filterwarnings("ignore")

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def predict(text, run_id: str = None) -> List:
    """Predict tag for text.

    Args:
        text (List or str): input text to predict label for.
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.
    """
    if not run_id:
        run_id = open(Path("run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)

    logger.info(text)

    if type(text) == str:
        text = [text]

    logger.info(type(text))

    y_pred = artifacts["model"].predict(text)
    predictions = [
        {
            "input_text": text,
            "predicted_tag": y_pred[0],
        }
    ]

    logger.info(predictions)
    return predictions


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    test_run: bool = False,
) -> None:
    """Train a model given arguments.

    Args:
        args_fp (str): location of args.
        experiment_name (str): name of experiment.
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
    # Load labeled data
    projects_fp = Path(config.DATA_DIR, "full_dataset.csv")
    df = utils.load_frames(filepath=projects_fp)

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["precision"]})
        mlflow.log_metrics({"recall": performance["recall"]})
        mlflow.log_metrics({"f1": performance["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        # open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        open(Path("run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path("performance.json"))


@app.command()
def optimize(
    args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 50
) -> None:
    """Optimize hyperparameters.

    Args:
        args_fp (str): location of args.
        study_name (str): name of optimization study.
        num_trials (int): number of trials to run in study.
    """

    # Optimize
    mlflow.set_experiment(study_name)
    search_space = hp.choice(
        "classifier_type",
        [
            {
                "type": "svm",
                "C": hp.lognormal("SVM_C", 0, 1.0),
            },
            {
                "type": "logreg",
                "C": hp.lognormal("LR_C", 0, 1.0),
                "solver": hp.choice("solver", ["liblinear", "lbfgs"]),
            },
        ],
    )

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=train.objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )

    # Best trial
    client = MlflowClient()

    experiment = client.get_experiment_by_name(study_name)
    run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="metrics.accuracy > .8",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.accuracy DESC"],
    )[0]

    # register the best model
    # mlflow.register_model( ... )

    if "C" in run.data.params:
        run.data.params["C"] = float(run.data.params["C"])

    utils.save_dict({**run.data.params, **run.data.metrics}, args_fp)
    logger.info(f"\nBest value (accuracy): {run.data.metrics['accuracy']:.4f}")
    logger.info(f"Best hyperparameters: {json.dumps(run.data.params, indent=2)}")


@app.command()
def load_artifacts(run_id: str = None) -> Dict:
    """Load artifacts for a given run_id.

    Args:
        run_id (str): id of run to load artifacts from.

    Returns:
        Dict: run's artifacts.
    """
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path("config/args.json")))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path("config/performance.json"))

    return {"args": args, "model": model, "performance": performance}


# def classifier():
#     # Config
#     vac = {
#         "database": "sm-web",
#         "collection": "vacancies",
#         "filter": {"analyzed": False},
#         "projection": {},
#     }

#     vac_db = MongoAPI(vac)
#     new_vacancies = vac_db.read()

#     if len(new_vacancies) == 0:
#         return {"Warning": "Nothing to analyze"}

#     new_vacancies_id = []

#     for i in new_vacancies:
#         new_vacancies_id.append(str(i["_id"]))

#     # Config ?
#     jobstr = {
#         "database": "sm-web",
#         "collection": "jobstrings",
#         "filter": {"vacancyId": {"$in": new_vacancies_id}},
#         "projection": {"tag": 1, "text": 1},
#     }

#     jobstr_db = MongoAPI(jobstr)
#     new_jobstr = jobstr_db.read()

#     # if len(new_jobstr) == 0:
#     #   for i in new_vacancies:
#     #     data = {'filter': {'_id': i['_id']}, 'updated_data': {'$set': {'analyzed': True}}}
#     #     vac_db.update(data)
#     #   return {"Status": "Analyzed status was updated"}

#     targets = prediction(new_jobstr)

#     res = []

#     for i in range(len(new_jobstr)):
#         new_jobstr[i]["target"] = int(targets[i])
#         data = {
#             "filter": {"_id": new_jobstr[i]["_id"]},
#             "updated_data": {"$set": {"target": new_jobstr[i]["target"]}},
#         }
#         res.append(jobstr_db.update(data))

#     res_analyze = []
#     if res.count("Nothing was updated") == 0:
#         for i in new_vacancies:
#             data = {"filter": {"_id": i["_id"]}, "updated_data": {"$set": {"analyzed": True}}}
#             res_analyze.append(vac_db.update(data))
#     else:
#         return {"Warning": "Nothing was updated"}

#     if res_analyze.count("Nothing was updated") == 0:
#         return {"Status": "Targets set successfully"}
#     else:
#         return {"Error": "Analyzed status wasn't updated"}


if __name__ == "__main__":
    app()  # pragma: no cover, live app
