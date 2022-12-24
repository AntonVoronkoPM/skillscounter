import shutil
from pathlib import Path

import mlflow

# import pytest
from typer.testing import CliRunner

from config import config
from skillscounter.main import app, load_artifacts

runner = CliRunner()
args_fp = Path(config.BASE_DIR, "tests", "code", "test_args.json")


def delete_experiment(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)


def test_train_model():
    experiment_name = "test_experiment"
    result = runner.invoke(
        app,
        [
            "train-model",
            f"--args-fp={args_fp}",
            f"--experiment-name={experiment_name}",
            "--test-run",
        ],
    )
    assert result.exit_code == 0

    # Delete experiment
    delete_experiment(experiment_name=experiment_name)
    shutil.rmtree(Path(config.MODEL_REGISTRY, ".trash"))


def test_optimize():
    study_name = "test_optimization"
    num_trials = 1
    result = runner.invoke(
        app,
        [
            "optimize",
            f"--args-fp={args_fp}",
            f"--study-name={study_name}",
            f"--num-trials={num_trials}",
        ],
    )
    assert result.exit_code == 0

    # Delete study
    delete_experiment(experiment_name=study_name)
    shutil.rmtree(Path(config.MODEL_REGISTRY, ".trash"))


def test_load_artifacts():
    # run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    run_id = open(Path("run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    assert len(artifacts)


# # shutil.rmtree(Path(config.MODEL_REGISTRY, ".trash"))
def test_predict():
    text = "Transfer learning with transformers for text classification."
    result = runner.invoke(app, ["predict", f"{[text]}"])
    assert result.exit_code == 0
