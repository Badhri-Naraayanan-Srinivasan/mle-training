import argparse
import os
import pickle
import sys
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.abspath(".."))
from src.log_config import configure_logger


def parse_args():
    """Gets Command line arguments

    Returns
    -------
    tuple
        - Directory Path to Read raw data
        - Directory Path to retrieve model files
        - Directory Path to save logs
        - Flag to write Logs

    """
    # Default Data path
    DATA_PATH = os.path.join(
        "..", "data", "datasets", "housing", "processed", "testing_set.csv"
    )
    # Default Model path
    ARTIFACT_PATH = os.path.join("..", "artifacts")
    # Default Log path
    LOG_PATH = os.path.join("..", "logs", "housing_value_prediction", "score")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        nargs="?",
        help="Enter Data Directory Path to retrieve files",
        default=DATA_PATH,
        type=str,
    )
    parser.add_argument(
        "--artifact_dir",
        nargs="?",
        help="Enter Model Directory Path to retrieve model files",
        default=ARTIFACT_PATH,
        type=str,
    )
    parser.add_argument(
        "--log_dir",
        nargs="?",
        help="Enter Directory Path to save logs",
        default=LOG_PATH,
        type=str,
    )
    parser.add_argument(
        "--log_level",
        nargs="?",
        help="Enter Log level: 'DEBUG' to write logs, default: 'INFO'",
        default="INFO",
        type=str,
    )
    parser.add_argument(
        "--mlflow-run_id",
        default=False,
        help="specify the run_id for the run, if you want to save the file in mlflow",
    )
    path = parser.parse_args()

    return path


def create_logger(logs_folder, log_level):
    """Creates a custom logger.

    Parameters
    ----------
    logs_folder : str
        The path to store the logs.
    logs_level : str, default : 'INFO'
        'DEBUG' to store logs.

    Returns
    -------
    logging.Logger
        A custom logger.
    """
    os.makedirs(logs_folder, exist_ok=True)
    # Configure logger
    logger = configure_logger(
        log_file=logs_folder
        + "/score"
        + " ("
        + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        + ").log",
        log_level=log_level,
    )
    return logger


def get_prepared_test_data():
    """Fetches preprocess data

    Returns
    -------
    - DataFrame
        Test DV dataframe
    - pd.Series
        Test IDV series

    """
    # Read raw test data
    test_data = pd.read_csv(args.data_dir).drop(columns="Unnamed: 0")
    logger.debug("Testing Data Fetching : Complete")
    target = "median_house_value"

    return test_data.drop(columns=target), test_data[target]


def score_models(X_test, y_test):
    """Scores Linear Regression. Decision Trees and Random Forest models

    Parameters
    ----------
    X_test : DataFrame
        Preprocessed DV Dataframe for scoring the model
    y_test : pd.Series
        IDV series for scoring the model
    """
    # Read Linear Regression model
    lr_model = pickle.load(
        open(
            args.artifact_dir
            + "/LINEAR_REGRESSION_{DATA_VERSION}.pkl".format(**master_cfg),
            "rb",
        )
    )
    logger.debug("Linear Regression Model Loading : Complete")
    # Predict for test data
    lr_predictions = lr_model.predict(X_test)
    # Calculating Mean Square Error
    lr_mse = mean_squared_error(y_test, lr_predictions)
    # Calculating Root Mean Square Error
    lr_rmse = np.sqrt(lr_mse)
    lr_mae = mean_absolute_error(y_test, lr_predictions)
    logger.debug(
        "RMSE on test set using Linear Regression is %s",
        round(np.sqrt(lr_rmse), 1),
    )
    logger.debug(
        "MAE on test set using Linear Regression is %s",
        round(np.sqrt(lr_mae), 1),
    )
    # Read Decision Tree model
    dt_model = pickle.load(
        open(
            args.artifact_dir
            + "/DECISION_TREE_{DATA_VERSION}.pkl".format(**master_cfg),
            "rb",
        )
    )
    logger.debug("Decision Tree Model Loading : Complete")
    # Predict for test data
    dt_predictions = dt_model.predict(X_test)
    # Calculating Mean Square Error
    dt_mse = mean_squared_error(y_test, dt_predictions)
    # Calculating Root Mean Square Error
    dt_rmse = np.sqrt(dt_mse)
    logger.debug(
        "RMSE on test set using Decision Tree is %s",
        round(np.sqrt(dt_rmse), 1),
    )
    # Read Random Forest model
    rf_model = pickle.load(
        open(
            args.artifact_dir
            + "/RANDOM_FOREST_{DATA_VERSION}.pkl".format(**master_cfg),
            "rb",
        )
    )
    logger.debug("Random Forest Model Loading : Complete")
    # Predict for test data
    rf_predictions = rf_model.predict(X_test)
    # Calculating Mean Square Error
    rf_mse = mean_squared_error(y_test, rf_predictions)
    # Calculating Root Mean Square Error
    rf_rmse = np.sqrt(rf_mse)
    logger.debug(
        "RMSE on test set using Random Forest is %s",
        round(np.sqrt(rf_rmse), 1),
    )

    if master_cfg["MODELLING"]["MODEL_NAME"] == "LINEAR_REGRESSION":
        mse = lr_mse
        rmse = lr_rmse

    if master_cfg["MODELLING"]["MODEL_NAME"] == "DECISION_TREE":
        mse = dt_mse
        rmse = dt_rmse

    if master_cfg["MODELLING"]["MODEL_NAME"] == "RANDOM_FOREST":
        mse = rf_mse
        rmse = rf_rmse

    if args.mlflow_run_id:
        with mlflow.start_run(run_id=args.mlflow_run_id) as run:
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
        mlflow.end_run()


def main():
    # Get prepared test data
    X_test_prepared, y_test = get_prepared_test_data()
    logger.debug("Testing Data Preparation : Complete")
    # Score models
    score_models(X_test_prepared, y_test)
    logger.debug("Code : score.py Execution is Completed")


if __name__ == "__main__":
    args = parse_args()
    logger = create_logger(args.log_dir, args.log_level)
    config_path = "./config.yml"
    with open(config_path, "r") as file:
        master_cfg = yaml.full_load(file)
    main()
