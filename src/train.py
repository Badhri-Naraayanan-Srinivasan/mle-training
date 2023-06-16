import argparse
import os
import pickle
import sys
from datetime import datetime

import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.abspath(".."))
from src.log_config import configure_logger


def parse_args():
    """Gets Command line arguments

    Returns
    -------
    tuple
        - Directory Path to Read Processed data
        - Directory Path to save model files
        - Directory Path to save logs
        - Flag to write Logs

    """
    # Default Data path
    DATA_PATH = os.path.join(
        "..", "data", "datasets", "housing", "processed", "training_set.csv"
    )
    # Default Model path
    ARTIFACT_PATH = os.path.join("..", "artifacts")
    # Default Log path
    LOG_PATH = os.path.join("..", "logs", "housing_value_prediction", "train")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        nargs="?",
        help="Enter Data Directory Path to Retrieve files",
        default=DATA_PATH,
        type=str,
    )
    parser.add_argument(
        "--artifact_dir",
        nargs="?",
        help="Enter Model Directory Path to store models",
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
    args = parser.parse_args()

    return args


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
    logger = configure_logger(
        log_file=logs_folder
        + "/train"
        + " ("
        + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        + ").log",
        log_level=log_level,
    )
    return logger


def fit_models(model_data):
    """Fits Linear Regression, Decision Trees and Random Forest models

    Parameters
    ----------
    model_data : DataFrame
        Preprocessed Data for training
    """
    target = "median_house_value"

    # Building Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(model_data.drop(columns=target), model_data[target])
    logger.debug("Linear Regression model : Generated")
    # Building Decision Tree model
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(model_data.drop(columns=target), model_data[target])
    logger.debug("Decision Tree model : Generated")
    # Initialize Parameter grid for Grid search
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    # Build Random forest model from best estimator from grid search
    rf_model = grid_search.fit(
        model_data.drop(columns=target), model_data[target]
    ).best_estimator_
    logger.debug("Random Forest model : Generated")

    # Saving the models with Pickle
    pickle.dump(
        lin_reg,
        open(
            args.artifact_dir
            + "/LINEAR_REGRESSION_{DATA_VERSION}.pkl".format(**master_cfg),
            "wb",
        ),
    )
    pickle.dump(
        tree_reg,
        open(
            args.artifact_dir
            + "/DECISION_TREE_{DATA_VERSION}.pkl".format(**master_cfg),
            "wb",
        ),
    )
    pickle.dump(
        rf_model,
        open(
            args.artifact_dir
            + "/RANDOM_FOREST_{DATA_VERSION}.pkl".format(**master_cfg),
            "wb",
        ),
    )

    logger.info("model dumped succesfully")

    # pdb.set_trace()
    if args.mlflow_run_id:
        with mlflow.start_run(run_id=args.mlflow_run_id) as run:
            mlflow.log_artifact(
                os.path.join(
                    master_cfg["PROCESSED_MODELS_PATH"],
                    "{}_{DATA_VERSION}.pkl".format(
                        master_cfg["MODELLING"]["MODEL_NAME"], **master_cfg
                    ),
                )
            )

        mlflow.end_run()

    logger.debug("Saving model pickles : Complete")


def main():
    model_data = pd.read_csv(args.data_dir).drop(columns="Unnamed: 0")
    logger.debug("Model Data Fetching : Complete")
    fit_models(model_data)


if __name__ == "__main__":
    args = parse_args()
    logger = create_logger(args.log_dir, args.log_level)
    config_path = "./config.yml"
    with open(config_path, "r") as file:
        master_cfg = yaml.full_load(file)
    main()
