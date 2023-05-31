import argparse
import os
import pickle
import sys
import tarfile
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.abspath(".."))
from src.log_config import configure_logger


def parse_args(arg):
    """Gets Command line arguments

    Returns
    -------
    tuple
        - Directory Path to store raw data
        - Directory Path to save logs
        - Flag to write Logs

    """
    # Default Data path
    HOUSING_PATH = os.path.join("..", "data", "datasets", "housing")
    # Default Log path
    LOG_PATH = os.path.join(
        "..", "logs", "housing_value_prediction", "ingest data"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        nargs="?",
        help="Enter Directory Path to store files",
        default=HOUSING_PATH,
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
    args = parser.parse_args(arg)
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
    # Configure the logger
    logger = configure_logger(
        log_file=logs_folder
        + "/ingest data"
        + " ("
        + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        + ").log",
        log_level=log_level,
    )
    print(logger)
    return logger


def fetch_housing_data(housing_url):
    """Fetches the raw data

    Parameters
    ----------
    housing_url : str
        URL string of the raw data

    Returns
    -------
    DataFrame
        Raw DataFrame to be processed
    """
    os.makedirs(args.data_dir + "/raw", exist_ok=True)
    # Set path for reading tgz file
    tgz_path = os.path.join(args.data_dir, "raw", "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    # Extract tgz file
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=args.data_dir + "/raw")
    housing_tgz.close()

    return pd.read_csv(os.path.join(args.data_dir, "raw", "housing.csv"))


def store_dataset(housing, strat_data, path, file_name):
    """Stores Output DataFrame to user defined path

    Parameters
    ----------
    housing : DataFrame
        DataFrame to be stored
    strat_data : DataFrame
        Stratified dataframe
    path : str
        Path for the file directory
    file_name : str
        name of the csv file
    """
    os.makedirs(path, exist_ok=True)
    housing.loc[strat_data.index].to_csv(path + file_name)


def split_data(housing):
    """
    Splits train and test data
    Stores Test data


    Parameters
    ----------
    housing : DataFrame
        Dataframe to be split

    Returns
    -------
    DataFrame
        Stratified split train data
    """
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # Split the data to train and test
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set, housing):
        set_.drop("income_cat", axis=1, inplace=True)

    # Store raw test dataset
    store_dataset(
        housing,
        strat_test_set,
        args.data_dir + "/raw",
        "/testing_set.csv",
    )
    return strat_train_set


def process_data(housing):
    """Preprocess the raw data

    Parameters
    ----------
    housing : DataFrame
        Raw Dataframe to be processed

    Returns
    -------
    DataFrame
        PReprocessed Dataframe
    """
    # Create simple imputer object
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    os.makedirs(args.data_dir, exist_ok=True)
    # Save imputer fit object
    with open(args.data_dir + "/imputer.pickle", "wb") as file:
        pickle.dump(imputer, file)

    X = imputer.transform(housing_num)
    logger.debug("Training Data Numerical Impute : Completed")
    logger.debug("Training Data Imputer pickle Storing : Completed")
    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    # Create new features for the model
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )
    logger.debug("Training Data Preprocessing : Completed")
    return housing_prepared


def main():
    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    )
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    # Fetch raw data
    housing = fetch_housing_data(housing_url=HOUSING_URL)
    print(housing.head())
    logger.debug("Raw data Fetching : Completed")
    # Split raw data
    housing = split_data(housing)

    # Preprocess raw train data
    housing_prepared = process_data(housing)
    # Store processed train data
    store_dataset(
        housing_prepared,
        housing,
        args.data_dir + "/processed",
        "/training_set.csv",
    )
    logger.debug("Training Data Storing : Completed")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    logger = create_logger(args.log_dir, args.log_level)
    main()
