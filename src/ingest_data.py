import argparse
import os
import pickle
import sys
import tarfile
import urllib.request
from datetime import datetime

import config
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted

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
    parser.add_argument(
        "--mlflow-run_id",
        default=False,
        help="specify the run_id for the run, if you want to save the file in mlflow",
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


def store_dataset(housing, path, file_name):
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
    housing.to_csv(path + file_name)


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

    return strat_train_set, strat_test_set


class Imputer(_BaseImputer, TransformerMixin):
    """
    Impute data based on the method passed in the config.

    Parameters
    ----------
        data: pd.DataFrame
            input data frame to be imputed
        num_impute: str, default mean
            numerical imputation method
        cat_impute: str, default mode
            categorical imputation method
        num_constant: numeric
            numerical constant to use when the num_imputer is constant
        cat_constant: str
            categorical constant to use when the cat_imputer is constant
    """

    def __init__(self):
        if config.IMPUTE["NUMERICAL_CONSTANT"] is not None:
            num_impute = "constant"
        else:
            num_impute = config.IMPUTE["NUMERICAL_STRATEGY"]

        if config.IMPUTE["CATEGORICAL_CONSTANT"] is not None:
            cat_impute = "constant"
        else:
            cat_impute = config.IMPUTE["CATEGORICAL_STRATEGY"]

        num_constant = config.IMPUTE["NUMERICAL_CONSTANT"]
        cat_constant = config.IMPUTE["CATEGORICAL_CONSTANT"]
        missing_values = np.nan
        add_indicator = config.IMPUTE["ADD_INDICATOR"]

        super().__init__(
            missing_values=missing_values, add_indicator=add_indicator
        )
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.num_constant = num_constant
        self.cat_impute = cat_impute
        self.cat_constant = cat_constant

    def fit(self, X, y=None):
        check_array(
            X,
            accept_large_sparse=False,
            dtype=object,
            force_all_finite="allow-nan",
        )
        self.dtype_dict_ = X.dtypes
        if self.num_impute == "constant":
            self.num_imputer_ = SimpleImputer(
                strategy=self.num_impute, fill_value=self.num_constant
            )
        else:
            self.num_imputer_ = SimpleImputer(strategy=self.num_impute)

        if self.cat_impute == "constant":
            self.cat_imputer_ = SimpleImputer(
                strategy=self.cat_impute, fill_value=self.cat_constant
            )
        else:
            self.cat_imputer_ = SimpleImputer(strategy=self.cat_impute)
        self.num_cols_ = list(X.select_dtypes(include=np.number).columns)
        self.cat_cols_ = list(X.select_dtypes(exclude=np.number).columns)
        self.imputer_ = ColumnTransformer(
            transformers=[
                ("num", self.num_imputer_, self.num_cols_),
                ("cat", self.cat_imputer_, self.cat_cols_),
            ]
        )
        self.imputer_.fit(X)
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "is_fitted_")
        X = self.imputer_.transform(X)
        X = pd.DataFrame(X, columns=self.num_cols_ + self.cat_cols_)
        X = X.astype(self.dtype_dict_)
        return X

    def impute(
        data,
        num_impute="mean",
        cat_impute="most_frequent",
        num_constant=None,
        cat_constant=None,
    ):
        """
        Impute data based on the method.

        Parameters
        ----------
            data: pd.DataFrame
                input data frame to be imputed
            num_impute: str, default mean
                numerical imputation method
            cat_impute: str, default mode
                categorical imputation method
            num_constant: numeric
                numerical constant to use when the num_imputer is constant
            cat_constant: str
                categorical constant to use when the cat_imputer is constant
        Returns
        -------
            data: pd.DataFrame
                input data frame to be imputed
            imputer: object
                imputer object

        """
        dtype_dict = data.dtypes
        if num_constant is None:
            num_imputer = SimpleImputer(strategy=num_impute)
        else:
            if isinstance(num_constant, int):
                num_imputer = SimpleImputer(
                    strategy="constant", fill_value=num_constant
                )

        if cat_constant is not None:
            cat_imputer = SimpleImputer(
                strategy="constant", fill_value=cat_constant
            )
        else:
            cat_imputer = SimpleImputer(strategy=cat_impute)

        num_cols = list(data.select_dtypes(include=np.number).columns)
        cat_cols = list(data.select_dtypes(exclude=np.number).columns)
        # logger.info("INFO: numerical columns imputation: {}".format(num_cols))
        # logger.info("INFO: cateogrical columns imputation: {}".format(cat_cols))
        imputer = ColumnTransformer(
            transformers=[
                ("num", num_imputer, num_cols),
                ("cat", cat_imputer, cat_cols),
            ]
        )
        imputer.fit(data)
        data = imputer.transform(data)
        data = pd.DataFrame(data, columns=num_cols + cat_cols)
        data = data.astype(dtype_dict)
        return data, imputer


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):  # no *args or **kargs
        return None

    def fit(self, X, y=None):
        self._cols = list(X.columns)
        self._household_ix = self._cols.index("households")
        self._population_ix = self._cols.index("population")
        self._bedrooms_ix = self._cols.index("total_bedrooms")
        self._rooms_ix = self._cols.index("total_rooms")

        self._cols += [
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_room",
        ]
        return self

    def transform(self, X, y=None):
        X = X.values
        rooms_per_household = X[:, self._rooms_ix] / X[:, self._household_ix]
        population_per_household = (
            X[:, self._population_ix] / X[:, self._household_ix]
        )
        bedrooms_per_room = X[:, self._bedrooms_ix] / X[:, self._rooms_ix]
        return pd.DataFrame(
            np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ],
            columns=self._cols,
        )


def process_data(housing, test):
    """Preprocess the raw data

    Parameters
    ----------
    housing : DataFrame
        Raw Dataframe to be processed

    Returns
    -------
    DataFrame
        Preprocessed Dataframe
    """
    cat_cols = list(housing.select_dtypes(exclude=np.number).columns)
    pl = Pipeline(
        [
            (
                "imputer",
                Imputer(),
            ),
            (
                "attribs_adder",
                CombinedAttributesAdder(),
            ),
            (
                "label_endcode",
                ColumnTransformer(
                    transformers=[
                        (
                            "onehot_encoder",
                            OneHotEncoder(sparse=False),
                            cat_cols,
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
        ]
    )
    os.makedirs(
        os.path.join(
            "..",
            "artifacts",
        ),
        exist_ok=True,
    )
    with open(
        os.path.join(
            "..",
            "artifacts",
            "pipeline_{}.pkl".format(config.DATA_VERSION),
        ),
        "wb",
    ) as path:
        pickle.dump(
            pl,
            path,
        )
    housing_prepared = pd.DataFrame(
        pl.fit_transform(housing),
        columns=[
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN",
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_room",
        ],
    )
    test_prepared = pd.DataFrame(
        pl.transform(test),
        columns=[
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN",
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_room",
        ],
    )
    return housing_prepared, test_prepared


def main():
    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    )
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    global args, logger

    args = parse_args(sys.argv[1:])
    logger = create_logger(args.log_dir, args.log_level)
    # config_path = "./config.yml"
    # with open(config_path, "r") as file:
    #     master_cfg = yaml.full_load(file)
    # # master_cfg = yaml.load(config_path)

    # for key_ in master_cfg:
    #     try:
    #         key_, value_ = key_, master_cfg[key_].format(**master_cfg)
    #         master_cfg[key_] = value_
    #     except Exception as e:
    #         type(e)  # to avoid flake8 error
    #         key_, value_ = key_, master_cfg[key_]

    # Fetch raw data
    housing = fetch_housing_data(housing_url=HOUSING_URL)
    logger.debug("Raw data Fetching : Completed")
    # Split raw data
    housing, test = split_data(housing)

    # Preprocess raw train data
    housing_prepared, test_prepared = process_data(housing, test)
    # Store processed train data
    store_dataset(
        housing_prepared,
        args.data_dir + "/processed",
        "/train_{}.csv".format(config.DATA_VERSION),
    )
    # Store raw test dataset
    store_dataset(
        test_prepared,
        args.data_dir + "/processed",
        "/test_{}.csv".format(config.DATA_VERSION),
    )
    print(args.mlflow_run_id)
    # Log in mlflow
    if args.mlflow_run_id:
        print(mlflow.get_tracking_uri())
        with mlflow.start_run(
            run_id=args.mlflow_run_id, run_name="ingest_data"
        ) as run:
            mlflow.log_artifact(
                os.path.join(
                    config.PROCESSED_DATA_PATH,
                    "train_{}.csv".format(config.DATA_VERSION),
                )
            )
            mlflow.log_artifact(
                os.path.join(
                    config.PROCESSED_DATA_PATH,
                    "test_{}.csv".format(config.DATA_VERSION),
                )
            )
            mlflow.log_artifact(
                os.path.join(
                    config.PROCESSED_MODELS_PATH,
                    "pipeline_{}.pkl".format(config.DATA_VERSION),
                )
            )
        print("logged in mlflow")
        mlflow.end_run()

    logger.debug("Training Data Storing : Completed")


if __name__ == "__main__":
    main()
