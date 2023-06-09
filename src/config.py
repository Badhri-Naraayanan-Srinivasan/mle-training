DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "../datasets/housing/raw"
HOUSING_URL = "{DOWNLOAD_ROOT}/datasets/housing/housing.tgz"
DATA_PATH = "/home/badhri/mle_training"
PROCESSED_DATA_PATH = (
    "/home/badhri/mle_training/data/datasets/housing/processed"
)
PROCESSED_MODELS_PATH = "/home/badhri/mle_training/artifacts/"
DATA_VERSION = 1
OVERWRITE = True

TRAIN_TEST_SPLIT = {"SAMPLING_METHOD": "random", "TEST_SIZE": 0.2, "SEED": 42}

IMPUTE = {
    "NUMERICAL_CONSTANT": None,
    "NUMERICAL_STRATEGY": "mean",
    "CATEGORICAL_CONSTANT": None,
    "CATEGORICAL_STRATEGY": "most_frequent",
    "ADD_INDICATOR": False,
}
MODELLING = {
    "MODEL_NAME": "RANDOM_FOREST",
    "PARAMS": {
        "LINEAR_REGRESSION": {"fit_intercept": True},
        "DECISION_TREE": {
            "criterion": "squared_error",
            "max_depth": 4,
            "splitter": "best",
            "max_features": "auto",
            "random_state": 42,
        },
        "RANDOM_FOREST": {
            "n_estimators": 10,
            # criterion: 'squared_error',
            "max_depth": 4,
            "max_features": "auto",
            "random_state": 42,
        },
    },
}
MLFLOW = {
    "LOG_INTO_MLFLOW": True,
    "REMOTE_SERVER_URI": "http://0.0.0.0:5000",
    "EXPERIMENT_NAME": "Test_Housing_",
}
