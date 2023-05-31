import os
import unittest

import pandas as pd
from src import ingest_data


class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        ingest_data.args = ingest_data.parse_args(
            [
                "--data_dir",
                str(
                    os.path.join("mle-training", "data", "datasets", "housing")
                ),
                "--log_dir",
                str(
                    os.path.join(
                        "mle-training",
                        "logs",
                        "housing_value_prediction",
                        "ingest data",
                    )
                ),
                "--log_level",
                "INFO",
            ]
        )
        ingest_data.logger = ingest_data.create_logger(
            ingest_data.args.log_dir, ingest_data.args.log_level
        )
        DOWNLOAD_ROOT = (
            "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        )
        self.HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def test_parse_args(self):
        args = ingest_data.parse_args(
            [
                "--data_dir",
                str(
                    os.path.join("mle-training", "data", "datasets", "housing")
                ),
            ]
        )
        self.assertTrue(os.path.exists(args.data_dir))
        # self.assertTrue(os.path.exists(log_dir))

    def test_fetch_housing_data(self):
        df = ingest_data.fetch_housing_data(self.HOUSING_URL)
        # Test for return type of fetch_housing_data
        assert type(df) == pd.core.frame.DataFrame

    def test_split_data(self):
        df = ingest_data.fetch_housing_data(self.HOUSING_URL)
        split_df = ingest_data.split_data(df)
        # Test for return type of load_housing_data
        assert type(split_df) == pd.core.frame.DataFrame
        self.assertTrue(len(split_df) < len(df))

    def test_process_data(self):
        df = ingest_data.fetch_housing_data(self.HOUSING_URL)
        split_df = ingest_data.split_data(df)
        prep_df = ingest_data.process_data(split_df)
        # Test for return type of load_housing_data
        assert type(prep_df) == pd.core.frame.DataFrame
        self.assertTrue(prep_df.shape[1] > df.shape[1])

    def test_store_dataset(self):
        df = ingest_data.fetch_housing_data(self.HOUSING_URL)
        split_df = ingest_data.split_data(df)
        prep_df = ingest_data.process_data(split_df)
        ingest_data.store_dataset(
            prep_df,
            split_df,
            ingest_data.args.data_dir + "/processed",
            "/training_set.csv",
        )
        # Test for return type of load_housing_data
        self.assertTrue(
            os.path.exists(
                ingest_data.args.data_dir + "/processed" + "/training_set.csv"
            )
        )


if __name__ == "__main__":
    unittest.main()
