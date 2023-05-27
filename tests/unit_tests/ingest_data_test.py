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
        self.assertTrue(os.path.exists(args.data_dir), msg=str(os.getcwd()))
        # self.assertTrue(os.path.exists(log_dir))

    def test_fetch_housing_data(self):
        df = ingest_data.fetch_housing_data(self.HOUSING_URL)
        # Test for return type of fetch_housing_data
        assert type(df) == pd.core.frame.DataFrame

    # def test_data_prep(self):
    #     df = ingest_data.load_housing_data()
    #     df = ingest_data.data_prep(df)
    #     # Test for return type of load_housing_data
    #     assert type(df) == pd.core.frame.DataFrame

    # def test_split_method_compare(self):
    #     df = ingest_data.load_housing_data()
    #     df = ingest_data.data_prep(df)
    #     dfs = ingest_data.split_method_compare(df)
    #     # Test for return type of load_housing_data
    #     assert len(dfs) == 3

    # def test_rawdata_generated(self):
    #     housing_raw = os.path.join(ingest_data.HOUSING_PATH, "housing.csv")
    #     self.assertTrue(os.path.exists(housing_raw))


if __name__ == "__main__":
    unittest.main()
