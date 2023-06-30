import os
import sysconfig

import config
import mlflow
import mlflow.sklearn


def main():
    # config_path = "config.yml"
    # with open(config_path, "r") as file:
    #     master_cfg = yaml.full_load(file)

    # for key_ in master_cfg:
    #     try:
    #         key_, value_ = key_, master_cfg[key_].format(**master_cfg)
    #         master_cfg[key_] = value_
    #     except Exception as e:
    #         type(e)  # to avoid flake8 error
    #         key_, value_ = key_, master_cfg[key_]

    remote_server_uri = config.MLFLOW["REMOTE_SERVER_URI"]
    mlflow.set_tracking_uri(remote_server_uri)

    exp_name = config.MLFLOW["EXPERIMENT_NAME"]
    mlflow.set_experiment(exp_name)

    print(mlflow.get_tracking_uri())

    with mlflow.start_run(run_name="parent run") as parent_run:
        mlflow.log_param("parent", "yes")
        print("parent run_id: {}".format(parent_run.info.run_id))
        # Start the child runs
        with mlflow.start_run(
            run_name="ingest_data", nested=True
        ) as child_run_1:
            print(
                "mlflow id for data ingestion: {}".format(
                    child_run_1.info.run_id
                )
            )
            os.system(
                "python "
                + sysconfig.get_paths()["purelib"]
                + f"/src/ingest_data.py --mlflow-run_id={child_run_1.info.run_id}"
            )

        with mlflow.start_run(
            run_name="model training", nested=True
        ) as child_run_2:
            print(
                "mlflow id for model training: {}".format(
                    child_run_2.info.run_id
                )
            )
            os.system(
                f"python "
                + sysconfig.get_paths()["purelib"]
                + f"/src/train.py --mlflow-run_id={child_run_2.info.run_id}"
            )

        with mlflow.start_run(
            run_name="model scoring", nested=True
        ) as child_run_3:
            print(
                "mlflow id for model scoring: {}".format(
                    child_run_3.info.run_id
                )
            )
            os.system(
                "python "
                + sysconfig.get_paths()["purelib"]
                + f"/src/score.py --mlflow-run_id={child_run_3.info.run_id}"
            )

    mlflow.end_run()


if __name__ == "__main__":
    main()
