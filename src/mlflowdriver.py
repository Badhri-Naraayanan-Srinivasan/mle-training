import os

import mlflow
import mlflow.sklearn
import yaml


def main():
    config_path = "config.yml"
    with open(config_path, "r") as file:
        master_cfg = yaml.full_load(file)

    for key_ in master_cfg:
        try:
            key_, value_ = key_, master_cfg[key_].format(**master_cfg)
            master_cfg[key_] = value_
        except Exception as e:
            type(e)  # to avoid flake8 error
            key_, value_ = key_, master_cfg[key_]

    remote_server_uri = master_cfg["MLFLOW"]["REMOTE_SERVER_URI"]
    mlflow.set_tracking_uri(remote_server_uri)

    exp_name = master_cfg["MLFLOW"]["EXPERIMENT_NAME"]
    mlflow.set_experiment(exp_name)

    print(mlflow.get_tracking_uri())

    with mlflow.start_run(run_name="parent run") as parent_run:
        mlflow.log_param("parent", "yes")
        print("parent run_id: {}".format(parent_run.info.run_id))
        # Start the child runs
        with mlflow.start_run(
            run_name="data_processing", nested=True
        ) as child_run_1:
            print(
                "mlflow id for data processing: {}".format(
                    child_run_1.info.run_id
                )
            )
            os.system(
                f"python ingest_data.py --mlflow-run_id={child_run_1.info.run_id}"
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
                f"python train.py --mlflow-run_id={child_run_2.info.run_id}"
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
                f"python score.py --mlflow-run_id={child_run_3.info.run_id}"
            )

    mlflow.end_run()


if __name__ == "__main__":
    main()
