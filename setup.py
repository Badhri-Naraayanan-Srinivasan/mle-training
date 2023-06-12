from setuptools import setup

setup(
    name="mle_training",
    version="0.1.0",
    description="Housing Value Prediction project.",
    author="Badhri Naraayanan Srinivasan",
    author_email="badhri.srinivasa@tigeranalytics.com",
    packages=["housing_value_prediction"],  # same as name
    install_requires=[
        "requests",
        'importlib-metadata; python_version == "3.11"',
    ],  # external packages as dependencies
    package_dir={"housing_value_prediction": "./"},
)
