from setuptools import setup

setup(
    name="housing_value_prediction_badhri_srinivasan",
    version="1.0.0",
    description="Housing Value Prediction project.",
    author="Badhri Naraayanan Srinivasan",
    author_email="badhri.srinivasa@tigeranalytics.com",
    packages=["src"],  # same as name
    install_requires=[
        "requests",
        'importlib-metadata; python_version == "3.11"',
    ],  # external packages as dependencies
    package_dir={"housing_value_prediction_badhri_srinivasan": "."},
)
