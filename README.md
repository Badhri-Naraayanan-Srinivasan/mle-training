# Median housing value prediction

# Pre-requisites

* Ensure you have `Miniconda` installed and can be run from your shell. If not, download the installer for your platform here: https://docs.conda.io/en/latest/miniconda.html

     **NOTE**

     * If you already have `Anaconda` installed, pls. still go ahead and install `Miniconda` and use it for development.
     * If `conda` cmd is not in your path, you can configure your shell by running `conda init`.


# Getting started

* [Download the repo from gdrive](https://drive.google.com/file/d/177NngnW6GVhzq6psKj7FX45YEKqJq65K/view?usp=sharing), unzip and switch to the root folder

* Setup a development environment and switch to it by running:
```
(base):~/$ conda env create -f deploy/conda/env.yml
(base):~/$ conda activate mle-dev
```

The above command should create a conda python environment named `mle-dev`

In case of installation issues, remove and recreate. You can remove by running:
```
(base):~/$ conda remove --name mle-dev --all -y
```

# Procedure to run the code

- conda env create -f deploy/conda/env.yml
- conda activate mle-dev
- cd src
- python ingest_data.py
- python train.py
- python score.py