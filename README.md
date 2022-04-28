# __Production-ready Breast Cancer ML project__

## __Work Description__

`ml_project` directory contains the project with CLI application to manage feature processing, model creation and evaluation. The dataset used is Wisconsin Breast Cancer Dataset, [source](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

The project structure is inspired (to a reasonable extent) by [datascience-cookiecutter template](https://drivendata.github.io/cookiecutter-data-science/). Some sections are not present, like references or sphinx build.

For configuration management of the project, `hydra` framework is utilized ([hydra's docs](https://hydra.cc/)). This expierence was pretty nice for me, as I have never tickled with `yaml` files seriously (except for `docker-compose` ones).

I do not perform much EDA in this section for a number of reasons: I have already worked with this dataset once, and that last time I implemented PCA to prove that the target variable (cancer type of the patient) was linearly separable. All the features are already denoised and distributed without outliers. This time I aimed to design a pipeline capable of multiple optional steps, that is why part for categorical features is present, yet there are no actual categorial features in the dataset.

## __Basic usage__

### __Run the training pipeline with hydra__

In directory `ml_project`, run the following command:

`>>> python3 pipeline.py`

After this, `outputs/` dir should be automatically created by hydra to manage runs.

![outputs dir](./screenshots/outputs.jpg)

__Note__: due to relative pathing, the paths convention is `../../../desired-dir-name`.

__Upd__: now `${dir_prefix}` variable can be used to manage pathing in configs.

### __Configure the pipeline!__

With hydra it is really simple and straightforward, just specify the required configuration
throgh the CLI like `estimator=random-forest` or `++random_state=42` (use `++` to override existing values).

### __Logging__

Loggers are used in most modules and the logfile can be found in hydra's `outputs/` directory as `pipeline.log`.

### __Unit-testing__

`Pytest` framework is used for testing, modular tests can be found at `testing/` directory. I tried to advance
my knowledge of pytest by using `mark.skipif` semantics in order to manage offline runs for data fetching.
Also, I found out that `pytest` has its own `setup` and `teardown` methods. Those were used in order to
create a temporary directory to write/read data during testing. The directory is cleaned up after each run.

In order to run all unit-tests, run the following command (see the example below):

`>>> python -m pytest`

![running network-sensitive tests](./screenshots/network-testing.jpg)
