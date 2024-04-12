
### README.md for the Predict Customer Churn Project

---

# Predict Customer Churn

This repository contains the machine learning project "Predict Customer Churn" which is part of the ML DevOps Engineer Nanodegree by Udacity. The project is focused on building a model to predict customer churn based on bank data.

## Project Description

This project implements a machine learning workflow including data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation to predict whether a customer will churn. The model could help the bank in taking proactive measures to retain clients at risk of churn, improving customer service, and enhancing customer satisfaction.

## Files and Data Description

The project repository includes various files organized as follows:

- `churn_library.py`: Main Python script containing functions for data loading, preprocessing, training, and evaluation.
- `churn_notebook.ipynb`: Jupyter Notebook with a detailed step-by-step analysis.
- `config.yaml`: Configuration file storing paths and parameters for processing and modeling.
- `constants.py`: Python script with constants used across the project.
- `environment.yml`: Conda environment file for setting up the Python environment.
- `Guide.ipynb`: A notebook guide explaining the project structure and execution.
- `LICENSE`: License file.
- `README.md`: Markdown file providing an overview of the project (this file).
- `test_churn_script_logging_and_tests.py`: Script for testing the main functions in `churn_library.py`.

### Directories

- `/.pytest_cache`: Contains cache data for pytest.
- `/data`: Contains the datasets `bank_data.csv` and `test_bank_data.csv` used for training and testing.
- `/images`: Contains subdirectories for EDA and model results:
  - `/eda`: Stores plots for exploratory data analysis.
  - `/results`: Stores plots for model results like feature importance and classification reports.
- `/logs`: Contains log files for debugging and tracking the workflow execution.
- `/models`: Contains serialized model files for logistic regression and random forest classifiers.
- `/__pycache__`: Contains Python cache files for improved loading speeds.

## Project Structure
------------
    |   churn_notebook.ipynb
    |   config.yaml
    |   constants.py
    |   environment.yml
    |   Guide.ipynb
    |   LICENSE
    |   README.md
    |   test_churn_script_logging_and_tests.py
    |
    +---.pytest_cache
    |   |   .gitignore
    |   |   CACHEDIR.TAG
    |   |   README.md
    |   |
    |   \---v
    |       \---cache
    |               lastfailed
    |               nodeids
    |               stepwise
    |
    +---data
    |       bank_data.csv
    |       test_bank_data.csv
    |
    +---images
    |   +---eda
    |   |       churn_histogram.png
    |   |       correlation_heatmap.png
    |   |       customer_age_histogram.png
    |   |       marital_status_distribution.png
    |   |       total_trans_ct_distribution.png
    |   |
    |   \---results
    |           rf_classification_report.png
    |           rf_feature_importance.png
    |
    +---logs
    |       churn_library_main.log
    |       churn_library_test.log
    |
    +---models
    |       logistic_model.pkl
    |       rfc_model.pkl
    |
    \---__pycache__
            churn_library.cpython-312-pytest-8.1.1.pyc
            churn_library.cpython-312.pyc
            test_churn_script_logging_and_tests.cpython-312-pytest-8.1.1.pyc 
--------

## Running Files

To run the project, execute the `churn_library.py` script. Ensure that all dependencies specified in `environment.yml` are installed in your Python environment. Here's a high-level overview of what happens when you run the script:

1. **Data Loading**: Data is loaded from the `bank_data.csv` file.
2. **Data Preprocessing**: Necessary preprocessing steps are performed, including creating a target variable for churn.
3. **Exploratory Data Analysis (EDA)**: Generate and save plots to visually analyze data trends and relationships.
4. **Feature Engineering**: Engineer features to improve model performance and split the data into training and testing sets.
5. **Model Training**: Train logistic regression and random forest models, optimize hyperparameters, and evaluate them.
6. **Results**: Save classification reports and feature importance plots.

### How to Execute

You can run the project by navigating to the project directory and running:

```bash
python churn_library.py
```

Ensure that the Python environment is activated and all dependencies are installed. For detailed instructions, refer to `Guide.ipynb`.

---

For further assistance on how to use and navigate this project, please refer to the `Guide.ipynb` notebook within this repository.

---
