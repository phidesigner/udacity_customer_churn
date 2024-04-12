"""This module contains tests for the churn prediction library
functionality."""

import logging
import os
import yaml
import pandas as pd
import pytest
import churn_library as cls

# Load configuration
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Setup logging as per the new test configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=config['logging']['filename_test'],
                    filemode='w',
                    format=config['logging']['format'],
                    level=config['logging']['level_test'])

# Define paths for data ingestion tests
TEST_CSV_PATH = "./data/test_bank_data.csv"
EMPTY_CSV_PATH = "./data/empty_test_data.csv"
NON_EXISTENT_FILE = "./data/non_existent_file.csv"
INVALID_FILE_PATH = "./data/test_not_csv.txt"


@pytest.fixture(scope="module")
def setup_empty_csv():
    """Create and remove an empty CSV file for testing purposes."""
    logging.info("Creating an empty CSV file for testing.")
    # Using 'with' ensures the file is closed after it is created.
    open(EMPTY_CSV_PATH, 'a', encoding='utf-8').close()
    yield
    os.remove(EMPTY_CSV_PATH)
    logging.info("Empty CSV file removed after testing.")


@pytest.fixture(scope="module")
def setup_invalid_file():
    """Create and remove a malformed non-CSV file for testing purposes."""
    logging.info("Creating a malformed non-CSV file for testing.")
    malformed_content = '''ID,Name,Value
1,"Alice,2
3,"Bob",Unclosed quote
4,Ca"rol,Strange"Character
5,Dave,"Mismatched"Quotes",6
'''
    with open(INVALID_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(malformed_content)
    yield
    os.remove(INVALID_FILE_PATH)
    logging.info("Malformed non-CSV file removed after testing.")


@pytest.fixture(scope="module")
def loaded_test_data():
    """
    Load and prepare test data from 'test_bank_data.csv'.

    This fixture ensures that data used in tests includes the necessary 'Churn'
    column by loading the data and applying the churn library's data
    preparation functions.

    Returns:
        pandas.DataFrame: The test data loaded and processed.    """
    df = cls.import_data(TEST_CSV_PATH)
    df = cls.create_churn_column(df)
    return df


def test_import_data_success():
    """
    Test the import_data function with a valid CSV file.

    This test verifies that the import_data function is able to correctly
    import data from a valid CSV file and returns a non-empty pandas DataFrame.
    """
    logging.info("Testing import_data with a valid CSV file.")
    df = cls.import_data(TEST_CSV_PATH)
    assert isinstance(
        df, pd.DataFrame), "The function did not return a DataFrame."
    assert not df.empty, "DataFrame should not be empty."
    logging.info("import_data successfully returned a non-empty DataFrame.")


def test_import_data_file_not_found():
    """
    Test the import_data function with a non-existent file path.

    This test verifies that the import_data function raises a FileNotFoundError
    when attempting to load data from a path that does not exist. This ensures
    that the function handles file not found errors correctly.
    """
    logging.info("Testing import_data with a non-existent file path.")
    with pytest.raises(FileNotFoundError):
        cls.import_data(NON_EXISTENT_FILE)
    logging.info("import_data correctly raised FileNotFoundError.")


def test_import_data_empty_csv(setup_empty_csv):  # pylint:disable=unused-argument
    """
    Test the import_data function with an empty CSV file.

    This test confirms that the import_data function returns an empty pandas
    DataFrame when given an empty CSV file as input. The setup_empty_csv
    fixture is used to create and later remove an empty CSV file,
    ensuring the environment is correctly prepared for this test.

    Args:
        setup_empty_csv (pytest fixture): Fixture that creates and removes
        an empty CSV file.
    """
    logging.info("Testing import_data with an empty CSV file.")
    df = cls.import_data(EMPTY_CSV_PATH)
    assert df.empty, "Expected an empty DataFrame for an empty CSV file."
    logging.info(
        "import_data correctly returned an empty DataFrame for an empty CSV \
        file.")


def test_import_data_invalid_file(setup_invalid_file):  # pylint: disable=unused-argument
    """
    Test the import_data function with a malformed non-CSV file.

    This test checks if the import_data function raises a pandas errors.
    ParserError when attempting to import data from a malformed non-CSV file.
    The setup_invalid_file fixture prepares the environment by creating a
    non-CSV file with improper formatting, which is used in this test.

    Args:
        setup_invalid_file (pytest fixture): Fixture that creates and removes
        a malformed file.
    """
    logging.info("Testing import_data with a malformed non-CSV file.")
    with pytest.raises(pd.errors.ParserError):
        cls.import_data(INVALID_FILE_PATH)
    logging.info(
        "import_data correctly raised ParserError for a malformed file.")


def test_perform_eda(tmpdir, loaded_test_data):  # pylint:disable=unused-argument
    """
    Test the perform_eda function for expected plot output.

    This test verifies that the perform_eda function generates expected plot
    files in the specified directory. The test utilizes a temporary directory
    (tmpdir) to store output files and checks for their existence to confirm
    the function's performance. The loaded_test_data fixture provides the
    necessary data pre-loaded with a 'Churn' column.

    Args:
        tmpdir (pytest fixture): Temporary directory provided by pytest for
        file output. loaded_test_data (pytest fixture): Data loaded with
        'Churn' column for EDA testing.
    """
    logging.info("Testing perform_eda for expected plot output.")
    original_eda_path = cls.config['EDA']['path']
    cls.config['EDA']['path'] = str(tmpdir)  # Temporary redirect EDA output

    cls.perform_eda(loaded_test_data)

    expected_files = [
        'churn_histogram.png',
        'customer_age_histogram.png',
        'marital_status_distribution.png',
        'total_trans_ct_distribution.png',
        'correlation_heatmap.png'
    ]

    generated_files = os.listdir(str(tmpdir))
    for expected_file in expected_files:
        assert expected_file in generated_files, f"{
            expected_file} not found in EDA output."
    logging.info("perform_eda successfully generated all expected plots.")

    cls.config['EDA']['path'] = original_eda_path  # Revert config changes
    logging.info("Original Config path restablished.")


def test_encoder_helper(loaded_test_data):  # pylint:disable=unused-argument
    """
    Test the encoder_helper function to ensure correct categorical encoding.

    This test verifies that the encoder_helper function from the churn library
    applies the correct categorical encoding to the specified columns of the
    data. It checks if the resulting DataFrame contains new columns for each
    category that indicate the relationship with the 'Churn' variable.

    Args:
        loaded_test_data (pytest fixture): Data loaded with 'Churn' column for
        encoding testing.
    """
    logging.info("Testing encoder_helper function.")
    category_lst = ['Gender', 'Education_Level', 'Marital_Status']
    df_encoded = cls.encoder_helper(loaded_test_data, category_lst)

    assert all(
        f"{x}_Churn" in df_encoded for x in category_lst), "Not all category \
            churn columns are present."
    logging.info("encoder_helper function passed basic existence checks.")


def test_perform_feature_engineering(loaded_test_data):  # pylint:disable=unused-argument
    """
    Test the perform_feature_engineering function for correct data processing.

    Ensures that the perform_feature_engineering function from the churn
    library processes the input data correctly and splits it into train & test
    sets with the expected features. It checks the presence of specific columns
    and non-emptiness of the resulting DataFrames.

    Args:
        loaded_test_data (pytest fixture): Data loaded with 'Churn' column for
        feature engineering.
    """
    logging.info("Testing perform_feature_engineering function.")

    category_lst = config['categories']
    df_encoded = cls.encoder_helper(loaded_test_data, category_lst)

    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        df_encoded, 'Churn')

    assert not x_train.empty and not x_test.empty, "X_train or X_test is empty"
    assert not y_train.empty and not y_test.empty, "y_train or y_test is empty"
    assert set(x_train.columns) == set(
        config['features']['keep_cols']), "X_train doesn't contain the correct\
        columns."
    logging.info("perform_feature_engineering function passed all checks.")


@pytest.mark.parametrize("model_name, model_file", [
    ("Random Forest", "rfc_model.pkl"),
    ("Logistic Regression", "logistic_model.pkl")
])
def test_train_models(loaded_test_data, tmpdir, model_name, model_file):   # pylint:disable=unused-argument
    """
    Test the train_models function to ensure models and outputs are correctly
    ssaved.

    This test evaluates the train_models function by verifying that the models
    are properly trained and saved to the specified directory, along with
    associated output files such as classification reports and feature
    importance plots. The test uses parametrization to check multiple model
    types.

    Args:
        loaded_test_data (pytest fixture): Data preprocessed and loaded with
        'Churn' column.
        tmpdir (pytest fixture): Temporary directory provided
        by pytest for file output.
        model_name (str): Name of the model to test.
        model_file (str): Filename to which the model should be saved.
    """
    logging.info("Testing train_models function with %s.", model_name)

    # Temporary redirect model and results output
    original_model_path = cls.config['models']['path']
    cls.config['models']['path'] = str(tmpdir)
    original_results_path = cls.config['EDA']['results']
    cls.config['EDA']['results'] = str(tmpdir)

    category_lst = cls.config['categories']
    df_encoded = cls.encoder_helper(loaded_test_data, category_lst)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        df_encoded, 'Churn')

    cls.train_models(x_train, x_test, y_train, y_test)

    # Check if model and output files are correctly saved
    model_path = os.path.join(str(tmpdir), model_file)
    assert os.path.exists(model_path), f"{model_name} model file not found."

    expected_files = [
        'rf_classification_report.png',
        'lr_classification_report.png',
        'rf_feature_importance.png'
    ]
    for expected_file in expected_files:
        assert os.path.exists(os.path.join(str(tmpdir), expected_file)), f"{
            expected_file} not found in results."

    for expected_file in expected_files:
        assert os.path.exists(os.path.join(str(tmpdir), expected_file)), f"{
            expected_file} not found in results."

    # Revert config changes
    cls.config['models']['path'] = original_model_path
    cls.config['EDA']['results'] = original_results_path
