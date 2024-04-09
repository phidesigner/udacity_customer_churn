# Test suite script for churn library
import logging
import pandas as pd
import pytest
import churn_library as cls
import yaml
import os

# Load configuration
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

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
    logging.info("Creating an empty CSV file for testing.")
    open(EMPTY_CSV_PATH, 'a').close()
    yield
    os.remove(EMPTY_CSV_PATH)
    logging.info("Empty CSV file removed after testing.")


@pytest.fixture(scope="module")
def setup_invalid_file():
    logging.info("Creating a malformed non-CSV file for testing.")
    malformed_content = '''ID,Name,Value
1,"Alice,2
3,"Bob",Unclosed quote
4,Ca"rol,Strange"Character
5,Dave,"Mismatched"Quotes",6
'''
    with open(INVALID_FILE_PATH, 'w') as f:
        f.write(malformed_content)
    yield
    os.remove(INVALID_FILE_PATH)
    logging.info("Malformed non-CSV file removed after testing.")


@pytest.fixture(scope="module")
def loaded_test_data():
    """
    Fixture to load data from 'test_bank_data.csv' for use in tests.
    """
    df = cls.import_data(TEST_CSV_PATH)
    # Ensure Churn column is created for loaded data
    df = cls.create_churn_column(df)
    return df


def test_import_data_success():
    logging.info("Testing import_data with a valid CSV file.")
    df = cls.import_data(TEST_CSV_PATH)
    assert isinstance(
        df, pd.DataFrame), "The function did not return a DataFrame."
    assert not df.empty, "DataFrame should not be empty."
    logging.info("import_data successfully returned a non-empty DataFrame.")


def test_import_data_file_not_found():
    logging.info("Testing import_data with a non-existent file path.")
    with pytest.raises(FileNotFoundError):
        cls.import_data(NON_EXISTENT_FILE)
    logging.info("import_data correctly raised FileNotFoundError.")


def test_import_data_empty_csv(setup_empty_csv):
    logging.info("Testing import_data with an empty CSV file.")
    df = cls.import_data(EMPTY_CSV_PATH)
    assert df.empty, "Expected an empty DataFrame for an empty CSV file."
    logging.info(
        "import_data correctly returned an empty DataFrame for an empty CSV file.")


def test_import_data_invalid_file(setup_invalid_file):
    logging.info("Testing import_data with a malformed non-CSV file.")
    with pytest.raises(pd.errors.ParserError):
        cls.import_data(INVALID_FILE_PATH)
    logging.info(
        "import_data correctly raised ParserError for a malformed file.")


def test_perform_eda(tmpdir, loaded_test_data):
    """
    Updated to use loaded_test_data with the 'Churn' column already created.
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


def test_encoder_helper(loaded_test_data):
    """
    Tests encoder_helper function to ensure correct categorical encoding.
    """
    logging.info("Testing encoder_helper function.")
    category_lst = ['Gender', 'Education_Level', 'Marital_Status']
    df_encoded = cls.encoder_helper(loaded_test_data, category_lst)

    assert all(
        f"{x}_Churn" in df_encoded for x in category_lst), "Not all category churn columns are present."
    logging.info("encoder_helper function passed basic existence checks.")


def test_perform_feature_engineering(loaded_test_data):
    """
    Tests perform_feature_engineering function for correct data processing.
    """
    logging.info("Testing perform_feature_engineering function.")

    category_lst = config['categories']
    df_encoded = cls.encoder_helper(loaded_test_data, category_lst)

    # Directly use loaded_test_data which now includes 'Churn' column
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df_encoded, 'Churn')

    assert not X_train.empty and not X_test.empty, "X_train or X_test is empty."
    assert not y_train.empty and not y_test.empty, "y_train or y_test is empty."
    assert set(X_train.columns) == set(
        config['features']['keep_cols']), "X_train does not contain the correct columns."
    logging.info("perform_feature_engineering function passed all checks.")


@pytest.mark.skip(reason="not yet implemented")
def test_train_models():
    """
    Placeholder for test_train_models test case.
    """
