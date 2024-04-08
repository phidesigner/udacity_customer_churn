# Make sure to replace 'your_module' with the actual name of your module
from churn_library import import_data, perform_eda
import logging
import churn_library as cls
import yaml
from pathlib import Path
import os
import pandas as pd
import pytest
import churn_library as cls

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename=config['logging']['filename_test'],
    filemode='w',
    format=config['logging']['format'],
    level=config['logging']['level_test']
)


# Test cases for Data Ingestion
TEST_CSV_PATH = "./data/test_bank_data.csv"
EMPTY_CSV_PATH = "./data/empty_test_data.csv"
NON_EXISTENT_FILE = "./data/non_existent_file.csv"
INVALID_FILE_PATH = "./data/test_not_csv.txt"


@ pytest.fixture(scope="module")
def setup_empty_csv():
    """
    Fixture to create an empty CSV file before tests run and remove
    it afterwards.
    """
    logging.info("Creating an empty CSV file for testing.")
    open(EMPTY_CSV_PATH, 'a').close()
    yield
    os.remove(EMPTY_CSV_PATH)
    logging.info("Empty CSV file removed after testing.")


@ pytest.fixture(scope="module")
def setup_invalid_file():
    """
    Fixture to create a non-CSV file before tests and remove it afterwards.
    """
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


def test_import_data_success():
    """
    Test that the function returns a DataFrame from a valid CSV file.
    """
    logging.info("Testing import_data with a valid CSV file.")
    df = cls.import_data(TEST_CSV_PATH)
    assert isinstance(
        df, pd.DataFrame), "The function did not return a DataFrame."
    assert not df.empty, "DataFrame should not be empty"
    logging.info("import_data successfully returned a non-empty DataFrame.")


def test_import_data_file_not_found():
    """
    Test handling of a non-existent file.
    """
    logging.info("Testing import_data with a non-existent file path.")
    with pytest.raises(FileNotFoundError):
        cls.import_data(NON_EXISTENT_FILE)
    logging.info("import_data correctly raised FileNotFoundError.")


def test_import_data_empty_csv(setup_empty_csv):
    """
    Test handling of an empty CSV file. Ensures an empty DataFrame is returned.
    """
    logging.info("Testing import_data with an empty CSV file.")
    df = cls.import_data(EMPTY_CSV_PATH)
    assert df.empty, "Expected an empty DataFrame for an empty csv file"
    logging.info(
        "import_data correctly returned an empty DataFrame for an empty CSV file.")


def test_import_data_invalid_file(setup_invalid_file):
    """
    Test handling of a file that is not a CSV.
    """
    logging.info("Testing import_data with a malformed non-CSV file.")
    with pytest.raises(pd.errors.ParserError):
        cls.import_data(INVALID_FILE_PATH)
    logging.info(
        "import_data correctly raised ParserError for a malformed file.")

# Test cases for EDA


def test_perform_eda(tmpdir):
    """
    Test the perform_eda function to ensure plots are saved as expected.
    """
    logging.info("Testing perform_eda for expected plot output.")
    df = cls.import_data(cls.config['data']['csv_path'])
    original_eda_path = cls.config['EDA']['path']
    # Redirect EDA output to temp directory
    cls.config['EDA']['path'] = str(tmpdir)

    cls.perform_eda(df)

    expected_files = [
        'churn_histogram.png',
        'customer_age_histogram.png',
        'marital_status_distribution.png',
        'total_trans_ct_distribution.png',
        'correlation_heatmap.png',
    ]
    generated_files = os.listdir(str(tmpdir))
    for expected_file in expected_files:
        assert expected_file in generated_files, f"{
            expected_file} not found in EDA output"
    logging.info("perform_eda successfully generated all expected plots.")

    # Cleanup: Revert config changes to avoid impacting other tests
    cls.config['EDA']['path'] = original_eda_path
    logging.info("Original Config path restablished.")


@ pytest.mark.skip(reason="not yet implemented")
def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


@ pytest.mark.skip(reason="not yet implemented")
def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


@ pytest.mark.skip(reason="not yet implemented")
def test_train_models(train_models):
    '''
    test train_models
    '''
