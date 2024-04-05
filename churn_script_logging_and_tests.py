# Make sure to replace 'your_module' with the actual name of your module
from churn_library import import_data
import pytest
import pandas as pd
import os
import logging
import churn_library as cls
import yaml

# Load configurations from config.yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Set up logging
logging.basicConfig(
    filename=config['logging']['filename_test'],
    filemode='w',
    format=config['logging']['format'],
    level=config['logging']['level_test']
)


# Assuming the test CSV files are located in the "./data" directory relative to this test script
TEST_CSV_PATH = "./data/test_bank_data.csv"
EMPTY_CSV_PATH = "./data/empty_test_data.csv"
NON_EXISTENT_FILE = "./data/non_existent_file.csv"
INVALID_FILE_PATH = "./data/test_not_csv.txt"


@pytest.fixture(scope="module")
def setup_empty_csv():
    """
    Fixture to create an empty CSV file before tests run and remove it afterwards.
    """
    open(EMPTY_CSV_PATH, 'a').close()
    yield
    os.remove(EMPTY_CSV_PATH)


@pytest.fixture(scope="module")
def setup_invalid_file():
    """
    Fixture to create a non-CSV file before tests and remove it afterwards.
    """
    with open(INVALID_FILE_PATH, 'w') as f:
        f.write("This is not a CSV file.")
    yield
    os.remove(INVALID_FILE_PATH)


def test_import_data_success():
    """
    Test that the function returns a DataFrame from a valid CSV file.
    """
    df = import_data(TEST_CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "DataFrame should not be empty"


def test_import_data_file_not_found():
    """
    Test handling of a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        import_data(NON_EXISTENT_FILE)


def test_import_data_empty_csv(setup_empty_csv):
    """
    Test handling of an empty CSV file.
    """
    df = import_data(EMPTY_CSV_PATH)
    assert isinstance(
        df, pd.DataFrame), "Function should return a DataFrame even if the CSV is empty"
    assert df.empty, "DataFrame should be empty for an empty CSV"


def test_import_data_invalid_file(setup_invalid_file):
    """
    Test handling of a file that is not a CSV.
    """
    with pytest.raises(pd.errors.ParserError):
        import_data(INVALID_FILE_PATH)


@pytest.mark.skip(reason="not yet implemented")
def test_eda(perform_eda):
    '''
    test perform eda function
    '''


@pytest.mark.skip(reason="not yet implemented")
def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


@pytest.mark.skip(reason="not yet implemented")
def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


@pytest.mark.skip(reason="not yet implemented")
def test_train_models(train_models):
    '''
    test train_models
    '''
