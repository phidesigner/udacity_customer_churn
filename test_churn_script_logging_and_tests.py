# Make sure to replace 'your_module' with the actual name of your module
from churn_library import import_data, perform_eda
import logging
import churn_library as cls
import yaml
from pathlib import Path
import os
import pandas as pd
import pytest


with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


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
    open(EMPTY_CSV_PATH, 'a').close()
    yield
    os.remove(EMPTY_CSV_PATH)


@ pytest.fixture(scope="module")
def setup_invalid_file():
    """
    Fixture to create a non-CSV file before tests and remove it afterwards.
    """
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
    Test handling of an empty CSV file. Ensures an empty DataFrame is returned.
    """
    df = import_data(EMPTY_CSV_PATH)
    assert df.empty, "Expected an empty DataFrame for an empty csv file"


def test_import_data_invalid_file(setup_invalid_file):
    """
    Test handling of a file that is not a CSV.
    """
    with pytest.raises(pd.errors.ParserError):
        import_data(INVALID_FILE_PATH)

# Test cases for EDA #


@ pytest.fixture(scope="module")
def df_sample():
    """Fixture to create a sample DataFrame similar to what perform_eda expects."""
    # Creating a minimal DataFrame that includes necessary columns for perform_eda
    data = {'Attrition_Flag': ['Existing Customer', 'Attrited Customer', 'Existing Customer'],
            'Customer_Age': [50, 40, 30],
            'Marital_Status': ['Married', 'Single', 'Divorced'],
            'Total_Trans_Ct': [50, 45, 30]}
    df = pd.DataFrame(data)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


@ pytest.mark.skip(reason="not yet implemented")
def test_perform_eda(df_sample):
    """Test the perform_eda function to ensure plots are saved as expected."""

    try:
        perform_eda(df_sample, test_config)
        expected_files = [
            'churn_histogram.png',
            'customer_age_histogram.png',
            'marital_status_distribution.png',
            'total_trans_ct_distribution.png',
            'correlation_heatmap.png'
        ]

        # Check all expected files are created in the temporary directory
        for file_name in expected_files:
            assert Path(test_config['EDA']['path'], file_name).exists(), f"{
                file_name} was not created by perform_eda"

    except Exception as e:
        pytest.fail(f"Test failed due to an unexpected exception: {e}")


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
