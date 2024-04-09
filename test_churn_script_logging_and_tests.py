# Make sure to replace 'your_module' with the actual name of your module
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


@pytest.fixture(scope="module")
def sample_dataframe():
    """
    Provides a sample DataFrame for testing purposes.
    """
    test_data = {
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'Education_Level': ['High School', 'Graduate', 'Uneducated', 'High School', 'Graduate'],
        'Marital_Status': ['Married', 'Single', 'Married', 'Single', 'Married'],
        'Attrition_Flag': ['Existing Customer', 'Attrited Customer', 'Existing Customer', 'Attrited Customer', 'Existing Customer']
    }
    df = pd.DataFrame(test_data)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


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


def test_encoder_helper(sample_dataframe):
    """
    Test the encoder_helper function to ensure it correctly adds new columns
    with encoded categorical features based on churn proportion.
    """
    logging.info("Testing encoder_helper function.")

    # Specify the categories to encode
    category_lst = ['Gender', 'Education_Level', 'Marital_Status']

    df_encoded = cls.encoder_helper(sample_dataframe, category_lst)

    # Actual expected churn rates for each category
    expected_gender_churn = sample_dataframe.groupby('Gender')['Churn'].mean()
    expected_education_churn = sample_dataframe.groupby('Education_Level')[
        'Churn'].mean()
    expected_marital_churn = sample_dataframe.groupby('Marital_Status')[
        'Churn'].mean()

    # Map the expected churn rates back to the DataFrame for comparison
    sample_dataframe['expected_Gender_Churn'] = sample_dataframe['Gender'].map(
        expected_gender_churn)
    sample_dataframe['expected_Education_Level_Churn'] = sample_dataframe['Education_Level'].map(
        expected_education_churn)
    sample_dataframe['expected_Marital_Status_Churn'] = sample_dataframe['Marital_Status'].map(
        expected_marital_churn)

    # Compare the actual encoded values with the expected values
    pd.testing.assert_series_equal(
        df_encoded['Gender_Churn'], sample_dataframe['expected_Gender_Churn'], check_names=False, check_dtype=False)
    pd.testing.assert_series_equal(
        df_encoded['Education_Level_Churn'], sample_dataframe['expected_Education_Level_Churn'], check_names=False, check_dtype=False)
    pd.testing.assert_series_equal(
        df_encoded['Marital_Status_Churn'], sample_dataframe['expected_Marital_Status_Churn'], check_names=False, check_dtype=False)

    logging.info("encoder_helper function passed all tests successfully.")


def test_perform_feature_engineering(sample_dataframe):
    """
    Test perform_feature_engineering function to ensure correct train-test split and column selection.
    """
    logging.info("Testing perform_feature_engineering function.")

    response = 'Churn'

    # Ensure sample_dataframe includes necessary columns for feature engineering
    category_lst = config['categories']
    sample_dataframe = cls.encoder_helper(sample_dataframe, category_lst)

    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        sample_dataframe, response)

    # Check if output data frames are of correct type
    assert isinstance(
        X_train, pd.DataFrame), "X_train should be a pandas DataFrame."
    assert isinstance(
        X_test, pd.DataFrame), "X_test should be a pandas DataFrame."
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series."
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series."

    # Ensure data frames are not empty
    assert not X_train.empty, "X_train should not be empty."
    assert not X_test.empty, "X_test should not be empty."
    assert not y_train.empty, "y_train should not be empty."
    assert not y_test.empty, "y_test should not be empty."

    # Validate the correct columns are included in X_train and X_test
    expected_cols = set(config['features']['keep_cols'])
    assert set(
        X_train.columns) == expected_cols, "X_train does not contain the correct columns."
    assert set(
        X_test.columns) == expected_cols, "X_test does not contain the correct columns."

    # Optionally, validate the split ratio
    total_size = len(sample_dataframe)
    train_size = len(X_train)
    test_size = len(X_test)
    assert train_size > test_size, "Training set should be larger than the test set."
    logging.info(
        "perform_feature_engineering function passed all tests successfully.")


@ pytest.mark.skip(reason="not yet implemented")
def test_train_models(train_models):
    '''
    test train_models
    '''
