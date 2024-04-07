# library doc string

from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import os

sns.set()

# Debbaging for matplotlib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


try:
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print(f"Error loading the config file: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    filename=config['logging']['filename_main'],
    filemode='w',
    format=config['logging']['format'],
    level=config['logging']['level_main']
)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        logging.info("Data loaded successfully from %s", pth)
        return df
    except FileNotFoundError:
        logging.error("File not found at %s", pth)
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        logging.error("No columns to parse from file %s", pth)
        sys.exit(1)
    except Exception as e:
        logging.error("Unexpected error occurred while importing data: %s", e)
        sys.exit(1)


def perform_eda(df):
    '''
    Perform EDA on df and save figures to images folder as specified in config.

    Input:
        df: pandas DataFrame to perform EDA on.

    Output:
        None: The function saves the EDA plots to files but does not return any value.
    '''
    eda_path = config['EDA']['path']
    if not os.path.exists(eda_path):
        os.makedirs(eda_path)
        logging.info("%s directory created for EDA output.", eda_path)
        # Log Descriptive Statistics
    try:
        descriptive_stats = df.describe()
        logging.info(f"Descriptive Statistics:\n{descriptive_stats}")
        # Consider saving to a file if needed
    except Exception as e:
        logging.error(f"Failed to generate descriptive statistics: {e}")

    # Feature engineering for churn column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    try:
        # Histogram for Churn
        plt.figure(figsize=(20, 10))
        df['Churn'].hist()
        plt.savefig(os.path.join(eda_path, 'churn_histogram.png'))
        plt.close()

        # Customer Age Histogram
        plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        plt.savefig(os.path.join(eda_path, 'customer_age_histogram.png'))
        plt.close()

        # Marital Status Distribution
        plt.figure(figsize=(20, 10))
        df['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
        plt.savefig(os.path.join(eda_path, 'marital_status_distribution.png'))
        plt.close()

        # Total Transactions Distribution
        plt.figure(figsize=(20, 10))
        sns.histplot(df['Total_Trans_Ct'], kde=True, stat='density')
        plt.savefig(os.path.join(eda_path, 'total_trans_ct_distribution.png'))
        plt.close()

        # Correlation Heatmap
        plt.figure(figsize=(20, 10))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sns.heatmap(df[numeric_cols].corr(), annot=False,
                    cmap='Dark2_r', linewidths=2)
        plt.savefig(os.path.join(eda_path, 'correlation_heatmap.png'))
        plt.close()

        logging.info("EDA performed and plots saved.")

    except Exception as e:
        logging.error("Failed to perform EDA: %s", e)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == '__main__':
    try:
        df = import_data(config['data']['csv_path'])
        perform_eda(df)
    except Exception as e:
        logging.error("Error in main execution: %s", e)
