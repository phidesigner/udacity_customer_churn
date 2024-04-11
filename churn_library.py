"""
A library for churn prediction analysis, incorporating best practices
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import yaml

# Set seaborn style
sns.set()

# Debbaging for matplotlib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
plt.switch_backend('agg')

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Set up logging
logging.basicConfig(
    filename=config['logging']['filename_main'],
    filemode='w',
    format=config['logging']['format'],
    level=config['logging']['level_main']
)


def import_data(pth):
    """
    Import data from a CSV file at the given path.

    Parameters:
        pth (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        data_frame = pd.read_csv(pth)
        logging.info("Data loaded successfully from %s", pth)
        return data_frame
    except FileNotFoundError:
        logging.error("File not found: %s", pth)
        raise
    except pd.errors.EmptyDataError as exc:
        logging.error("No columns to parse from file: %s", pth)
        raise pd.errors.EmptyDataError(
            "No columns to parse from file") from exc
        # return pd.DataFrame()
    except Exception as ex:
        logging.error(
            "Unexpected error while loading data from % s: % s", pth, ex)
        raise


def create_churn_column(data_frame):
    """
    Adds a 'Churn' column to the DataFrame based on the 'Attrition_Flag'
    column.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame to be processed.

    Returns:
        pd.DataFrame: The DataFrame with the 'Churn' column added.
    """
    try:
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        # Ensuring 'Churn' column is numeric
        data_frame['Churn'] = pd.to_numeric(
            data_frame['Churn'], errors='coerce')
        logging.info("Churn column created successfully.")
        return data_frame
    except KeyError:
        logging.error("'Attrition_Flag' column not found in DataFrame.")
        raise
    except Exception as ex:
        logging.error("Unexpected error while creating churn column: %s", ex)
        raise


def perform_eda(data_frame):
    '''
    Perform EDA on df and save figures to images folder as specified
    in config.

    Input:
        data_frame: pandas DataFrame to perform EDA on.

    Output:
        None: The function saves the EDA plots to files but does not return
        any value.
    '''
    eda_path = config['EDA']['path']
    if not os.path.exists(eda_path):
        os.makedirs(eda_path)
        logging.info("%s directory created for EDA output.", eda_path)

    try:
        # Histogram for Churn
        plt.figure(figsize=(20, 10))
        data_frame['Churn'].hist()
        plt.savefig(os.path.join(eda_path, 'churn_histogram.png'))
        plt.close()

        # Customer Age Histogram
        plt.figure(figsize=(20, 10))
        data_frame['Customer_Age'].hist()
        plt.savefig(os.path.join(eda_path, 'customer_age_histogram.png'))
        plt.close()

        # Marital Status Distribution
        plt.figure(figsize=(20, 10))
        data_frame['Marital_Status'].value_counts(
            normalize=True).plot(kind='bar')
        plt.savefig(os.path.join(eda_path, 'marital_status_distribution.png'))
        plt.close()

        # Total Transactions Distribution
        plt.figure(figsize=(20, 10))
        sns.histplot(data_frame['Total_Trans_Ct'], kde=True, stat='density')
        plt.savefig(os.path.join(eda_path, 'total_trans_ct_distribution.png'))
        plt.close()

        # Correlation Heatmap
        plt.figure(figsize=(20, 10))
        numeric_cols = data_frame.select_dtypes(include=[np.number]).columns
        sns.heatmap(data_frame[numeric_cols].corr(), annot=False,
                    cmap='Dark2_r', linewidths=2)
        plt.savefig(os.path.join(eda_path, 'correlation_heatmap.png'))
        plt.close()

        logging.info("EDA performed and plots saved.")
    except KeyError as key_error:
        logging.error("Failed to generate plots: KeyError %s", key_error)
        raise
    except Exception as ex:
        logging.error(
            "Failed to perform EDA due to an unexpected error: %s", ex)
        raise


def encoder_helper(data_frame, category_lst, response='Churn'):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with Cell 15 from the
    notebook.

    input:
            data_frame: pandas DataFrame
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            data_frame: pandas DataFrame with new columns for each categorical
            feature with suffix '_Churn'
    '''
    try:
        for category in category_lst:
            # Group by the category and calculate the mean of 'Churn' for each
            # category
            category_groups = data_frame.groupby(category)[response].mean()

            # Create a new column for each category with the mean churn rate
            new_column_name = f"{category}_{response}"
            data_frame[new_column_name] = data_frame[category].map(
                category_groups)

            logging.info("Encoded column %s added to dataframe.",
                         new_column_name)
        return data_frame
    except KeyError as key_error:
        logging.error(
            "KeyError in encoder_helper function: %s does not exist in \
             DataFrame: %s", category, key_error)
        raise
    except Exception as ex:
        logging.error("Unexpected error in encoder_helper function: %s", ex)
        raise


def perform_feature_engineering(data_frame, response='Churn'):
    '''
    Performs feature engineering on the DataFrame and splits the data into
    training and testing sets.

    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        y = data_frame[response]

        # Specify columns to keep for the model features
        keep_cols = config['features']['keep_cols']

        # Validate existence of columns in DataFrame
        missing_cols = [
            col for col in keep_cols if col not in data_frame.columns]
        if missing_cols:
            error_message = "Missing columns in DataFrame: %s"
            logging.error(error_message, missing_cols)
            raise KeyError(error_message % missing_cols)

        # Create the features DataFrame X with the columns specified
        # in keep_cols
        X = data_frame[keep_cols]

        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        logging.info(
            "Feature engineering and data splitting completed successfully.")
        return X_train, X_test, y_train, y_test

    except KeyError as key_error:
        logging.error("KeyError in perform_feature_engineering: %s", key_error)
        raise

    except Exception as ex:
        logging.error(
            "Unexpected error in perform_feature_engineering: %s", ex)
        raise


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image
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

    try:
        plt.rc('figure', figsize=(5, 5))

        # Random Forest
        plt.figure()
        plt.text(0.01, 1.25, str('Random Forest Train'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train,
                                                       y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Random Forest Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test,
                                                      y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(os.path.join(
            config['EDA']['results'], 'rf_classification_report.png'))
        plt.close()

        # Logistic Regression
        plt.figure()
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train,
                                                       y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test,
                                                      y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(os.path.join(
            config['EDA']['results'], 'lr_classification_report.png'))
        plt.close()

        logging.info("Classification reports have been saved as images.")
    except Exception as ex:
        logging.error(
            "Failed to generate or save classification report images: %s", ex)


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
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [X_data.columns[i] for i in indices]

        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(X_data.shape[1]), importances[indices])
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.savefig(output_pth)
        plt.close()

        logging.info(
            "Feature importance plot saved successfully to %s", output_pth)
    except AttributeError as ex:
        logging.error("Model does not support feature importances: %s", ex)
        raise
    except Exception as ex:
        logging.error(
            "Failed to generate or save feature importance plot: %s", ex)
        raise


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
    try:
        # Pipeline for Logistic Regression
        pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
        ])
        pipeline_lr.fit(X_train, y_train)
        logging.info("Logistic Regression model trained successfully.")

        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),  # Scaler for consistency only
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        # Setting up parameter grid for GridSearchCV
        param_grid_rf = {
            'classifier__n_estimators': [200, 500],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__max_depth': [4, 5, 100],
            'classifier__criterion': ['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=pipeline_rf,
                              param_grid=param_grid_rf, cv=5)
        cv_rfc.fit(X_train, y_train)
        logging.info(
            "Random Forest model trained successfully with GridSearchCV.")

        # Generate and save classification reports
        y_train_preds_lr = pipeline_lr.predict(X_train)
        y_test_preds_lr = pipeline_lr.predict(X_test)
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        classification_report_image(
            y_train, y_test, y_train_preds_lr, y_train_preds_rf,
            y_test_preds_lr, y_test_preds_rf)

        # Feature importance plot for Random Forest
        feature_importance_plot(cv_rfc.best_estimator_['classifier'],
                                X_train, os.path.join(
            config['EDA']['results'], 'rf_feature_importance.png'))

        # Save models
        joblib.dump(cv_rfc.best_estimator_, os.path.join(
            config['models']['path'], 'rfc_model.pkl'))
        joblib.dump(pipeline_lr, os.path.join(
            config['models']['path'], 'logistic_model.pkl'))

        logging.info("Models and reports have been saved successfully.")
    except Exception as ex:
        logging.error("Model training or saving failed: %s", ex)
        raise


if __name__ == '__main__':
    try:
        # Import data
        df = import_data(config['data']['csv_path'])
        logging.info("Data import complete.")

        # Create 'Churn' column
        df = create_churn_column(df)
        logging.info("'Churn' column creation complete.")

        # Perform EDA
        perform_eda(df)
        logging.info("EDA complete.")

        # Encode categorical features
        category_lst = config['categories']
        df_encoded = encoder_helper(df, category_lst)
        logging.info("Categorical encoding complete.")

        # Perform feature engineering
        RESPONSE = 'Churn'
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df_encoded, RESPONSE)
        logging.info(
            "Feature engineering and data splitting completed successfully.")

        # Train models
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Model training complete.")

    except Exception as ex:
        logging.error("Error in main execution: %s", ex)
