#%%
!pip install pandas numpy scikit-learn sqlalchemy openpyxl
!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine
from typing import Tuple, List
import sqlite3
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#%%
# Download the yfinance data
def download_data(tickers, start_date, end_date):
    """
    Downloads historical 'Adj Close' or 'Close' data for the given tickers from Yahoo Finance.
    Handles cases where 'Adj Close' is missing.

    Parameters:
        tickers (list): A list of ticker symbols.
        start_date (str): The start date of the historical data (YYYY-MM-DD).
        end_date (str): The end date of the historical data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame with the date as the index and columns for each ticker.
    """
    data_dict = {}

    for ticker in tickers:
        # Download data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        # Check if 'Adj Close' exists; otherwise, use 'Close'
        if 'Adj Close' in df.columns:
            price_series = df['Adj Close']
        else:
            price_series = df['Close']

        # Rename the series to match the ticker
        price_series.name = ticker
        data_dict[ticker] = price_series

    # Merge all price series on their date index (inner join to ensure matching dates)
    merged_data = pd.concat(data_dict.values(), axis=1, join='inner')
    return merged_data

#Define the tickers for S&P 500, Volatility Index, Crude Oil Futures, and Gold Futures
tickers = ["^GSPC", "^VIX", "CL=F", "GC=F"]

# Fetch historical adjusted close prices for all tickers
data = yf.download(tickers, start="2014-01-24", end="2025-01-15", interval="1d", auto_adjust=False)

# Display the first few rows of the dataset
print(data.head())
#%%
def clean_data(data):
    """
    Cleans the data by dropping rows with missing values,
    calculates daily percentage change for S&P 500 (^GSPC),
    and creates a binary target variable: 1 if next day's S&P 500 return > 0, else 0.

    Parameters:
        data (pd.DataFrame): DataFrame containing the raw price data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with added columns for S&P 500 percentage change and target.
    """
    # Drop rows with missing data
    cleaned = data.dropna().copy()

    # Calculate the daily percentage change for S&P 500
    cleaned['SP500_pct_change'] = cleaned['^GSPC'].pct_change()

    # Create binary target variable: Next day's return > 0 -> 1, else 0
    cleaned['target'] = (cleaned['SP500_pct_change'].shift(-1) > 0).astype(int)

    # Drop rows with missing values resulting from pct_change and shifting
    cleaned.dropna(inplace=True)

    return cleaned

# Check usage:
if __name__ == "__main__":
    tickers = ['^GSPC', '^VIX', 'CL=F', 'GC=F']
    start_date = '2014-01-24'
    end_date = '2025-01-15'

    data = download_data(tickers, start_date, end_date)
    cleaned_data = clean_data(data)

    print("\nCleaned Data:")
    print(cleaned_data.head())
#%%
def save_data_to_db(data, db_name='market_data.db', table_name='market_data'):
    """
    Saves the provided DataFrame to a local SQLite database.

    Parameters:
        data (pd.DataFrame): The DataFrame to save.
        db_name (str): Path to the SQLite database file.
        table_name (str): The name of the table to which data is saved.
                           Defaults to 'stock_data'.
    """
    # Create a connection string for SQLite
    engine = create_engine(f"sqlite:///{db_name}")

    # Save the DataFrame to the SQLite database; if the table exists, replace it
    data.to_sql(name=table_name, con=engine, if_exists='replace', index=True)

    # Optional: Close the engine connection (context-managed connections are recommended)
    engine.dispose()
#%%
def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the data into training, validation, and testing sets based on the provided ratios.
    The data is first sorted by date.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data. It is assumed to have a DatetimeIndex.
        train_ratio (float): The proportion of the data to use for training (default 0.7).
        val_ratio (float): The proportion of the data to use for validation (default 0.15).
        test_ratio (float): The proportion of the data to use for testing (default 0.15).

    Returns:
        tuple: A tuple containing the training, validation, and testing sets.
               Returns None if the input data is invalid or if ratios do not sum to 1.
    """

    # Sort the data by date
    data_sorted = data.sort_index()

    # Calculate split indices
    total_samples = len(data_sorted)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # Split the data
    train_data = data_sorted[:train_size]
    val_data = data_sorted[train_size:train_size + val_size]
    test_data = data_sorted[train_size + val_size:]

    return train_data, val_data, test_data

# Check usage:
if __name__ == "__main__":
    tickers = ['^GSPC', '^VIX', 'CL=F', 'GC=F']
    start_date = '2014-01-24'
    end_date = '2025-01-15'

    data = download_data(tickers, start_date, end_date)
    cleaned_data = clean_data(data)
    train_data, val_data, test_data = split_data(cleaned_data)

    print("\nTraining Data:")
    print(train_data.head())
#%%
def logistic_regression_model(X_train, y_train, X_test, y_test, zero_division=0):
    """
    Analyzes the data by training logistic regression models with ridge (L2 penalty)
    and lasso (L1 penalty) regularisation. Evaluates using accuracy, logistic loss,
    and classification report. The zero_division parameter controls division errors in
    the classification report.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target values.
        zero_division (int or str): Sets the value to return when there is a zero
                                    division in the classification metric calculation.

    Returns:
        dict: A dictionary with model names as keys and their accuracy scores as values.
    """
    models = {
        'Ridge (L2 penalty)': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000),
        'Lasso (L1 penalty)': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        class_report = classification_report(y_test, preds, zero_division=zero_division)
        loss = log_loss(y_test, prob)

        results[name] = {'accuracy': acc, 'log_loss': loss, 'classification_report': class_report}

        print(f"{name} accuracy: {acc:.4f}")
        print(f"Log Loss: {loss:.4f}")
        print(f"Classification Report:\n{class_report}")

    return results

# Check usage:
if __name__ == "__main__":
    tickers = ['^GSPC', '^VIX', 'CL=F', 'GC=F']
    start_date = '2014-01-24'
    end_date = '2025-01-15'

    data = download_data(tickers, start_date, end_date)
    cleaned_data = clean_data(data)
    train_data, val_data, test_data = split_data(cleaned_data)
    X_train, y_train = train_data.drop(columns=['SP500_pct_change', 'target']), train_data['target']
    X_val, y_val = val_data.drop(columns=['SP500_pct_change', 'target']), val_data['target']
    X_test, y_test = test_data.drop(columns=['SP500_pct_change', 'target']), test_data['target']

    results = logistic_regression_model(X_train, y_train, X_val, y_val)
#%%
def sgd_classifier_with_callback(X_train, y_train, n_epochs=10):
    # Callback function to print information during each epoch
    def iteration_callback(epoch, loss, accuracy):
        print(f"Epoch {epoch}")
        print(f"  Loss: {loss:.4f}")
        print(f"  Training Accuracy: {accuracy:.4f}")

    # Instantiate SGDClassifier with logistic loss
    clf = SGDClassifier(loss='log_loss', max_iter=1, tol=None, warm_start=True)
    classes = np.unique(y_train)

    for epoch in range(1, n_epochs + 1):
        # Update model using partial_fit
        clf.partial_fit(X_train, y_train, classes=classes)

        # Compute predictions, training accuracy, and training loss
        y_pred = clf.predict(X_train)
        accuracy_val = accuracy_score(y_train, y_pred)
        y_pred_proba = clf.predict_proba(X_train)
        loss_val = log_loss(y_train, y_pred_proba)

        # Call the iteration callback with current metrics for the epoch
        iteration_callback(epoch, loss_val, accuracy_val)

    return clf

# Check usage:
if __name__ == "__main__":
    tickers = ['^GSPC', '^VIX', 'CL=F', 'GC=F']
    start_date = '2014-01-24'
    end_date = '2025-01-15'

    data = download_data(tickers, start_date, end_date)
    cleaned_data = clean_data(data)
    train_data, val_data, test_data = split_data(cleaned_data)
    X_train, y_train = train_data.drop(columns=['SP500_pct_change', 'target']), train_data['target']
    X_val, y_val = val_data.drop(columns=['SP500_pct_change', 'target']), val_data['target']
    X_test, y_test = test_data.drop(columns=['SP500_pct_change', 'target']), test_data['target']
    logistic_regression_model(X_train, y_train, X_val, y_val)
    sgd_classifier_with_callback(X_train, y_train, n_epochs=10)
#%%
def explore_classification_algorithms(X_train, y_train, X_test, y_test):
    """
    Explores a set of classification algorithms to predict whether the S&P 500's
    next-day return will be positive (1) or negative (0).

    Algorithms:
        - Nearest Neighbor (KNeighborsClassifier)
        - SVM (SVC with probability enabled)
        - Decision Trees (DecisionTreeClassifier)
        - Random Forest (RandomForestClassifier)

    For each algorithm, the function computes accuracy and a detailed classification report.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target values.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target values.

    Returns:
        dict: A dictionary with model names as keys and their evaluation metrics as values.
    """

    models = {
        'Nearest Neighbor': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'Decision Trees': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)

        results[name] = {
            'accuracy': accuracy,
            'classification_report': report
        }

        print(f"{name} Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
    return results

# Check usage:
if __name__ == "__main__":
    tickers = ['^GSPC', '^VIX', 'CL=F', 'GC=F']
    start_date = '2014-01-24'
    end_date = '2025-01-15'

    data = download_data(tickers, start_date, end_date)
    cleaned_data = clean_data(data)
    train_data, val_data, test_data = split_data(cleaned_data)
    X_train, y_train = train_data.drop(columns=['SP500_pct_change', 'target']), train_data['target']
    X_val, y_val = val_data.drop(columns=['SP500_pct_change', 'target']), val_data['target']
    X_test, y_test = test_data.drop(columns=['SP500_pct_change', 'target']), test_data['target']

    explore_classification_algorithms(X_train, y_train, X_val, y_val)
#%%
def run_pipeline():
    # Define parameters
    tickers = ["^GSPC", "^VIX", "CL=F", "GC=F"]
    start_date = "2014-01-24"
    end_date = "2025-01-15"

    # Step 1: Download data
    data = download_data(tickers, start_date, end_date)
    print("Data downloaded.")

    # Step 2: Clean the data
    cleaned_data = clean_data(data)
    print("Data cleaned.")

    # Step 3: Split the data into training, validation, and testing sets
    train_data, val_data, test_data = split_data(cleaned_data)
    print("Data split into training, validation, and test sets.")

    # Step 4: Prepare features and target variables
    X_train = train_data.drop(columns=["SP500_pct_change", "target"])
    y_train = train_data["target"]
    X_val = val_data.drop(columns=["SP500_pct_change", "target"])
    y_val = val_data["target"]
    X_test = test_data.drop(columns=["SP500_pct_change", "target"])
    y_test = test_data["target"]

    # Step 5: Run logistic regression models
    print("Running logistic regression models...")
    logistic_regression_model(X_train, y_train, X_val, y_val)

    # Step 6: Run SGD classifier with callback
    print("Running SGD classifier with callback...")
    sgd_classifier_with_callback(X_train, y_train, n_epochs=10)

    # Step 7: Explore classification algorithms
    print("Exploring classification algorithms...")
    explore_classification_algorithms(X_train, y_train, X_val, y_val)

    # Step 8: Optionally, save the cleaned data to a database for further use
    save_data_to_db(cleaned_data)
    print("Data saved to the database.")

# Optional: Schedule or call the pipeline
if __name__ == "__main__":
    run_pipeline()