# Obtain yfinance package
!pip install yfinance

import yfinance as yf
import pandas as pd

# the download part is written in the next subsection

"""### 2. Inspect
2.1 Load the dataset into a Pandas DataFrame
"""

# Download SPY data from 01/01/2020 to present
spy_data = yf.download("SPY", start="2020-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'), auto_adjust=False)

"""2.2 Print the first 5 rows and identify any missing values."""

# Print first 5 rows
print(spy_data.head())

# Check for missing values in each column
print("\nMissing values per column:")
print(spy_data.isnull().sum())

"""### 3. Clean the Data
3.1 Drop missing values
"""

# Drop rows containing any missing values
spy_data = spy_data.dropna()

# Verify that no missing values remain
print("Missing values after dropna():")
print(spy_data.isnull().sum())

"""3.2 Add a new column for Daily Return
$$ \text{Daily Return} = \frac{\text{Close Price (Today)} - \text{Close Price (Yesterday)}}{\text{Close Price (Yesterday)}} $$
"""

# use Pandas' built-in pct_change() for calculating daily return and put in new column
spy_data[('Daily_Return', 'SPY')] = spy_data[('Close', 'SPY')].pct_change()

# Print the first few rows to confirm the new column
print(spy_data.head())

"""### 4. Explore the Data
4.1 Plot the Ajusted Close Price over time.
"""

import matplotlib.pyplot as plt

# Plot the Close price for SPY over time
plt.figure(figsize=(10, 6))
plt.plot(spy_data[('Adj Close', 'SPY')], label='SPY Close Price')
plt.title('SPY Adjusted Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

"""4.2 Visualize the distribution of daily returns (e.g., histogram).

"""

plt.figure(figsize=(10, 6))
plt.hist(spy_data[('Daily_Return', 'SPY')], bins=50, edgecolor='k')
plt.title('SPY Daily Return Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

"""4.3 Calcualte basic statistics (mean, median, standard deviation) for the Daily Returns"""

# Calculate mean, median, and standard deviation of daily returns
mean_return = spy_data[('Close', 'SPY')].pct_change().mean()
median_return = spy_data[('Close', 'SPY')].pct_change().median()
std_return = spy_data[('Close', 'SPY')].pct_change().std()

print(f"Mean Daily Return: {mean_return}")
print(f"Median Daily Return: {median_return}")
print(f"Standard Deviation of Daily Returns: {std_return}")

"""4.4 Add a 10-day Exponential Moving Average (EMA) column to the data based on the "Adj Close" column (Hint: using the pandas_ta package.)"""

!pip install pandas_ta

import pandas_ta as ta

# Calculate the 10-day EMA of the 'Close' price
spy_data[('EMA_10', 'SPY')] = ta.ema(spy_data[('Adj Close', 'SPY')], length=10)

# Check that the new column is added:
print(spy_data.columns)
spy_data.head(20)

"""## 2 Linear Regression
Use Linear Regression to predict the Adjusted Close Price of the ETF based on the 10-day EMA. (Hint: Use scikit-learn package)
"""

# This is a Sample Code
# X_test = test_df['EMA_10']
# y_test = test_df['Adj Close']
# test_date=test_df.index
# X_train = train_df['EMA_10']
# y_train = train_df['Adj Close']
# train_date=train_df.index




# from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
# reg = LinearRegression().fit(train_df['EMA_10'].values.reshape(-1, 1), train_df['Adj Close'])
# print("R2:", reg.score(train_df['EMA_10'].values.reshape(-1, 1), train_df['Adj Close']))
# print("intercept:", reg.intercept_)
# print("coeficient:", reg.coef_[0])
# model = sm.OLS(y_test, test_df['EMA_10'])
# results = model.fit()
# print(results.summary())

"""1. Handle missing values:


Replace any missing EMA values with reasonable estimates (e.g., the mean or mode of the initial EMA values).


Ensure there are no missing values remaining.
"""

# Replace missing value by mean value and check if there is any missing value left

spy_data[('EMA_10', 'SPY')] = spy_data[('EMA_10', 'SPY')].fillna(spy_data[('EMA_10', 'SPY')].mean())
print("Missing values after handling:")
print(spy_data.isnull().sum())
print(spy_data.head())

"""2. Apply a Linear Regression model:

2.1 Use the entire dataset as input to the regression model.
*(Optional) Split the data into training and testing sets with an 80/20 ratio.*


2.2 Extract and interpret the model's coefficients and intercept.
Compute and visualize the residuals (the differences between actual and predicted values).
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ensure  DataFrame has the needed columns, then drop rows with any NaNs
mldf_linear = spy_data[[('EMA_10', 'SPY'), ('Adj Close', 'SPY')]]

# Define feature matrix X (the EMA_10 column, reshape for sklearn) and target vector y (Adj Close)
X = mldf_linear[('EMA_10', 'SPY')].values.reshape(-1, 1)
y = mldf_linear[('Adj Close', 'SPY')].values

# Train-test split (80% train, 20% test; no shuffling to preserve time series order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train (fit) the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the learned parameters
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# Interpretation:
# Coefficient (slope): Tells how much the Adjusted Close price is expected to change for a 1-unit change in the 10-day EMA.
# Intercept: The predicted Adjusted Close when EMA = 0 (not necessarily meaningful but it’s part of the linear formula).

# Predict on the test set
y_pred = model.predict(X_test)

# Simple performance metrics
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Interpretation:
# Mean Squared Error (MSE): The average of the squared differences between predicted and actual values.
# R-squared: Measures the proportion of variance explained by the model (1.0 is perfect, 0.0 means no explanatory power).

# Visualizing the residual

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5, label='Residuals')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (y_test - y_pred)")
plt.title("Residuals vs. Predicted Values")
plt.legend()
plt.show()

# Visualizing the residual (over time)

test_index = spy_data.index[len(y_train):]

plt.figure(figsize=(10, 5))
plt.plot(test_index, residuals, label='Residual')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
plt.title("Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Residual (y_test - y_pred)")
plt.legend()
plt.show()

"""3. Visualize the results:

3.1 Plot the actual Adjusted Close Prices against the model’s predictions.


3.2 Add a trendline showing the regression fit.
"""

# Plot over time

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Adj. Close", color='blue')
plt.plot(y_pred, label="Predicted Adj. Close", color='red')
plt.title("Actual vs. Predicted Adj. Close Prices (Test Set)")
plt.xlabel("Time Steps (Test Set Index)")
plt.ylabel("Adjusted Close Price")
plt.legend()
plt.show()

# Scatter Plot with a Trendline (Actual vs. Predicted)

import numpy as np

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label="Data Points")

m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m*y_test + b, color="red", label=f"Trendline: y={m:.2f}x+{b:.2f}")

plt.title("Actual vs. Predicted Adj. Close with Trendline")
plt.xlabel("Actual Adj. Close")
plt.ylabel("Predicted Adj. Close")
plt.legend()
plt.show()

"""## 3 Logistic Regression

Use Logistic Regression to classify whether the next day’s return is positive (1) or negative (0) based on historical data.

1. Prepare the Target Variable:

Create a binary target variable


$$
\text{Next Day Return (Binary)} =
\begin{cases}
1 & \text{if } \text{Daily Return} \geq 0, \\
0 & \text{if } \text{Daily Return} < 0.
\end{cases}
$$
"""

# Create next-day return by shifting Daily_Return by -1 to align the next day's return with today's row
spy_data[('Next_Day_Return','SPY')] = spy_data[('Daily_Return','SPY')].shift(-1)
spy_data[('Target','SPY')] = (spy_data[('Next_Day_Return','SPY')] >= 0).astype(int)
spy_data.head(20)

"""2. Apply Logistic Regression:

2.1 Use scikit-learn’s LogisticRegression model.
*(Optional) Split the data into training and testing sets with an 80/20 ratio.*


2.2 Fit the model on the entire dataset with 10-day EMA as the feature and the binary Next Day Return as the target.
"""

from sklearn.model_selection import train_test_split

# Select the relevant columns (feature & target) and drop rows with NaNs
mldf_logistic = spy_data[[('EMA_10','SPY'), ('Target','SPY')]].dropna()

# Define feature matrix (X) and target vector (y)
X = mldf_logistic[('EMA_10','SPY')].values.reshape(-1, 1)  # reshape for sklearn
y = mldf_logistic[('Target','SPY')].values

# Split into train & test sets (80% train, 20% test); no shuffle to preserve time order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
log_reg = LogisticRegression()

# Fit the model on the training set
log_reg.fit(X_train, y_train)

# Print out model coefficients
print("Coefficient (weight for EMA_10):", log_reg.coef_[0][0])
print("Intercept:", log_reg.intercept_[0])

# Evaluate on the test set
y_pred = log_reg.predict(X_test)

# Combine them into a single DataFrame to compare result
test_index = mldf_logistic.index[len(X_train):]
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index = test_index)

# Show the first 10 rows
print(results_df.head(10))

# Calculate accuracy and other metrics
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""3. Interpret the Results:

3.1 Extract the model coefficients.
What does the coefficient represent in this context?


3.2 Explain the role of the intercept in determining probabilities.
"""

# The coefficient essentially quantifies the relationship between the predictor (EMA_10) and the log odds of the target (positive return = 1).
# It could be explained that for every 1-unit increase in the 10-day Exponential Moving Average (EMA_10), the log odds of the next day’s return being positive (vs. negative) decreases by 0.001191
# But since the magnitude is so small, higher EMA_10 values are just weakly associated with a lower probability of a positive return
# The coefficient suggests that EMA_10 has minimal predictive power for next-day returns
# perhaps it is reflecting a mean-reversion tendency but I am not too sure due to the magnitude of the coefficient

# The intercept is the baseline log odds of a positive return when predictor (EMA_10) is zero.
# Interpretation: If EMA_10 = 0 (which is unrealistic in practice, as prices cannot be zero), the log odds of a positive return would be 0.6105717
# The intercept shifts the baseline probability in the logistic function.
# If EMA_10 = 0, Log odds = 0.6105717 + (−0.001191*0) = 0.6105717 and probability would be e^0.6105717 / (1 + e^0.6105717) =
# Another case if EMA_10 = 400 (more realistic), Log odds = 0.6105717 + ( − 0.001191 * 400 ) = 0.6105717 − 0.4764 = 0.134 and probability would be
