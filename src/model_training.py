from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Initialize the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)  # R-squared Score

    # Return the trained model and evaluation metrics
    return model, mse, mae, r2

# You can add additional functions or classes as needed for your model training



'''#Model Training
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2'''


'''from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Function to train and evaluate a model on a given dataset
def train_and_evaluate_model(dataset, model_name):
    # Splitting the dataset into features (X) and target (y)
    X = dataset.drop('RUL', axis=1)
    y = dataset['RUL']

    # Splitting the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Making predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculating evaluation metrics
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Model: {model_name}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}\n")

    return model

# Train and evaluate for each dataset
model_FD001 = train_and_evaluate_model(datasets['FD001']['train'], 'FD001')
model_FD002 = train_and_evaluate_model(datasets['FD002']['train'], 'FD002')
model_FD003 = train_and_evaluate_model(datasets['FD003']['train'], 'FD003')
model_FD004 = train_and_evaluate_model(datasets['FD004']['train'], 'FD004')
'''