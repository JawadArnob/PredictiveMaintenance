import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize(df):
    scaler = MinMaxScaler()
    numerical_features = df.select_dtypes(include=['float64', 'int', 'int64']).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def feature_engineering(train_df, test_df):
    # The target is the Remaining Useful Life (RUL)
    target_column_name = 'RUL'
    
    # Ensure the target column exists
    if target_column_name not in train_df.columns:
        raise ValueError(f"The target column '{target_column_name}' is not in the training dataframe.")
    
    # Separate out the target column from the training data
    y_train = train_df[target_column_name].copy()
    X_train = train_df.drop(target_column_name, axis=1)
    
    # Normalize the training data
    X_train_normalized = normalize(X_train)
    
    # Assume that the test data is already only features, no target column
    # Normalize the test data using the same scaler fitted on the training data
    X_test_normalized = normalize(test_df)

    # Add additional feature engineering steps as needed
    
    return X_train_normalized, y_train, X_test_normalized
