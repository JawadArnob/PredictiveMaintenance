# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fill_missing_values(df):
    # Fill missing values, for example, with the mean of each column
    return df.fillna(df.mean())


def add_rolling_features(df, window_size=5):
    # Example: Add rolling mean and rolling standard deviation as features
    for col in df.columns:
        if col.startswith('sensor_measurement'):
            df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
    return df


def normalize_data(df):
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df


def preprocess_data(df):
    df = fill_missing_values(df)
    df = normalize_data(df)

    # Use forward fill, then fill any remaining NaNs with 0
    df = df.ffill().fillna(0)
    return df


def calculate_rul(df):
    # Calculate the max cycle number for each unit
    max_cycle_per_unit = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycle_per_unit = max_cycle_per_unit.rename(columns={'time_in_cycles': 'max_cycle'})

    # Merge this max cycle back into the original dataframe
    df = df.merge(max_cycle_per_unit, on='unit_number')

    # Calculate RUL for each row
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']

    # Drop the max_cycle column as it's no longer needed
    df.drop(columns=['max_cycle'], inplace=True)

    return df