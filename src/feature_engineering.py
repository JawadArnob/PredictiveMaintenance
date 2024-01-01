#feature engineering
import pandas as pd


def add_features(df, window_size=5, lag_steps=1):
    sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]
    # Lists to hold new features
    rolling_means = []
    rolling_stds = []
    lags = []

    # Generate new features
    for sensor in sensor_columns:
        grouped_data = df.groupby('unit_number')[sensor]
        rolling_means.append(grouped_data.rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True).rename(f'{sensor}_rolling_mean'))
        rolling_stds.append(grouped_data.rolling(window=window_size, min_periods=1).std().fillna(0).reset_index(level=0, drop=True).rename(f'{sensor}_rolling_std'))
        for lag in range(1, lag_steps + 1):
            lags.append(grouped_data.shift(lag).rename(f'{sensor}_lag_{lag}'))

    # Combine all features
    all_new_features = pd.concat(rolling_means + rolling_stds + lags, axis=1)
    df_combined = pd.concat([df, all_new_features], axis=1)
    df_combined.fillna(0, inplace=True)  # Replace NaNs with 0

    return df_combined