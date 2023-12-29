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


'''#v3
import pandas as pd

def add_features(df, window_size=5, lag_steps=1):
    sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]

    # Create a DataFrame to hold new features
    new_features = pd.DataFrame(index=df.index)

    # Add rolling and lag features
    for sensor in sensor_columns:
        grouped_data = df.groupby('unit_number')[sensor]
        rolling_mean = grouped_data.rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_std = grouped_data.rolling(window=window_size, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)

        new_features[f'{sensor}_rolling_mean'] = rolling_mean
        new_features[f'{sensor}_rolling_std'] = rolling_std

        for lag in range(1, lag_steps + 1):
            new_features[f'{sensor}_lag_{lag}'] = grouped_data.shift(lag)

    # Combine original DataFrame with new features
    df_combined = pd.concat([df, new_features], axis=1)
    df_combined.fillna(0, inplace=True)  # Replace NaNs with 0

    return df_combined
    '''

'''import pandas as pd

def add_features(df, window_size=5, lag_steps=1):
    sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]
    
    # Compute rolling features and lag features
    for sensor in sensor_columns:
        # Add rolling mean and std
        df[f'{sensor}_rolling_mean'] = df.groupby('unit_number')[sensor].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{sensor}_rolling_std'] = df.groupby('unit_number')[sensor].rolling(window=window_size, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
        
        # Add lag features
        for lag in range(1, lag_steps + 1):
            df[f'{sensor}_lag_{lag}'] = df.groupby('unit_number')[sensor].shift(lag)

    # Remove rows with NaN values created by lag features
    df.fillna(0, inplace=True)
    return df

def load_and_enhance_dataset(file_name, data_folder):
    df = pd.read_csv(f'{data_folder}/{file_name}', sep=' ', header=None)
    df.dropna(axis=1, inplace=True)
    columns = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + [f'sensor_measurement_{i}' for i in range(1, 22)]
    df.columns = columns
    return add_features(df)'''



'''import pandas as pd

# Function to add rolling statistics and lag features
def add_features(df, window_size=5, lag_steps=1):
    # Identify sensor columns (assuming they are named 'sensor_measurement_X')
    sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]

    # Add rolling features
    for sensor in sensor_columns:
        # Rolling mean
        df[f'{sensor}_rolling_mean'] = df.groupby('unit_number')[sensor].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        # Rolling standard deviation
        df[f'{sensor}_rolling_std'] = df.groupby('unit_number')[sensor].rolling(window=window_size, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)

    # Add lag features
    for lag in range(1, lag_steps + 1):
        for sensor in sensor_columns:
            df[f'{sensor}_lag_{lag}'] = df.groupby('unit_number')[sensor].shift(lag)

    # Remove rows with NaN values created by lag features
    df.dropna(inplace=True)
    return df

# Load and apply feature engineering to each dataset
data_folder = '/Users/jawadarnob/PredMain/data/'

def load_and_enhance_dataset(file_name):
    df = pd.read_csv(f'{data_folder}/{file_name}', sep=' ', header=None)
    df.dropna(axis=1, inplace=True)  # Drop columns with all NaN values
    columns = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + [f'sensor_measurement_{i}' for i in range(1, 22)]
    df.columns = columns
    return add_features(df)

train_FD001 = load_and_enhance_dataset('train_FD001.txt')
# Repeat for other datasets
'''