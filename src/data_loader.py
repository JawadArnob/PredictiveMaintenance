import pandas as pd
import os

def load_data():
    datasets = {}
    column_names = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + [f'sensor_measurement_{i}' for i in range(1, 27)]

    # Loop to process each dataset
    for i in range(1, 5):
        # Absolute file paths for the dataset
        train_file = f'/Users/jawadarnob/PredMain/data/train_FD00{i}.txt'
        test_file = f'/Users/jawadarnob/PredMain/data/test_FD00{i}.txt'
        rul_file = f'/Users/jawadarnob/PredMain/data/RUL_FD00{i}.txt'

        # Loading training data
        train_df = pd.read_csv(train_file, sep=' ', header=None, names=column_names, engine='python').dropna(axis=1, how='all')

        # Loading test data
        test_df = pd.read_csv(test_file, sep=' ', header=None, names=column_names, engine='python').dropna(axis=1, how='all')

        # Loading RUL data
        rul_df = pd.read_csv(rul_file, header=None, names=['RUL'], engine='python')

        # Merge RUL data with test data
        # Assuming the RUL data is in the same order as the test data engines
        test_df = test_df.groupby('unit_number').last().reset_index()
        test_df = test_df.merge(rul_df, left_index=True, right_index=True)

        # Storing in the datasets dictionary
        datasets[f'FD00{i}'] = {'train': train_df, 'test': test_df}

    return datasets
