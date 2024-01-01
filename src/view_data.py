import pandas as pd
import os

def load_dataset(filename):
    column_names = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + [f'sensor_measurement_{i}' for i in range(1, 22)]
    df = pd.read_csv(filename, sep=' ', header=None)
    df.dropna(axis=1, inplace=True)  # Drop columns with all NaN values
    df.columns = column_names
    return df

def view_data(data_folder):
    dataset_names = ['train_FD001', 'test_FD001', 'train_FD002', 'test_FD002', 'train_FD003', 'test_FD003', 'train_FD004', 'test_FD004']

    for name in dataset_names:
        file_path = os.path.join(data_folder, f'{name}.txt')
        print(f"\nViewing {name} Dataset:")
        df = load_dataset(file_path)
        print(df.head())  # Display the first few rows

if __name__ == "__main__":
    data_folder = '/Users/jawadarnob/PredMain/data/' 
    view_data(data_folder)
