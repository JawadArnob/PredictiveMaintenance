import os
from src.data_loader import load_data

def test_load_data():
    # Assuming load_data() returns a dictionary of DataFrames
    datasets = load_data()
    assert isinstance(datasets, dict), "load_data should return a dictionary"
    
    # Test if the datasets have the expected keys
    expected_keys = ['FD001', 'FD002', 'FD003', 'FD004']
    assert all(key in datasets for key in expected_keys), "Datasets missing expected keys"

    # Test if the data for each key is a DataFrame and has the expected columns
    for key, data in datasets.items():
        assert 'train' in data, f"Training data missing for dataset {key}"
        assert 'test' in data, f"Test data missing for dataset {key}"

        # Check for essential columns
        essential_columns = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'sensor_measurement_1']
        for column in essential_columns:
            assert column in data['train'].columns, f"{column} is missing in the training data for dataset {key}"
            assert column in data['test'].columns, f"{column} is missing in the test data for dataset {key}"

        # Add more assertions as necessary for your specific datasets
