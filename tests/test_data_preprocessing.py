import pandas as pd
from src.data_loader import load_data  # Replace 'your_script' with the actual script name
from src.data_preprocessing import preprocess_data


def test_load_data():
    data = load_data()
    assert isinstance(data, dict), "Data should be loaded as a dictionary"
    
    for key, value in data.items():
        assert isinstance(value, dict), f"Value for {key} should be a dictionary"
        assert 'train' in value and 'test' in value, f"{key} should contain 'train' and 'test' data"
        assert isinstance(value['train'], pd.DataFrame), f"Train data for {key} is not a DataFrame"
        assert isinstance(value['test'], pd.DataFrame), f"Test data for {key} is not a DataFrame"


def test_preprocess_data():
    # Test if preprocessing function works correctly
    sample_data = pd.DataFrame({'sensor_measurement_1': [1, 2, 3], 'sensor_measurement_2': [4, 5, 6]})
    processed_data = preprocess_data(sample_data)
    assert not processed_data.isnull().values.any(), "There should be no null values after preprocessing"
    # Might add more assertions as necessary