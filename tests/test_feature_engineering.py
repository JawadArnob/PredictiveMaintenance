import pandas as pd
from src.feature_engineering import add_features  # Replace with your actual script name

def test_add_features():
    # Test if feature engineering adds expected new features
    sample_data = pd.DataFrame({'sensor_measurement_1': [1, 2, 3], 'unit_number': [1, 1, 1]})
    engineered_data = add_features(sample_data)
    # Check for new columns
    assert 'sensor_measurement_1_rolling_mean' in engineered_data.columns, "Rolling mean feature missing"
    # Might add more checks for other features
