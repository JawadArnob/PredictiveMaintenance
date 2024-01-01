from data_loader import load_data
from data_preprocessing import preprocess_data, calculate_rul  # Import calculate_rul here
from feature_engineering import add_features
from model_training import train_and_evaluate_model

def main():
    datasets = load_data()

    for key, dataset in datasets.items():
        print(f"\nDataset: {key}")

        # Preprocess Data
        datasets[key]['train'] = preprocess_data(dataset['train'])
        datasets[key]['test'] = preprocess_data(dataset['test'])

        # Feature Engineering
        datasets[key]['train'] = add_features(datasets[key]['train'])
        datasets[key]['test'] = add_features(datasets[key]['test'])

        # Calculate RUL for training data
        datasets[key]['train'] = calculate_rul(datasets[key]['train'])

        # Split data into features and target
        X_train = datasets[key]['train'].drop(columns=['RUL'])
        y_train = datasets[key]['train']['RUL']
        X_test = datasets[key]['test'].drop(columns=['RUL'])
        y_test = datasets[key]['test']['RUL']

        # Train and Evaluate Model
        model, mse, mae, r2 = train_and_evaluate_model(X_train, y_train, X_test, y_test)
        print(f"Model Performance on {key}: MSE = {mse}, R^2 = {r2}, MAE = {mae}")

if __name__ == "__main__":
    main()


# main.py
'''from data_loader import load_data
from data_preprocessing import preprocess_data
from feature_engineering import add_features  # Ensure this module exists with the add_features function

def main():
    datasets = load_data()

    for key, dataset in datasets.items():
        print(f"\nDataset: {key}")

        # Count NaNs before preprocessing
        total_nans_before = dataset['train'].isna().sum().sum()
        total_rows_before = len(dataset['train'])
        print(f"Total NaN values in Training Data before preprocessing: {total_nans_before}")
        print(f"Total number of rows in Training Data: {total_rows_before}")

        # Preprocess Data
        datasets[key]['train'] = preprocess_data(dataset['train'])
        datasets[key]['test'] = preprocess_data(dataset['test'])

        # Feature Engineering
        datasets[key]['train'] = add_features(datasets[key]['train'])
        datasets[key]['test'] = add_features(datasets[key]['test'])

        # Count NaNs after preprocessing and feature engineering
        total_nans_after = datasets[key]['train'].isna().sum().sum()
        total_rows_after = len(datasets[key]['train'])
        print(f"Total NaN values in Training Data after preprocessing and feature engineering: {total_nans_after}")
        print(f"Total number of rows in Training Data after preprocessing and feature engineering: {total_rows_after}")

    # Print out samples of the loaded and processed data for verification
    for key, dataset in datasets.items():
        print(f"\nDataset: {key}")
        print("Training Data Sample:")
        print(dataset['train'].head())
        print("\nTest Data Sample:")
        print(dataset['test'].head())

if __name__ == "__main__":
    main()

'''

# main.py
'''from data_loader import load_data
from data_preprocessing import preprocess_data

def main():


    #data_folder = '../data'  # Adjust if your data directory is located differently
    datasets = load_data()

    for key, dataset in datasets.items():

        print(f"\nDataset: {key}")

        # Count NaNs before preprocessing
        total_nans_before = dataset['train'].isna().sum().sum()
        total_rows_before = len(dataset['train'])
        print(f"Total NaN values in Training Data before preprocessing: {total_nans_before}")
        print(f"Total number of rows in Training Data: {total_rows_before}")

        datasets[key]['train'] = preprocess_data(dataset['train'])
        datasets[key]['test'] = preprocess_data(dataset['test'])

        # Count NaNs after preprocessing
        total_nans_after = datasets[key]['train'].isna().sum().sum()
        total_rows_after = len(datasets[key]['train'])
        print(f"Total NaN values in Training Data after preprocessing: {total_nans_after}")
        print(f"Total number of rows in Training Data after preprocessing: {total_rows_after}")

    # Print out samples of the loaded data for verification
    for key, dataset in datasets.items():
        print(f"\nDataset: {key}")
        print("Training Data Sample:")
        print(dataset['train'].head())
        print("\nTest Data Sample:")
        print(dataset['test'].head())

if __name__ == "__main__":
    main()'''
