# main.py
from data_loader import load_data
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
    main()
