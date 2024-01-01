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