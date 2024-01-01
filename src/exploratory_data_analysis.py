import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Define the function to plot sensor distributions
def plot_sensor_distribution(data, sensor_number, dataset_id, data_folder):
    plt.figure(figsize=(10, 4))
    sns.histplot(data[f'sensor_measurement_{sensor_number}'], kde=True, bins=30)
    plt.title(f'Sensor Measurement {sensor_number} Distribution - {dataset_id}')
    plt.savefig(f'{data_folder}/eda/histograms/Sensor_Measurement_{sensor_number}_Distribution_{dataset_id}.png')  # Save the figure
    plt.close()  # Close the plot to save memory


def load_dataset(filename):
    columns = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
              [f'sensor_measurement_{i}' for i in range(1, 22)]
    df = pd.read_csv(filename, sep=' ', header=None)
    df.dropna(axis=1, inplace=True)  # Drop columns with all NaN values
    df.columns = columns
    return df

data_folder = '/Users/jawadarnob/PredMain/data/'  # Update with the path to your data folder

# Load training datasets
train_FD001 = load_dataset(f'{data_folder}/train_FD001.txt')
train_FD002 = load_dataset(f'{data_folder}/train_FD002.txt')
train_FD003 = load_dataset(f'{data_folder}/train_FD003.txt')
train_FD004 = load_dataset(f'{data_folder}/train_FD004.txt')

# Load test datasets (assuming same file naming convention)
test_FD001 = load_dataset(f'{data_folder}/test_FD001.txt')
test_FD002 = load_dataset(f'{data_folder}/test_FD002.txt')
test_FD003 = load_dataset(f'{data_folder}/test_FD003.txt')
test_FD004 = load_dataset(f'{data_folder}/test_FD004.txt')

print(train_FD001.describe())

# EDA - Use the function to plot histograms for each sensor
for i in range(1, 22):

    #train datasets
    plot_sensor_distribution(train_FD001, i, 'train_FD001', data_folder)
    plot_sensor_distribution(train_FD002, i, 'train_FD002', data_folder)
    plot_sensor_distribution(train_FD003, i, 'train_FD003', data_folder)
    plot_sensor_distribution(train_FD004, i, 'train_FD004', data_folder)

    #test datasets
    plot_sensor_distribution(test_FD001, i, 'test_FD001', data_folder)
    plot_sensor_distribution(test_FD002, i, 'test_FD002', data_folder)
    plot_sensor_distribution(test_FD003, i, 'test_FD003', data_folder)
    plot_sensor_distribution(test_FD004, i, 'test_FD004', data_folder)

    # Will Repeat for other datasets if necessary


#plotting sensor measurements over time

def plot_sensor_data_over_time(df, dataset_name, sensor_num, unit_num, data_folder):
    plt.figure(figsize=(10, 4))
    sns.lineplot(x='time_in_cycles', y=f'sensor_measurement_{sensor_num}', data=df[df['unit_number'] == unit_num])
    plt.title(f'Sensor Measurement {sensor_num} Over Time - Unit {unit_num} - {dataset_name}')
    plt.savefig(f'{data_folder}/eda/time_serieses/Sensor_{sensor_num}_Over_Time_Unit_{unit_num}_{dataset_name}.png')
    plt.close()

data_folder = '/Users/jawadarnob/PredMain/data/'  # Update with your data folder path

dataset_names = ['FD001', 'FD002', 'FD003', 'FD004']
sensors_to_plot = range(1, 22)  # Sensors to plot
unit_numbers = [1, 2, 3, 4]  # Example unit numbers to plot for each dataset

# Loop through each dataset
for dataset in dataset_names:
    # Load training data
    train_data = load_dataset(f'{data_folder}/train_{dataset}.txt')
    
    # Loop through each sensor and unit
    for sensor in sensors_to_plot:
        for unit in unit_numbers:
            plot_sensor_data_over_time(train_data, f'train_{dataset}', sensor, unit, data_folder)
    
    # Load test data
    test_data = load_dataset(f'{data_folder}/test_{dataset}.txt')
    
    # Repeat the plotting for test data
    for sensor in sensors_to_plot:
        for unit in unit_numbers:
            plot_sensor_data_over_time(test_data, f'test_{dataset}', sensor, unit, data_folder)

print("EDA plots generated successfully.")

#if needed


#correlation mattrix
def plot_correlation_matrix(df, dataset_name, data_folder):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.iloc[:, 2:].corr()  # Assuming first two columns are not sensor readings
    sns.heatmap(correlation_matrix, annot=False, cmap='viridis')
    plt.title(f'Correlation Matrix - {dataset_name}')
    plt.savefig(f'{data_folder}/eda/correlation_matrix/Correlation_Matrix_{dataset_name}.png')
    plt.close()

# plotting correlation mattrix

#for train datasets
plot_correlation_matrix(train_FD001, 'train_FD001', data_folder)
plot_correlation_matrix(train_FD002, 'train_FD002', data_folder)
plot_correlation_matrix(train_FD003, 'train_FD003', data_folder)
plot_correlation_matrix(train_FD004, 'train_FD004', data_folder)

#for test datasets
plot_correlation_matrix(test_FD001, 'test_FD001', data_folder)
plot_correlation_matrix(test_FD002, 'test_FD002', data_folder)
plot_correlation_matrix(test_FD003, 'test_FD003', data_folder)
plot_correlation_matrix(test_FD004, 'test_FD004', data_folder)
# Repeat for other datasets as needed

print("EDA plots generated successfully.")