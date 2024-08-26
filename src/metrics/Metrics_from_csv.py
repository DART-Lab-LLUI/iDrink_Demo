import os
import pandas as pd

def compute_motion_metrics(file_path, output_dir):
    data = pd.read_csv(file_path)
    
    # Identifying relevant columns
    relevant_columns = [col for col in data.columns if col not in ['Frame#', 'Time']]
    
    min_values = data[relevant_columns].min()
    max_values = data[relevant_columns].max()
    range_values = max_values - min_values
    
    # Derivatives
    derivatives = data[relevant_columns].diff().fillna(0)
    
    # Calculate
    derivative_min = derivatives.min()
    derivative_max = derivatives.max()
    derivative_range = derivative_max - derivative_min
    
    results_df = pd.DataFrame({
        'Variable': relevant_columns,
        'Min': min_values,
        'Max': max_values,
        'Range': range_values,
        'Derivative Min': derivative_min,
        'Derivative Max': derivative_max,
        'Derivative Range': derivative_range
    })
    
    output_file_name = os.path.basename(file_path).replace('.csv', '_met.csv')
    output_file_path = os.path.join(output_dir, output_file_name)
    
    results_df.to_csv(output_file_path, index=False)

    print(f"Motion metrics computed and saved to {output_file_path}")

def process_csv_files(input_directory, output_directory):

    # os.makedirs(output_directory, exist_ok=True)
    
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            compute_motion_metrics(file_path, output_directory)




