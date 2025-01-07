# 随机取一定百分比的数据，在[-0.2x,0.2x]内添加扰动
# Randomly select a certain percentage of the data and add perturbations within the range of [-0.2x, 0.2x].
import pandas as pd
import numpy as np
import random

def add_noise_to_csv(input_file, output_file, proportion, columns=None):
    """
    Add random noise to a proportion of rows in specified columns of a CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the modified CSV file.
        proportion (float): Proportion of rows to modify (0 < proportion <= 1).
        columns (list): List of column names to modify. If None, all columns except the first are modified.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Ensure the proportion is valid
    if not (0 < proportion <= 1):
        raise ValueError("Proportion must be between 0 and 1.")

    # Determine columns to modify
    if columns is None:
        columns = df.columns[1:]  # Exclude the first column (time)

    # Randomly sample rows to modify
    num_rows = len(df)
    num_to_modify = int(proportion * num_rows)
    rows_to_modify = random.sample(range(num_rows), num_to_modify)

    # Apply noise to the selected rows and columns
    for row in rows_to_modify:
        for col in columns:
            original_value = df.at[row, col]
            noise = np.random.uniform(-0.2 * original_value, 0.2 * original_value)
            df.at[row, col] += noise

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Modified CSV saved to {output_file}")

# Example usage with multiple proportions and dataset variable
dataset = 'traffic.csv'  # Dataset name
input_csv = f'D:/model/PerioDformer/dataset/{dataset}'
output_base_path = 'D:/model/PerioDformer/dataset/'
proportions = [0.01, 0.05, 0.1]  # List of proportions

for proportion in proportions:
    output_csv = f"{output_base_path}{int(proportion * 100)}%/{dataset}"
    add_noise_to_csv(input_csv, output_csv, proportion)
