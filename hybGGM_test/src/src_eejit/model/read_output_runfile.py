import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

def extract_run_details(log_content):
    # Regular expression pattern to capture the details in required format
    pattern = (r"(?P<model_type>\w+)\s+with\s+(?P<test_size>[\d.]+)\s+(?P<train_size>[\d.]+)\s+"
               r"(?P<epochs>\d+)\s+(?P<learning_rate>[\d.]+)\s+(?P<batch_size>\d+)\s+\w+\s+"
               r"(?P<status>done|start)\s+at\s+(?P<timestamp>.+)")
    
    # Find all matches in the log content
    matches = re.finditer(pattern, log_content)

    # Dictionary to store start and end times for each unique run configuration
    runs = {}

    for match in matches:
        # Extract details from named groups
        run_key = (
            match.group('model_type'),
            int(match.group('epochs')),
            float(match.group('learning_rate')),
            int(match.group('batch_size'))
        )
        
        timestamp = datetime.strptime(match.group('timestamp'), "%a %b %d %H:%M:%S %Z %Y")
        status = match.group('status')

        # Initialize entry if the configuration key is new
        if run_key not in runs:
            runs[run_key] = {'start': None, 'done': None}

        # Assign timestamp based on the status
        if status == 'start':
            runs[run_key]['start'] = timestamp
        elif status == 'done':
            runs[run_key]['done'] = timestamp

    # Calculate runtime for each run that has both start and done times
    run_durations = []
    for key, times in runs.items():
        if times['start'] and times['done']:
            runtime = times['done'] - times['start']
            run_details = {
                'model_type': key[0],
                'epochs': key[1],
                'learning_rate': key[2],
                'batch_size': key[3],
                'runtime_minutes': runtime.total_seconds() / 60  # Convert runtime to minutes
            }
            run_durations.append(run_details)

    # Return as a DataFrame with selected columns
    return pd.DataFrame(run_durations, columns=['model_type', 'epochs', 'learning_rate', 'batch_size', 'runtime_minutes'])

# Read the output file
output_files_small = glob.glob('/eejit/home/hausw001/HybGGM/hybGGM_test/src/model/jobfiles/hyperparam_small_area*.out')
total = []
for of in output_files_small:
    print(of)
    f = of
    with open(f, 'r') as file:
        log_content = file.read()

    # Extract run details into a DataFrame
    df = extract_run_details(log_content)
    df['run'] = df.model_type + '_' + df.epochs.astype(str) + '_' + df.learning_rate.astype(str) + '_' + df.batch_size.astype(str)
    total.append(df)

# Concatenate all DataFrames into a single one
dftot_small = pd.concat(total)
dftot_small.reset_index(drop=True, inplace=True)

output_files_large = glob.glob('/eejit/home/hausw001/HybGGM/hybGGM_test/src/model/jobfiles/hyperparam_large_area*.out')
total = []
for of in output_files_large:
    print(of)
    f = of
    with open(f, 'r') as file:
        log_content = file.read()

    # Extract run details into a DataFrame
    df = extract_run_details(log_content)
    df['run'] = df.model_type + '_' + df.epochs.astype(str) + '_' + df.learning_rate.astype(str) + '_' + df.batch_size.astype(str)
    total.append(df)

# Concatenate all DataFrames into a single one
dftot_large = pd.concat(total)
dftot_large.reset_index(drop=True, inplace=True)

output_files_half = glob.glob('/eejit/home/hausw001/HybGGM/hybGGM_test/src/model/jobfiles/hyperparam_halftile*.out')
for of in output_files_half:
    print(of)
    f = of
    with open(f, 'r') as file:
        log_content = file.read()

    # Extract run details into a DataFrame
    df = extract_run_details(log_content)
    df['run'] = df.model_type + '_' + df.epochs.astype(str) + '_' + df.learning_rate.astype(str) + '_' + df.batch_size.astype(str)

dftot_half = df
dftot_half.reset_index(drop=True, inplace=True)

colors_half = plt.cm.gist_grey(np.linspace(0, 1, len(dftot_half)))
colors_large = plt.cm.winter(np.linspace(0, 1, len(dftot_large)))
colors_small = plt.cm.autumn(np.linspace(0, 1, len(dftot_small))) 
# Plot runtime vs. batch sizelen
plt.figure(figsize=(10, 6))
#plot every row(run) in a different color
for i,j in zip(range(len(dftot_small)), colors_small):
    plt.scatter(dftot_small['batch_size'][i], dftot_small['runtime_minutes'][i], label=dftot_small['run'][i], color=j)
for i,j in zip(range(len(dftot_large)), colors_large):
    plt.scatter(dftot_large['batch_size'][i], dftot_large['runtime_minutes'][i], label=dftot_large['run'][i], color=j)
for i,j in zip(range(len(dftot_half)), colors_half):
    plt.scatter(dftot_half['batch_size'][i], dftot_half['runtime_minutes'][i], label=dftot_half['run'][i], color=j)
#make the background of the figure light gray
plt.gca().set_facecolor('#f0f0f0')
plt.xlabel('batch_size')
plt.ylabel('Runtime (minutes)')
plt.title('Runtime by epocbatch_sizehs')
plt.xticks(rotation=45)
#legend outside of plot, with two columns
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)

# plt.legend()
plt.show()


############################################################################################################
# import os
# from pathlib import Path
# from datetime import datetime

# def get_creation_date(path):
#     """Gets the creation date of a file or directory."""
#     try:
#         stat = os.stat(path)
#         creation_time = stat.st_ctime
#         return datetime.fromtimestamp(creation_time)
#     except Exception as e:
#         print(f"Error accessing {path}: {e}")
#         return None

# def extract_run_info_from_path(folder_path):
#     # Extract model details from folder name
#     folder_name = Path(folder_path).name
#     model_parts = folder_name.split('_')
#     model_type = model_parts[0]
#     epochs = int(model_parts[1])
#     learning_rate = float(model_parts[2])
#     batch_size = int(model_parts[3])
    
#     # Paths for start (plots folder) and end (y_pred_raw.npy file)
#     plots_folder = Path(folder_path) / 'plots'
#     y_pred_raw_file = Path(folder_path) / 'y_pred_raw.npy'
    
#     # Get start and end times
#     start_time = get_creation_date(plots_folder) if plots_folder.exists() else None
#     end_time = get_creation_date(y_pred_raw_file) if y_pred_raw_file.exists() else None

#     # Calculate runtime
#     runtime_minutes = None
#     if start_time and end_time:
#         print(start_time, end_time)
#         runtime = end_time - start_time
#         runtime_minutes = runtime.total_seconds() / 60  # Convert to minutes

#     # Compile results into a dictionary
#     run_info = {
#         'model_type': model_type,
#         'epochs': epochs,
#         'learning_rate': learning_rate,
#         'batch_size': batch_size,
#         'start_time': start_time,
#         'end_time': end_time,
#         'runtime_minutes': runtime_minutes
#     }
#     # return info as dataframe
#     return pd.DataFrame(run_info, index=[0],columns=['model_type', 'epochs', 'learning_rate', 'batch_size', 'runtime_minutes'])


# # Example usage
# folder_path = '/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/small_area_048_initialtesting/UNet6_10_0.001_1'
# run_info = extract_run_info_from_path(folder_path)

# # Display the result
# print(run_info)

# get_creation_date(folder_path)