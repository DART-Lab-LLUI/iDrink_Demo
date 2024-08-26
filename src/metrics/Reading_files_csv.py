import os
import pandas as pd
from trc import TRCData
from io import StringIO

# Process .trc files
def get_keypoint_positions(path_file):
    import numpy as np
    trc = TRCData()
    trc.load(filename=path_file)

    list_columns = ["Frame#", "Time"]
    df = pd.DataFrame(columns=list_columns)
    df["Frame#"] = trc["Frame#"]
    df["Time"] = trc["Time"]

    for component in trc["Markers"]:
        df[f"{component}_X"] = np.array(trc[component])[:, 0]
        df[f"{component}_Y"] = np.array(trc[component])[:, 1]
        df[f"{component}_Z"] = np.array(trc[component])[:, 2]

    return df

# Process .sto files
def process_sto_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    endheader_index = next((i for i, line in enumerate(lines) if line.strip() == 'endheader'), None)

    if endheader_index is None:
        raise ValueError("No 'endheader' line found in the file.")

    metadata = {}
    for line in lines[:endheader_index]:
        if '=' in line:
            key, value = line.split('=', 1)
            metadata[key.strip()] = value.strip()

    data_start_index = endheader_index + 1
    column_names = lines[data_start_index].strip().split('\t')
    data_lines = lines[data_start_index + 1:]

    df = pd.read_csv(StringIO(''.join(data_lines)), sep='\t', header=None, names=column_names, engine='python')

    return df

# Determine output filename based on file content
def get_output_filename(file_name, file_ext):
    base_name_parts = file_name.split('_')  # Split the filename to extract relevant parts
    if len(base_name_parts) > 2:
        # Base name is up to the middle segment
        base_name = '_'.join(base_name_parts[:3])
        middle_segment = '_'.join(base_name_parts[1:3])

    # base_name = file_name.split('_')[0]  # Base name from the file
    if file_ext.lower() == '.sto':
        if "BodyKinematics_acc_global" in file_name:
            return f"{base_name}_BK_acc_gb_sto.csv"
        elif "BodyKinematics_vel_global" in file_name:
            return f"{base_name}_BK_vel_gb_sto.csv"
        elif "BodyKinematics_pos_global" in file_name:
            return f"{base_name}_BK_pos_gb_sto.csv"
        elif "Kinematics_dudt" in file_name:
            return f"{base_name}_Kin_dudt_sto.csv"
        elif "Kinematics_q" in file_name:
            return f"{base_name}_Kin_q_sto.csv"
        elif "Kinematics_u" in file_name:
            return f"{base_name}_Kin_u_sto.csv"
    elif file_ext.lower() == '.trc':
        if "filt_butterworth" in file_name:
            return f"{base_name}_filt_trc.csv"
        else:
            return f"{base_name}_trc.csv"
    elif file_ext.lower() == '.mot':
        if "filt_butterworth" in file_name:
            return f"{base_name}_filt_mot.csv"
        else:
            return f"{base_name}_mot.csv"

# Process files in pose-3d and kin_opensim_analyzetool folders
def process_files(input_dir, output_dir):

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)

            try:
                print(f"Processing file: {file_path}")

                if file_ext.lower() == '.sto':
                    df = process_sto_file(file_path)
                elif file_ext.lower() == '.mot':
                    df = pd.read_csv(file_path, sep=None, engine='python', skiprows=10, on_bad_lines='skip')
                elif file_ext.lower() == '.trc':
                    df = get_keypoint_positions(file_path)

                # Determine the new output file name
                output_file_name = get_output_filename(file_name, file_ext)
                output_file_path = os.path.join(output_dir, output_file_name)

                df.to_csv(output_file_path, index=False)
                print(f"Processed and saved: {output_file_path}")

            except Exception as e:
                print(f"Failed to process file {file_path}: {e}")

