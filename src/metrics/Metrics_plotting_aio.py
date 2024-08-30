import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors

# files to csv
from trc import TRCData
from io import StringIO

from imu import calculate_smoothness


def generate_plots(input_dir, output_dir):
    # Setup directories
    csv_dir = os.path.join(output_dir, 'CSV')
    results_dir = os.path.join(output_dir, 'Results')
    plot_dir = os.path.join(output_dir, 'Plots')

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"CSV Directory: {csv_dir}")
    print(f"Results Directory: {results_dir}")
    print(f"Plots Directory: {plot_dir}")
    
    def plot_smoothing():
        input_dir_path = Path(input_dir)
        found_files = [str(file) for file in input_dir_path.glob( '*_motion.csv')]
        smoothness_dict = {}
        for file in found_files:
            match = re.search(r'-(?!.*-)([^-_]+)_motion\.csv', file)
            if match:
                segment = match.group(1)
                smoothness_dict[segment] = calculate_smoothness(file)
        data = {k: round(v, 2) for k, v in smoothness_dict.items()}
        labels = list(data.keys())
        values = list(data.values())
        fig, ax = plt.subplots()
        bars = ax.bar(labels, values, color='lightcoral')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval}', ha='center', va='bottom')
        ax.set_xlabel('Segments')
        ax.set_ylabel('Smoothing')
        plt.savefig(os.path.join(plot_dir, 'smoothing.png'), bbox_inches='tight')

    plot_smoothing()

    # Import functions from the external iDrink library
    # from Reading_files_csv import process_files
    # from Metrics_from_csv import process_csv_files

    # files to csv
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

    # csv to metrics
    def compute_motion_metrics(file_path, output_dir):
        data = pd.read_csv(file_path)
        
        # Identifying relevant columns
        relevant_columns = [col for col in data.columns if col not in ['Frame#', 'Time']]
        
        min_values = data[relevant_columns].min()
        max_values = data[relevant_columns].max()
        range_values = max_values - min_values
        mean_values = data[relevant_columns].mean()  # Added mean calculation
        
        # Derivatives
        derivatives = data[relevant_columns].diff().fillna(0)
        
        # Calculate derivatives metrics
        derivative_min = derivatives.min()
        derivative_max = derivatives.max()
        derivative_range = derivative_max - derivative_min
        
        results_df = pd.DataFrame({
            'Variable': relevant_columns,
            'Min': min_values,
            'Max': max_values,
            'Range': range_values,
            'Mean': mean_values,  # Added mean to the results
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


    # Process motion files to generate CSVs
    process_files(input_dir, csv_dir)
    process_csv_files(csv_dir, results_dir)

    # Features for spider plots
    spider_features = [
        'elbow_flex_r', 'elbow_flex_l', 'pro_sup_r', 'pro_sup_l',
        'arm_add_r', 'arm_add_l', 'arm_flex_r', 'arm_flex_l'
    ]

    # Helper function to abbreviate names
    def abbreviate_name(name):
        parts = name.split('_')
        return f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[4]}"

    def compute_euclidean_distance(df, prefix):
        # Extract the rows for the specific prefix
        x_row = df[df['Variable'] == f'{prefix}_X']
        y_row = df[df['Variable'] == f'{prefix}_Y']
        z_row = df[df['Variable'] == f'{prefix}_Z']

        if not x_row.empty and not y_row.empty and not z_row.empty:
            x_min = x_row['Min'].values[0]
            y_min = y_row['Min'].values[0]
            z_min = z_row['Min'].values[0]
            return ((x_min ** 2 + y_min ** 2 + z_min ** 2) ** 0.5)
        else:
            missing_vars = [f'{prefix}_X', f'{prefix}_Y', f'{prefix}_Z']
            found_vars = [row['Variable'].values[0] for row in [x_row, y_row, z_row] if not row.empty]
            missing_vars = [var for var in missing_vars if var not in found_vars]
            print(f"Warning: Rows for variables {missing_vars} not found in DataFrame")
            return np.nan

    def compute_velocity(df, x_col, y_col, z_col):
        print("ok")
        x = df[df['Variable'] == x_col]['Mean'].iloc[0]
        y = df[df['Variable'] == y_col]['Mean'].iloc[0]
        z = df[df['Variable'] == z_col]['Mean'].iloc[0]
        # print(f"df[x_col]{x.values}")
        return np.sqrt(x**2 + y**2 + z**2)

    def extract_additional_features(df_pos_gb, df_vel_gb, df_u_mot):
        additional_features = {}

        # Compute torso distances
        torso_distance = compute_euclidean_distance(df_pos_gb, 'torso')
        additional_features['torso_max_dist'] = torso_distance

        # Compute hand velocities
        additional_features['hand_r_vel'] = compute_velocity(df_vel_gb, 'hand_r_X', 'hand_r_Y', 'hand_r_Z')
        additional_features['hand_l_vel'] = compute_velocity(df_vel_gb, 'hand_l_X', 'hand_l_Y', 'hand_l_Z')

        # Compute elbow flexion/extension velocities
        elbow_flex_r_row = df_u_mot[df_u_mot['Variable'] == 'elbow_flex_r']
        elbow_flex_l_row = df_u_mot[df_u_mot['Variable'] == 'elbow_flex_l']

        if not elbow_flex_r_row.empty and not elbow_flex_l_row.empty:
            if 'Mean' in elbow_flex_r_row.columns and 'Mean' in elbow_flex_l_row.columns:
                additional_features['elbow_flex_r_vel'] = elbow_flex_r_row['Mean'].iloc[0]
                additional_features['elbow_flex_l_vel'] = elbow_flex_l_row['Mean'].iloc[0]
            else:
                print("Warning: One or more derivative columns are missing for elbow flexion/extension")
        else:
            print("Warning: Required rows for elbow flexion/extension not found.")

        return additional_features


    # Create a spider plot
    def create_spider_plot(metrics, ax, title, metrics_type='standard'):
        features = list(metrics.keys())
        
        if metrics_type == 'standard':
            # Extract min, max, and range values
            min_values = [metrics[feature].get('min', 0) for feature in features]
            max_values = [metrics[feature].get('max', 0) for feature in features]
            range_values = [metrics[feature].get('range', 0) for feature in features]
            
            # Close the circle
            min_values += min_values[:1]
            max_values += max_values[:1]
            range_values += range_values[:1]
            
        elif metrics_type == 'additional':
            # Extract single values
            values = [metrics[feature] for feature in features]
            
            # Since only one value is provided, we'll use it for all dimensions
            min_values = max_values = range_values = values
            
            # Close the circle
            min_values += min_values[:1]
            max_values += max_values[:1]
            range_values += range_values[:1]
            
        num_vars = len(features)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Plot the values
        ax.fill(angles, min_values, color='red', alpha=0.25, label='Min Values' if metrics_type == 'standard' else 'Values')
        ax.fill(angles, max_values, color='blue', alpha=0.25, label='Max Values' if metrics_type == 'standard' else 'Values')
        ax.fill(angles, range_values, color='green', alpha=0.25, label='Range Values' if metrics_type == 'standard' else 'Values')

        ax.plot(angles, min_values, color='red', linewidth=2, linestyle='solid')
        ax.plot(angles, max_values, color='blue', linewidth=2, linestyle='solid')
        ax.plot(angles, range_values, color='green', linewidth=2, linestyle='solid')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=12)

        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=12)


    # Create a combined spider plot including additional features
    def create_combined_spider_plot(metrics_groups, title, metrics_type='standard'):
        all_metrics = {}
        for group_metrics in metrics_groups.values():
            all_metrics.update(group_metrics)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        create_spider_plot(all_metrics, ax, title=title, metrics_type=metrics_type)

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        file_name = f"{title.replace(' ', '_')}.png"
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Saved plot: {file_name}")
        plt.close()

    def plot_spider_add(additional_features, title):
        # Extract the labels and values from the dictionary
        features = list(additional_features.keys())
        values = list(additional_features.values())
        
        # Creating the angles for the spider plot
        num_vars = len(features)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Close the circle for values
        values += values[:1]
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.25, label='Values')
        ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')
        
        # Adding labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=12)
        
        # Adding y-tick labels to indicate the levels of the circles
        ticks = np.linspace(0, 1.5, num=7)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in ticks], fontsize=12)

        # Title and layout adjustments
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        ax.set_title(title, size=15, color='black', y=1.1)
        
        # Save the plot
        file_name = f"{title.replace(' ', '_')}.png"
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Saved plot: {file_name}")
        plt.close()

    def sp_process_csv_files(output_dir):
        pos_gb_files = {}
        vel_gb_files = {}
        u_files = {}

        # Load files into dictionaries based on their suffixes
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            file_name, file_ext = os.path.splitext(file)
            base_name = '_'.join(file_name.split('_')[:3])
            
            if file_ext.lower() == '.csv':
                if '_pos_gb' in file_name.lower():
                    pos_gb_files[base_name] = pd.read_csv(file_path)
                elif '_vel_gb' in file_name.lower():
                    vel_gb_files[base_name] = pd.read_csv(file_path)
                elif '_u' in file_name.lower():
                    u_files[base_name] = pd.read_csv(file_path)

        # Process files and generate plots
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            file_name, file_ext = os.path.splitext(file)

            if file_ext.lower() == '.csv' and ('_q' in file_name.lower() or '_mot' in file_name.lower()):
                try:
                    print(f"Processing file: {file_path}")
                    df = pd.read_csv(file_path)

                    base_name = '_'.join(file_name.split('_')[:3])

                    related_pos_gb_file = pos_gb_files.get(base_name)
                    related_vel_gb_file = vel_gb_files.get(base_name)
                    related_u_file = u_files.get(base_name)

                    # Extract features and metrics
                    additional_features = extract_additional_features(related_pos_gb_file, related_vel_gb_file, related_u_file)

                    metrics_standard = {}
                    if not df.empty and all(col in df.columns for col in ['Variable', 'Min', 'Max', 'Range']):
                        for feature in spider_features:
                            feature_row = df[df['Variable'] == feature]
                            if not feature_row.empty:
                                feature_row = feature_row.iloc[0]
                                metrics_standard[feature] = {
                                    "min": feature_row["Min"],
                                    "max": feature_row["Max"],
                                    "range": feature_row["Range"]
                                }

                    # Create spider plots
                    if metrics_standard:
                        create_combined_spider_plot(
                            {"all": metrics_standard},
                            title=f'{abbreviate_name(file_name)}_std',
                            metrics_type='standard'
                        )

                    if additional_features:
                        plot_spider_add(additional_features, title=f'{abbreviate_name(file_name)}_sup')

                except Exception as e:
                    print(f"Failed to process file {file}: {e}")

    sp_process_csv_files(results_dir)

    # Function to plot 2D data from .csv file

    def plot_sto_data(df, title, plot_dir, spider_features):
        # Determine the Y-axis label based on the filename suffix
        if '_dudt' in title:
            y_label = 'Acceleration'
        elif '_u' in title:
            y_label = 'Velocity'
        elif '_q' in title:
            y_label = 'Position'
        elif '_mot' in title:
            y_label = 'Degrees'  # Adjust if needed for your specific case
        else:
            y_label = 'Degrees'
        
        # Generate line plots with standard deviation fill
        for feature in df.columns:
            if feature in spider_features:
                plt.figure(figsize=(10, 6))
                
                # Calculate mean and standard deviation
                mean = df[feature].mean()
                std_dev = df[feature].std()
                
                plt.plot(df['time'], df[feature], label=feature)
                plt.fill_between(df['time'], mean - std_dev, mean + std_dev, alpha=0.2, color='orange', label='±1 SD')
                
                plt.xlabel('Time')
                plt.ylabel(y_label)
                # plt.title(f"{title} - {feature}")
                plt.legend()
                plt.tight_layout()

                if '_dudt' in title:
                    plt.ylim(-10000, 10000)
                elif '_mot' in title:
                    plt.ylim(-50, 200)
                elif '_q' in title:
                    plt.ylim(-50, 200)
                elif '_u' in title:
                    plt.ylim(-2000, 2000)
                
                file_name = f"{title}_{feature}.png"
                plt.savefig(os.path.join(plot_dir, file_name),bbox_inches='tight')
                print(f"Saved plot: {file_name}")
                plt.close()

    def plot_spider_feature(df, feature, title, plot_dir):
        # Generate spider (radar) plot
        if feature not in df.columns:
            return

        num_vars = len(spider_features)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = [df[feature].mean() for feature in spider_features]  # Example: Mean value

        # Make the plot circular
        angles += angles[:1]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='red', alpha=0.25)
        ax.plot(angles, values, color='red', linewidth=2)

        # Labels for features
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spider_features)

        # plt.title(f"{title} - Spider Plot for {feature}")
        plt.tight_layout()
        file_name = f"{title}_{feature}_spider.png"
        plt.savefig(os.path.join(plot_dir, file_name), bbox_inches='tight')
        print(f"Saved spider plot: {file_name}")
        plt.close()

    # Function to create 3D motion data plots
    def plot_3d_motion_data(df, title_prefix, plot_dir):
        def plot_individual_3d(data, label, file_suffix):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            if 'time' in data.columns:
                # Normalize time for colormap
                norm = mcolors.Normalize(vmin=data['time'].min(), vmax=data['time'].max())
                cmap = plt.get_cmap('viridis')
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                
                # Scatter plot with color mapping
                sc = ax.scatter(data['X'], data['Y'], data['Z'], c=data['time'], cmap=cmap, norm=norm, s=10)
                # Add colorbar
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink = 0.6, pad = 0.1)
                cbar.set_label('Time')
                ax.plot(data['X'], data['Y'], data['Z'], color='gray', alpha=0.8)
            else:
                # Plot without color mapping if time is not available
                ax.plot(data['X'], data['Y'], data['Z'], label=label, color='b')

            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
            # if 'hand' in file_suffix :
            #     ax.set_xlim(-0.1, 0.3)
            #     ax.set_ylim(-1.5, 1.5)
            #     ax.set_zlim(0, 0.25)
            # elif  'torso' in file_suffix :
            #     ax.set_xlim(-0.08, 0.02)
            #     ax.set_ylim(1.25, 1.5)
            #     ax.set_zlim(-0.2, -0.16)
            ax.legend()
            ax.set_title(f"{title_prefix}_{file_suffix} - 3D Motion Trajectory")

            file_name = f"{title_prefix}_{file_suffix.lower()}_3D.png"
            plt.savefig(os.path.join(plot_dir, file_name), bbox_inches='tight')
            print(f"Saved 3D plot: {file_name}")
            plt.close()

        try:
            # Plot for each body part
            for body_part in ['hand_r', 'hand_l', 'torso']:
                if all(col in df.columns for col in [f'{body_part}_X', f'{body_part}_Y', f'{body_part}_Z']):
                    body_part_df = df[[f'{body_part}_X', f'{body_part}_Y', f'{body_part}_Z']].copy()
                    body_part_df.columns = ['X', 'Y', 'Z']  # Rename columns for consistency
                    
                    # Add time column if it exists
                    if 'time' in df.columns:
                        body_part_df['time'] = df['time']
                    
                    plot_individual_3d(body_part_df, body_part, body_part)

        except Exception as e:
            print(f"Failed to create 3D plot: {e}")
        finally:
            plt.close()

    # Function to process CSV files for 2D
    def ts_process_csv_files(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            file_name, file_ext = os.path.splitext(file)
            
            if file_ext.lower() == '.csv':
                try:
                    print(f"Processing file: {file_path}")
                    df = pd.read_csv(file_path)

                    plot_sto_data(df, f"{abbreviate_name(file_name)}_2D", plot_dir, spider_features)

                except Exception as e:
                    print(f"Failed to process file {file_path}: {e}")

    # Function to process CSV files for 3D motion plots only
    def mot_process_csv_files(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            file_name, file_ext = os.path.splitext(file)
            
            if 'pos_gb_' in file_name.lower():
                try:
                    print(f"Processing file: {file_path}")
                    df = pd.read_csv(file_path)

                    plot_3d_motion_data(df, f"T{abbreviate_name(file_name)}", plot_dir)

                except Exception as e:
                    print(f"An error occurred while processing file: {file_path}")
                    print(e)
            else:
                print(f"Skipped file (not a relevant CSV): {file_path}")


    # Call the functions to generate plots
    ts_process_csv_files(csv_dir)
    mot_process_csv_files(csv_dir)
    

generate_plots('/home/arashsm79/bids_root/sub-4a20/ses-20240901a/motion/', '/home/arashsm79/bids_root/sub-4a20/ses-20240901a/metric/')