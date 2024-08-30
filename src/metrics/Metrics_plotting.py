# pip install ipykernel

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import get_cmap

# Run the other scripts

def generate_plots(input_dir, output_dir):

    csv_dir = os.path.join(output_dir, 'CSV')
    results_dir = os.path.join(output_dir, 'Results')
    plot_dir = os.path.join(output_dir, 'Plots')

    # Create directories if they don't exist
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"CSV Directory: {csv_dir}")
    print(f"Results Directory: {results_dir}")
    print(f"Plots Directory: {plot_dir}")

    import sys
    sys.path.append(os.path.join(output_dir, 'src', 'iDrink'))
    from Reading_files_csv import process_files
    process_files(motion_dir, csv_dir)
    from Metrics_from_csv import process_csv_files
    process_csv_files(csv_dir, results_dir)

    # Relevant features include: elbow, shoulder, trunk
    mot_features = ['torso_X', 'torso_Y', 'torso_Z', 'ulna_r_X', 'ulna_r_Y', 'ulna_r_Z', 'ulna_l_X', 'ulna_l_Y', 'ulna_l_Z', 'hand_r_X', 'hand_r_Y', 'hand_r_Z', 'hand_l_X', 'hand_l_Y', 'hand_l_Z'] # same for '.sto'
    trc_features = ['Nose_X', 'Nose_Y', 'Nose_Z', 'LElbow_X', 'LElbow_Y', 'LElbow_Z', 'RElbow_X', 'RElbow_Y', 'RElbow_Z', 'LShoulder_X', 'LShoulder_Y', 'LShoulder_Z', 'RShoulder_X', 'RShoulder_Y', 'RShoulder_Z']


    # ### Plotting

    # ##### Plotting the metrics

    # #### With derivative

    def abbreviate_name(name):
        """
        Create a filename abbreviation without redundant parts.
        
        Args:
        - name (str): Original file name.
        
        Returns:
        - str: Abbreviated file name.
        """
        parts = name.split('_')
        # Construct filename with unique parts
        return f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[4]}"

    def extract_body_parts(features):
        """
        Extract body parts from features.
        
        Args:
        - features (list): List of feature names.
        
        Returns:
        - list: Sorted list of unique body parts.
        """
        body_parts = set(feature.split('_')[0] for feature in features)
        return sorted(body_parts)

    def create_spider_plot(metrics, ax, title, metrics_type='standard'):
        """
        Create a spider plot for the given metrics.
        
        Args:
        - metrics (dict): Dictionary of metrics.
        - ax (matplotlib.axes.Axes): Axes to plot on.
        - title (str): Title of the plot.
        - metrics_type (str): Type of metrics ('standard' or 'derivative').
        """
        features = list(metrics.keys())
        if metrics_type == 'standard':
            min_values = [metrics[feature].get('min', 0) for feature in features]
            max_values = [metrics[feature].get('max', 0) for feature in features]
            range_values = [metrics[feature].get('range', 0) for feature in features]
        elif metrics_type == 'derivative':
            min_values = [metrics[feature].get('der_min', 0) for feature in features]
            max_values = [metrics[feature].get('der_max', 0) for feature in features]
            range_values = [metrics[feature].get('der_range', 0) for feature in features]
        else:
            return

        num_vars = len(features)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        min_values += min_values[:1]
        max_values += max_values[:1]
        range_values += range_values[:1]
        angles += angles[:1]

        # Adjust scale ticks based on whether '_acc' is in the title
        if '_acc' in title.lower():
            scale_ticks = [0, 2, 4, 6]
        else:
            scale_ticks = [0, 0.5, 1, 1.5, 2]
        
        max_scale_value = scale_ticks[-1]

        ax.set_ylim(0, max_scale_value)
        ax.set_yticks(scale_ticks)
        ax.set_yticklabels([str(tick) for tick in scale_ticks])

        ax.fill(angles, min_values, color='red', alpha=0.25, label='Min Values')
        ax.fill(angles, max_values, color='blue', alpha=0.25, label='Max Values')
        ax.fill(angles, range_values, color='green', alpha=0.25, label='Range Values')

        ax.plot(angles, min_values, color='red', linewidth=2, linestyle='solid', label='Min Values')
        ax.plot(angles, max_values, color='blue', linewidth=2, linestyle='solid', label='Max Values')
        ax.plot(angles, range_values, color='green', linewidth=2, linestyle='solid', label='Range Values')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, rotation=45, ha='right')

        ax.set_title(title, size=15, color='black', y=1.1)


    def create_spider_plot_panels(metrics_groups, title, metrics_type='standard'):
        """
        Create and save spider plot panels as separate figures, one for each body part.
        
        Args:
        - metrics_groups (dict): Dictionary of metric groups.
        - title (str): Title of the plot.
        - metrics_type (str): Type of metrics ('standard' or 'derivative').
        """
        for body_part, metrics in metrics_groups.items():
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            create_spider_plot(metrics, ax, title, metrics_type=metrics_type)

            plt.tight_layout(rect=[0, 0, 0.9, 0.95])

            # Save the figure in the specified plot directory
            file_name = f"{title.replace(' ', '_')}_{body_part}.png"
            file_path = os.path.join(plot_dir, file_name)
            plt.savefig(file_path)
            print(f"Saved plot: {file_path}")

            plt.close()

    def create_combined_spider_plot(metrics_groups, title, metrics_type='standard'):
        """
        Create and save a single combined spider plot.
        
        Args:
        - metrics_groups (dict): Dictionary of metric groups.
        - title (str): Title of the plot.
        - metrics_type (str): Type of metrics ('standard' or 'derivative').
        """
        all_metrics = {}
        for group_metrics in metrics_groups.values():
            all_metrics.update(group_metrics)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        create_spider_plot(all_metrics, ax, title=title, metrics_type=metrics_type)

        # plt.suptitle(title, size=24, y=1.05)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        file_name = f"{title.replace(' ', '_')}.png"
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path)
        print(f"Saved plot: {file_name}")
        plt.close()

    def sp_process_csv_files(output_dir = results_dir):
        """
        Process CSV files to extract metrics and create plots.
        
        Args:
        - output_dir (str): Directory containing the CSV files.
        """
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            file_name, file_ext = os.path.splitext(file)
            
            if file_ext.lower() == '.csv':
                try:
                    print(f"Processing file: {file_path}")
                    df = pd.read_csv(file_path)

                    metrics_standard = {}
                    metrics_derivative = {}

                    if all(col in df.columns for col in ['Variable', 'Min', 'Max', 'Range']):
                        for feature in mot_features:
                            if feature in df['Variable'].values:
                                feature_row = df[df['Variable'] == feature].iloc[0]
                                metrics_standard[feature] = {
                                    "min": feature_row["Min"],
                                    "max": feature_row["Max"],
                                    "range": feature_row["Range"]
                                }
                        print(f"Extracted standard metrics for file: {file_name}")

                    if all(col in df.columns for col in ['Derivative Min', 'Derivative Max', 'Derivative Range']):
                        for feature in mot_features:
                            if feature in df['Variable'].values:
                                feature_row = df[df['Variable'] == feature].iloc[0]
                                metrics_derivative[feature] = {
                                    "der_min": feature_row["Derivative Min"],
                                    "der_max": feature_row["Derivative Max"],
                                    "der_range": feature_row["Derivative Range"]
                                }
                        print(f"Extracted derivative metrics for file: {file_name}")

                    body_parts = extract_body_parts(mot_features)
                    print(f"Body parts extracted: {body_parts}")

                    metrics_groups_standard = {}
                    metrics_groups_derivative = {}

                    for body_part in body_parts:
                        if body_part == 'torso':
                            metrics_groups_standard[body_part] = {
                                key: metrics_standard[key] for key in metrics_standard if key.startswith(body_part)
                            }
                            metrics_groups_derivative[body_part] = {
                                key: metrics_derivative[key] for key in metrics_derivative if key.startswith(body_part)
                            }
                        else:
                            if any(key.startswith(f"{body_part}_r") for key in metrics_standard):
                                metrics_groups_standard[f"{body_part}_r"] = {
                                    key: metrics_standard[key] for key in metrics_standard if key.startswith(f"{body_part}_r")
                                }
                            if any(key.startswith(f"{body_part}_l") for key in metrics_standard):
                                metrics_groups_standard[f"{body_part}_l"] = {
                                    key: metrics_standard[key] for key in metrics_standard if key.startswith(f"{body_part}_l")
                                }
                            if any(key.startswith(f"{body_part}_r") for key in metrics_derivative):
                                metrics_groups_derivative[f"{body_part}_r"] = {
                                    key: metrics_derivative[key] for key in metrics_derivative if key.startswith(f"{body_part}_r")
                                }
                            if any(key.startswith(f"{body_part}_l") for key in metrics_derivative):
                                metrics_groups_derivative[f"{body_part}_l"] = {
                                    key: metrics_derivative[key] for key in metrics_derivative if key.startswith(f"{body_part}_l")
                                }

                    # Create and save plots
                    if metrics_standard:
                        create_spider_plot_panels(metrics_groups_standard, title=f'{abbreviate_name(file_name)}', metrics_type='standard')
                        create_combined_spider_plot(metrics_groups_standard, title=f'{abbreviate_name(file_name)}', metrics_type='standard')

                    if metrics_derivative:
                        create_spider_plot_panels(metrics_groups_derivative, title=f'{abbreviate_name(file_name)}', metrics_type='derivative')
                        create_combined_spider_plot(metrics_groups_derivative, title=f'{abbreviate_name(file_name)}', metrics_type='derivative')

                except Exception as e:
                    print(f"Failed to process file {file_path}: {e}")
                    
    sp_process_csv_files(results_dir)

    # Function to plot .sto data
    def plot_sto_data(df, title):
        for feature in mot_features:
            if feature in df.columns:
                # Create a separate plot for each feature
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['time'], df[feature], label=feature)
                min_val = df[feature].min()
                max_val = df[feature].max()
                ax.fill_between(df['time'], min_val, max_val, alpha=0.2)

                ax.set_xlabel('Time')
                ax.set_ylabel('Angle/Position')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                plt.title(f"{title} - {feature}")
                plt.tight_layout(rect=[0, 0, 0.75, 1])
                
                # Save plot
                file_name = f"{title.replace(' ', '_')}_{feature}.png"
                file_path = os.path.join(plot_dir, file_name)
                plt.savefig(file_path)
                plt.close()
                print(f"Saved plot: {file_path}")

    # Function to plot 2D panels for .sto files
    def plot_sto_2d_panel(df, title, features=mot_features):
        # Extract and group features by body parts
        body_parts = {}
        for feature in features:
            parts = feature.split('_')
            body_part = f"{parts[0]}_{parts[1]}" if len(parts) == 3 else parts[0]
            if body_part not in body_parts:
                body_parts[body_part] = []
            body_parts[body_part].append(feature)

        # Create a separate plot for each body part
        for body_part, features in body_parts.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            for feature in features:
                if feature in df.columns:
                    ax.plot(df['time'], df[feature], label=feature)
                    min_val = df[feature].min()
                    max_val = df[feature].max()
                    ax.fill_between(df['time'], min_val, max_val, alpha=0.2)
            
            ax.set_ylabel('Angle/Position')
            ax.set_xlabel('Time')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            ax.set_title(body_part)
            
            # Save plot
            file_name = f"{title.replace(' ', '_')}_{body_part}.png"
            file_path = os.path.join(plot_dir, file_name)
            plt.savefig(file_path)
            plt.close()
            print(f"Saved plot: {file_path}")

    # Function to plot .trc data
    def plot_trc_data(df, title):
        for feature in trc_features:
            if feature in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['Frame#'], df[feature], label=feature)
                min_val = df[feature].min()
                max_val = df[feature].max()
                ax.fill_between(df['Frame#'], min_val, max_val, alpha=0.2)

                ax.set_xlabel('Frame')
                ax.set_ylabel('Position')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                plt.title(f"{title} - {feature}")
                plt.tight_layout(rect=[0, 0, 0.75, 1])

                # Save plot
                file_name = f"{title.replace(' ', '_')}_{feature}.png"
                file_path = os.path.join(plot_dir, file_name)
                plt.savefig(file_path)
                plt.close()
                print(f"Saved plot: {file_path}")

    # Function to plot 3D trajectory data
    def plot_trc_3d(df, title):
        for i in range(0, len(trc_features), 3):
            x_col, y_col, z_col = trc_features[i], trc_features[i+1], trc_features[i+2]
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(df[x_col], df[y_col], df[z_col], label=x_col[:-2], marker='o')
                ax.scatter(df[x_col], df[y_col], df[z_col], s=10)

                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_zlabel('Z Position')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                plt.title(f"{title} - {x_col[:-2]} Trajectory")

                # Save plot
                file_name = f"{title.replace(' ', '_')}_{x_col[:-2]}_3D.png"
                file_path = os.path.join(plot_dir, file_name)
                plt.savefig(file_path)
                plt.close()
                print(f"Saved plot: {file_path}")

    # Function to plot 3D trajectory with gradient color
    def plot_trc_3d_panel(df, title, body_parts):
        fig = plt.figure(figsize=(15, 10))
        cmap = get_cmap('viridis')
        norm = mcolors.Normalize(vmin=df['Frame#'].min(), vmax=df['Frame#'].max())

        for i, (x_col, y_col, z_col) in enumerate(body_parts):
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                colors = cmap(norm(df['Frame#']))
                ax.plot(df[x_col], df[y_col], df[z_col], color='black', alpha=0.7)
                sc = ax.scatter(df[x_col], df[y_col], df[z_col], c=colors, s=10, cmap='viridis')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'{x_col[:-2]} Trajectory')

        cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
        cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Frame Number')
        
        plt.suptitle(title, size=16)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        # Save plot
        file_name = f"{title.replace(' ', '_')}_3D_Panel.png"
        file_path = os.path.join(plot_dir, file_name)
        plt.savefig(file_path)
        plt.close()
        print(f"Saved plot: {file_path}")

    def plot_pos_gb_sto_3d(df, title):
        fig = plt.figure(figsize=(10, 8))
        cmap = get_cmap('viridis')
        norm = mcolors.Normalize(vmin=df['time'].min(), vmax=df['time'].max())

        body_parts = [
            ('hand_r_X', 'hand_r_Y', 'hand_r_Z'),
            ('hand_l_X', 'hand_l_Y', 'hand_l_Z')
        ]
        
        for i, (x_col, y_col, z_col) in enumerate(body_parts):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                colors = cmap(norm(df['time']))
                ax.plot(df[x_col], df[y_col], df[z_col], color='black', alpha=0.7)
                sc = ax.scatter(df[x_col], df[y_col], df[z_col], c=colors, s=10, cmap='viridis')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'{x_col[:-2]} Trajectory')

                # Add colorbar
                cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
                cbar.set_label('Time')

                # Save each plot separately
                file_name = f"{title.replace(' ', '_')}_{x_col[:-2]}_3D.png"
                file_path = os.path.join(plot_dir, file_name)
                plt.savefig(file_path)
                plt.close()
                print(f"Saved plot: {file_path}")


    # Function to process CSV files
    def ts_process_csv_files(output_dir):
        for file in os.listdir(output_dir):
            if '_BK' in file.lower() or '_mot' in file.lower():
                print(f"Ignoring file: {file}")
                continue  
            file_path = os.path.join(output_dir, file)
            file_name, file_ext = os.path.splitext(file)
            
            if file_ext.lower() == '.csv':
                try:
                    df = pd.read_csv(file_path)
                    
                    if '_sto' in file_name.lower() and not 'kin' in file_name.lower():
                        # print(f"Processing .sto file: {file_name}")
                        plot_sto_data(df, title=f'{file_name}')
                        plot_sto_2d_panel(df, title=f'{file_name}', features=mot_features)
                    
                    elif '_trc' in file_name.lower():
                        # print(f"Processing .trc file: {file_name}")
                        body_parts = get_body_parts_from_features(trc_features)
                        plot_trc_data(df, title=f'{file_name}')
                        plot_trc_3d(df, title=f'{file_name}')
                        plot_trc_3d_panel(df, title=f'{file_name}', body_parts=body_parts)

                    if '_pos_gb' in file_name.lower():
                        print(f"Processing _pos_gb file: {file_name}")
                        plot_pos_gb_sto_3d(df, title=f'{file_name}')

                except Exception as e:
                    print(f"Failed to process file {file_path}: {e}")

    # Run the function to process and plot the files
    ts_process_csv_files(csv_dir)