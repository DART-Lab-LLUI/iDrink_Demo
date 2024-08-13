import logging
import logging.handlers
import os
import string

import numpy as np
import opensim
import pandas as pd
from trc import TRCData

from .iDrinkUtilities import edit_xml

def ik_tool_to_csv(curr_trial):
    """
    This function saves the angles calculated by the Inverse Kinematics Tool to a .csv file.
    """
    # Get Filename and path
    filename = curr_trial.get_filename()
    dir_out = curr_trial.dir_kin_ik_tool

    # Make sure, Folder exists
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    file_path = os.path.realpath(os.path.join(dir_out, filename))

    # Read data and save to .csv
    opensim_motion = os.path.join(curr_trial.dir_trial, curr_trial.opensim_motion)
    _, dat_measured = read_opensim_file(opensim_motion)
    dat_measured.to_csv(f"{file_path}.csv", index=False)


def read_opensim_file(file_path):
    """
    Reads an opensim file (.mot, .sto) and returns metadata as 2D-list and Data as pandas Dataframe.

    input:
        Path to file

    output:
        Metadata: List
        Data: pandas Dataframe
    """
    # Read Metadata and end at "endheader"
    metadata = []
    with open(file_path, 'r') as file:
        for row in file:
            metadata.append(row)
            if "endheader" in row.strip().lower():
                break

        file.close()

    # Read the rest of the file into a DataFrame
    df = pd.read_csv(file_path, skiprows=len(metadata), sep="\t")

    return metadata, df


def open_sim_pipeline(curr_trial):
    """
    This function loads an opensim Model and runs the Scale- and Inverse Kinematics Tool on it.
    the .mot file will be saved to the determined folder.
    Then it creates the table for the measured data and returns it to tha calling instance.

    Input:
        dict_path:  Dictionary containing the following paths:
                        - model:        OpenSim Model
                        - scaling:      .xml for scaling
                        - invkin:       .xml for Inverse Kinematics
                        - modelscaled:  scaled model for Inverse Kinematics
                        - motfile:      Motion file for Inverse Kinematics
                        - marker:       marker files
                        - filmark:      filtered marker files
    """

    seq_name = os.path.basename(curr_trial.dir_trial)

    logging.info("\n\n---------------------------------------------------------------------")
    logging.info(f"Running OpenSim for {seq_name}, for all frames.")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {curr_trial.dir_trial}")

    def correct_skeleton_orientation(opensim_motion, realign=True, center=True):
        """
        Loads an openSim Motion File and changes Alignment and position in Coordinate-System.

        This Function is probably not needed anymore. FOr now, I will keep it

        if realign True:
            Tilt, List and Rotation are set to 0

        if center True:
            X, Z axes are set to 0
            Y axis is set to 1

        """

        metadata, data = read_opensim_file(opensim_motion)

        # Process Data
        if realign:
            data[["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx"]] = 0

        if center:
            data[["pelvis_tx", "pelvis_ty", "pelvis_tz"]] = [0, 1, 0]

        with open(opensim_motion, 'w') as file:
            # Write Metadata to file
            for row in metadata:
                file.write(row)

            # Add the DatFrame data to the file
            data.to_csv(file, sep='\t', index=False)

            file.close()

    def stabilize_hip_movement(path_trc):
        """
        Take and trc file and smooth the movement of CHip, RHip, and LHip.

        I used different methods. Sliding Window mean / median, Splines, Interpolation.

        The most stable result gave putting the mean on all values of CHip, RHip, and LHip.

        TODO: Make function more efficient
        """

        trc = TRCData()

        trc.load(filename=path_trc)
        comp = {"CHip": -1,
                "RHip": -1,
                "LHip": -1
                }

        """comp = ["CHip", "RHip", "LHip"]
        comp_in_trc = [False, False, False]"""
        hips = {}

        for c in comp.keys():
            if c in trc.keys():
                comp[c] = list(trc.keys()).index(c) - list(trc.keys()).index("Time") - 1
                print(f"{c} found at index {comp[c]}")

                hips[c] = np.array(trc[c])
                hips[c][:, ] = np.nanmean(hips[c], axis=0, keepdims=True)

            else:
                print(f"{c} not found in .trc File.")

        for frame in trc['Frame#']:
            for c in comp.keys():
                if comp[c] > -1:
                    trc[frame][1][comp[c]] = hips[c].tolist()[frame - 1]

        trc.save(path_trc)

    try:
        """get paths to default files"""
        scaling_default = os.path.join(curr_trial.dir_default, f"Scaling_Setup_iDrink_{curr_trial.pose_model}.xml")
        invkin_default = os.path.join(curr_trial.dir_default, f"IK_Setup_iDrink_{curr_trial.pose_model}.xml")
        analyze_default = os.path.join(curr_trial.dir_default, f"AT_Setup.xml")

        """edit xml files for scaling and Inverse Kinematics"""
        edit_xml(scaling_default, ["model_file", "output_model_file", "marker_file", "time_range"],
                 [curr_trial.opensim_model, curr_trial.opensim_model_scaled, curr_trial.opensim_marker_filtered,
                  curr_trial.opensim_scaling_time_range], target_file=curr_trial.opensim_scaling)
        edit_xml(invkin_default, ["marker_file", "output_motion_file", "model_file", "time_range"],
                 [curr_trial.opensim_marker_filtered, curr_trial.opensim_motion, curr_trial.opensim_model_scaled,
                  curr_trial.opensim_IK_time_range], target_file=curr_trial.opensim_inverse_kinematics)
        edit_xml(analyze_default, ["results_directory", "model_file", "coordinates_file", "initial_time", "final_time"],
                 [curr_trial.opensim_dir_analyze_results, curr_trial.opensim_model_scaled, curr_trial.opensim_motion,
                  curr_trial.opensim_ana_init_t, curr_trial.opensim_ana_final_t],
                 target_file=curr_trial.opensim_analyze)

    except FileNotFoundError as e:
        print("Either Scaling or Inverse Kinematics Setup file not found. Check default folder.\n")
        print(e)

    if curr_trial.stabilize_hip:
        stabilize_hip_movement(os.path.join(curr_trial.dir_trial, curr_trial.opensim_marker_filtered))

    model = opensim.Model(curr_trial.opensim_model)
    state = model.initSystem()
    scaleTool = opensim.ScaleTool(curr_trial.opensim_scaling)
    ikTool = opensim.InverseKinematicsTool(curr_trial.opensim_inverse_kinematics)
    # Necessary for Table Processing
    ikTool.set_marker_file(curr_trial.opensim_marker_filtered)
    ikTool.set_output_motion_file(curr_trial.opensim_motion)
    # Run Scaling and Invkin Tools
    scaleTool.run()
    ikTool.run()
    ik_tool_to_csv(curr_trial=curr_trial)
    kinematic_analysis = opensim.Kinematics(curr_trial.opensim_analyze)
    if not os.path.exists(os.path.join(curr_trial.dir_trial, curr_trial.opensim_dir_analyze_results)):
        os.makedirs(os.path.join(curr_trial.dir_trial, curr_trial.opensim_dir_analyze_results))
    # Create analyze tool
    analyzetool = opensim.AnalyzeTool(curr_trial.opensim_analyze, True)
    analyzetool.setName(curr_trial.identifier)
    # Set Model file and motion file paths
    model_relpath = curr_trial.get_opensim_path(curr_trial.opensim_model_scaled)
    analyzetool.setModelFilename(model_relpath)
    analyzetool.setCoordinatesFileName(curr_trial.opensim_motion)
    analyzetool.run()
    from Pose2Sim_081_iDrink.Utilities import bodykin_from_mot_osim
    bodykin_csv = os.path.realpath(os.path.join(curr_trial.dir_kin_p2s,
                                                f"{curr_trial.get_filename()}_Body_kin_p2s_pos.csv"))
    if not os.path.exists(curr_trial.dir_kin_p2s):
        os.makedirs(curr_trial.dir_kin_p2s)
    bodykin_from_mot_osim.bodykin_from_mot_osim_func(os.path.join(curr_trial.dir_trial, curr_trial.opensim_motion),
                                                     os.path.join(curr_trial.dir_trial,
                                                                  curr_trial.opensim_model_scaled),
                                                     bodykin_csv)
    """if curr_trial.correct_skeleton:
        correct_skeleton_orientation(os.path.join(curr_trial.dir_trial, curr_trial.opensim_motion))"""  # TODO: Check if still needed
