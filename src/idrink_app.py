#!/usr/bin/env python3

import glob
import os
import sys
import time
import re
import shutil
from pathlib import Path
from tqdm import tqdm

import argparse
import pandas as pd

from iDrink import iDrinkTrial, iDrinkVisualInput, iDrinkOpenSim
from metrics.Metrics_plotting import generate_plots

dir_root = None
dir_root_recs = None
dir_mot_out = None
dir_session_data = None


def move_files_to_bids(trial, dir_mot_out):
    """
    Moves .mov and .sto files to the movement folder in the bids-tree


    :param trial:
    :return:
    """

    path_mot = glob.glob(os.path.join(trial.dir_trial, "pose-3d", "*.mot"))[0]

    shutil.copy2(path_mot, dir_mot_out)
    shutil.copy2(trial.path_opensim_ana_pos, dir_mot_out)
    shutil.copy2(trial.path_opensim_ana_vel, dir_mot_out)
    shutil.copy2(trial.path_opensim_ana_acc, dir_mot_out)
    shutil.copy2(trial.path_opensim_ana_ang_pos, dir_mot_out)
    shutil.copy2(trial.path_opensim_ana_ang_vel, dir_mot_out)
    shutil.copy2(trial.path_opensim_ana_ang_acc, dir_mot_out)



def create_trials(id_s, id_p, df_events, task, calib_file):
    """
    Create Trial objects from the events dataframe.

    :param id_s: session_ID
    :param id_p: participant_ID
    :param df_events: Dataframe with events - onset, offset, trial_id
    :return trials: List of iDrinkTrial objects
    """
    trials = []
    n_trials = df_events.shape[0]
    id_s = f"S{id_s}"
    id_p = f"P{id_p}"

    for i in range(n_trials):
        onset = df_events.loc[i, "onset"]
        offset = df_events.loc[i, "offset"]
        trial_id = df_events.loc[i, "trial_id"]
        id_t = f"T{trial_id:03d}"

        identifier = f"{id_s}_{id_p}_{id_t}"

        dir_session = os.path.join(dir_session_data, id_s)
        dir_calib = os.path.join(dir_session, f"{id_s}_{id_p}_Calibration")
        dir_participant = os.path.join(dir_session, f"{id_s}_{id_p}")
        dir_trial = os.path.join(dir_participant, f"{id_s}_{id_p}_{id_t}")

        calib_file_new = os.path.join(dir_calib, os.path.basename(calib_file))

        trial = iDrinkTrial.Trial(identifier=identifier, id_s=id_s, id_p=id_p, id_t = trial_id,
                                  onset=onset, offset=offset, assessement="Drinking-Task", task=task,
                                  dir_trial=dir_trial, dir_participant=dir_participant, dir_session=dir_session,
                                  dir_root=dir_root, used_framework='pose2sim', pose_model="Coco17_UpperBody",
                                  path_calib=calib_file_new)

        trial.create_trial()
        trial.load_configuration()

        trial.config_dict.get("project").update({"project_dir": trial.dir_trial})
        trial.config_dict['pose']['pose_framework'] = trial.used_framework
        trial.config_dict['pose']['pose_model'] = trial.pose_model

        trial.save_configuration()

        if not os.path.exists(dir_calib):
            os.makedirs(dir_calib, exist_ok=True)
        shutil.copy2(calib_file, dir_calib)

        trials.append(trial)

    return trials

def run(id_s, id_p, task='drink', stabilize_hip=True, correct_skeleton=False):
    """
    Runs the iDrink Pipeline for the Demo in September 2024.

    The Demo uses the Pose Estimation implemented by Pose2Sim.

    :param id_s: Session ID e.g. "rest" or "13082024-1428"
    :param id_p: Patient ID e.g. "4a2" or "5051234"
    :param task: Task e.g. "drink"
    :param stabilize_hip: Boolean whether hip should be stabilized using the mean after inverse Kinematics
    :param correct_skeleton: Boolean whether the skeleton should be rotated in the global coordinate System to sit upright.
    :return:
    """


    """Get video files, calibration_file and events file"""
    global dir_measurement
    dir_measurement = os.path.join(dir_root_recs, f"sub-{id_p}", f"ses-{id_s}")

    global dir_mot_out
    dir_mot_out = os.path.join(dir_measurement, "motion")

    calib_file = glob.glob(os.path.join(dir_measurement, "**",  "*calibration.toml"))[0]
    video_files = glob.glob(os.path.join(dir_measurement, "video", f"*task-{task}*.mp4"))
    cams = [re.search(r"cam\d", video)[0] for video in video_files]

    df_events = pd.read_csv(glob.glob(os.path.join(dir_measurement, "video", f"*task-{task}_events.csv"))[0])

    """Create trial Objects and Folder Structure for Pipeline"""
    trials = create_trials(id_s, id_p, df_events, task, calib_file)




    for trial in trials:
        """Cut recording down to individual trials"""
        videos = iDrinkVisualInput.cut_videos_to_trials(video_files, trial)

        """Run Pipeline for each trial"""
        trial.config_dict["pose"]["videos"] = videos
        trial.config_dict["pose"]["cams"] = cams

        trial.config_dict.get("project").update({"project_dir": trial.dir_trial})
        trial.config_dict['pose']['pose_framework'] = trial.used_framework
        trial.config_dict['pose']['pose_model'] = trial.pose_model

        trial.stabilize_hip = stabilize_hip
        trial.correct_skeleton = correct_skeleton

        trial.run_pose2sim()
        trial.prepare_opensim()

        iDrinkOpenSim.open_sim_pipeline(trial)
        move_files_to_bids(trial, dir_mot_out)
    
    # generate plots
    generate_plots(dir_mot_out, Path(dir_mot_out).parent / 'metric')


def main():
    parser = argparse.ArgumentParser(description='Run iDrink Backend for the Demo in September 2024')
    parser.add_argument('--dir_root', metavar='dr', type=str, help='Path to the root folder of the BackEnd')
    parser.add_argument('--bids_root', metavar='br', type=str, help='Path to the BIDS root folder')
    parser.add_argument('--id_s', metavar='ids', type=str, help='Session ID, e.g. rest')
    parser.add_argument('--id_p', metavar='idp', type=str, help='Patient ID, e.g. 4a2')
    parser.add_argument('--task', metavar='t', type=str, default='drink', help='Task to be analyzed default: drink')
    parser.add_argument('--DEBUG', action='store_true', default=False, help='Debug mode - specific folder structure necessary')

    args = parser.parse_args()

    global dir_root
    dir_root = args.dir_root
    
    global dir_root_recs
    dir_root_recs = args.bids_root

    global dir_session_data
    dir_session_data = os.path.join(dir_root, "session_data")

    id_s = args.id_s
    id_p = args.id_p
    task = args.task

    run(id_s=id_s, id_p=id_p, task=task)


if __name__ == '__main__':
    main()
