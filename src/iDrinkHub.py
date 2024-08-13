import glob
import os
import sys
import time
import re
import shutil
from tqdm import tqdm

import argparse
import pandas as pd

from iDrink import iDrinkTrial, iDrinkPoseEstimation, iDrinkVisualInput, iDrinkOpenSim

from Pose2Sim import Pose2Sim

"""
This File runs the iDrink Pipeline for the Demo for September 2024.

It gets 
"""
dir_root = os.path.realpath(r"C:\iDrink_Demo")
dir_root_recs = os.path.join(dir_root, r"bids_root_folder")
dir_mot_out = None

dir_session_data = os.path.join(dir_root, "session_data")

def move_files_to_bids(trial, dir_mot_out):
    """
    Moves .mov and .sto files to the movement folder in the bids-tree


    :param trial:
    :return:
    """

    dir_trial = trial.dir_trial


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

def run(id_s, id_p, task='drink'):


    """Get video files, calibration_file and events file"""
    dir_measurement = os.path.join(dir_root_recs, f"sub-{id_p}", f"ses-{id_s}")
    dir_mot_out = os.path.join(dir_measurement, "motion")

    calib_file = glob.glob(os.path.join(dir_measurement, "**",  "*calibration.toml"))[0]
    video_files = glob.glob(os.path.join(dir_measurement, "video", f"*task-{task}*.mp4"))
    cams = [re.search(r"cam\d", video)[0] for video in video_files]

    df_events = pd.read_csv(glob.glob(os.path.join(dir_measurement, "video", f"*task-{task}_events.csv"))[0])

    """Create trial Objects and Folder Structure for Pipeline"""
    trials = create_trials(id_s, id_p, df_events, task, calib_file)




    for trial in trials:
        #Cut recording down to individual trials

        videos = iDrinkVisualInput.cut_videos_to_trials(video_files, trial)


        """Run Pipeline for each trial"""
        trial.config_dict["pose"]["videos"] = videos
        trial.config_dict["pose"]["cams"] = cams

        trial.config_dict.get("project").update({"project_dir": trial.dir_trial})
        trial.config_dict['pose']['pose_framework'] = trial.used_framework
        trial.config_dict['pose']['pose_model'] = trial.pose_model

        trial.run_pose2sim()

        trial.prepare_opensim()

        iDrinkOpenSim.open_sim_pipeline(trial)

        move_files_to_bids(trial, dir_mot_out)

        # TODO: Add that somewhere before Pose Estimation



    pass


if __name__ == '__main__':
    # Parse command line arguments

    if sys.gettrace() is not None:
        print("Debug Mode is activated\n"
              "Starting debugging script.")

        id_s = "rest"
        id_p = "4a2"
        task = "drink"

    run(id_s, id_p, task)
