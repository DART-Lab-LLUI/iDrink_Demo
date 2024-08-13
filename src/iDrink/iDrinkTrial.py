import glob
import os
import re

import numpy as np
import pandas as pd

"""
Class for Trial objects.

Trial objects should contain the following informations:
- id_s: Session ID
- id_p: Patient ID (PID)
- id_t: Trial ID

- date_time: Date and Time of Recording e.g. 240430_1430

- path_config: Path to Config.toml

Information for Therapists after Recording:pip
- skipped: "True" / "False" - flag that tells whether step / trial was skipped
- skip_reason: Explanation why the trial was skipped
- Score: Score assessed by therapist

Paths to directories:
- dir_trial: Trial directory
- dir_default: directory containing all default files
- dir_ref: Directory containing reference files

Information about Recording
- Frame Rate: 30 or 60 FPS

About the used model
- used framework: e.g. MMPose, OpenPose etc.
- pose_model: e.g. B25, B25_UPPER_EARS


For Movement Analysis
- measured_side: "left" / "right"
- Affected: "True" / "False" --> tells whether the measured side is the affected side of the patient

- ang_joint_kin: angular joint kinematics
- ep_kin_movement_time: endpoint kinematics - movement Time
- ep_kin_movement_units: endpoint kinematics - movement units
- ep_kin_peak_velocities: endpoint kinematics - peak velocity

- path_murphy_measures: path to directory containing the murphy measure data
- path_ang_joint_kin: path to angular joint kinematics
- path_ep_kin_movement_time: path to endpoint kinematics - movement Time
- path_ep_kin_movement_units: path to endpoint kinematics - movement units
- path_ep_kin_peak_velocities: path to endpoint kinematics - peak velocity

For Opensim
- opensim_model: Path to the .osim file
- opensim_model_scaled: Path to the scaled .osim file

- opensim_scaling: path to the scaling .xml
- opensim_inverse_kinematics. path to the inverse kinematics .xml file
- opensim_analyze: Path to Analyze Tool .xml

- opensim_marker: relative path to .trc file
- opensim_marker_filtered: relative path to filtered .trc file
- opensim_marker_scaling: path to filtered scaling .trc files
- opensim_motion: relative path to .mot file

- opensim_dir_analyze_results: Path to the Analyze Results directory

- opensim_scaling_time_range: time range for scaling (string) e.g. "0.017 0.167"
- opensim_IK_time_range: time range for the Inverse Kinematics (string) e.g. "0 6.65"
- opensim_ana_init_t: Start time of for analyze tool (string) e.g. "0"
- opensim_ana_final_t: end time of for analyze tool (string) e.g. "6.65"

"""


class Trial:
    def __init__(self, identifier=None, id_s=None, id_p=None, id_t=None,
                 onset=None, offset=None, assessement="", task="",
                 dir_root=None, dir_default=None, dir_reference=None, dir_session=None, dir_calib=None,
                 dir_calib_videos=None, dir_calib_files=None, dir_participant=None, dir_trial=None,
                 path_config=None, skip=False, skip_reason=None, affected=False,
                 frame_rate=60, rec_resolution=(1920, 1080), clean_video=True,
                 used_cams=None, video_files=None, path_calib_videos=None,

                 config_dict=None, path_calib=None, calib=None, used_framework=None, pose_model=None, measured_side=None,

                 stabilize_hip=True, correct_skeleton=False, chosen_components=None, show_confidence_intervall=False,
                 use_analyze_tool=False, bod_kin_p2s=False, use_torso=True, is_reference=False):


        """Variables for Deployment"""
        # Trial Info
        self.id_s = id_s
        self.id_p = id_p
        self.id_t = id_t
        self.identifier = identifier

        self.onset = onset
        self.offset = offset

        # Directories and Paths
        self.dir_trial = dir_trial
        self.dir_participant = dir_participant
        self.dir_session = dir_session

        self.dir_root = dir_root
        self.dir_default = os.path.join(dir_root, "default_files")
        self.dir_calib = dir_calib
        self.dir_calib_videos = dir_calib_videos
        self.dir_calib_files = dir_calib_files



        # Clinical info for review
        self.assessement = assessement  # Name of the assessement e.g. Drinking Task, iArat etc.
        self.task = task  # The executed Task
        self.measured_side = measured_side
        self.affected = affected
        self.is_reference = is_reference

        # Configurations, Calibrations, Settings
        self.path_config = path_config
        self.config_dict = config_dict
        self.path_calib = path_calib
        self.calib = calib

        # Data preparation and filtering
        self.stabilize_hip = stabilize_hip

        # Recording
        self.used_cams = used_cams  # List of used Cameras
        self.dir_recordings = os.path.realpath(os.path.join(dir_trial, "videos", "recordings"))
        self.dir_rec_pose = os.path.realpath(os.path.join(dir_trial, "videos", "pose"))
        self.dir_rec_blurred = os.path.realpath(os.path.join(dir_trial, "videos", "blurred"))
        self.frame_rate = frame_rate
        self.rec_resolution = rec_resolution
        self.clean_video = clean_video

        self.video_files = video_files  # List of Recordings if not in dir_recordings (Move them to dir_recordings before running the pipeline)
        self.path_calib_videos = path_calib_videos  # List of Calibration Videos if not in dir_calib_videos (Move them to dir_calib_videos before running the pipeline)


        # Visual Output
        self.render_out = os.path.join(dir_trial, "videos", 'Render_out')

        # Blender


        # Plots
        self.filteredplots = True # Data filtered before plotted
        self.chosen_components = chosen_components
        self.show_confidence_intervall = show_confidence_intervall


        # Pose Estimation
        self.PE_dim = 2  # Dimension of Pose Estimation
        self.used_framework = "mmpose" # openpose, mmpose, pose2sim
        self.pose_model = pose_model
        self.write_pose_videos = False

        # Movement Analysis
        # Settings
        self.use_analyze_tool = use_analyze_tool
        self.bod_kin_p2s = bod_kin_p2s
        self.use_torso = use_torso
        self.filenename_appendix = ''

        # Setting for Phase Detection
        self.use_dist_handface = False
        self.use_acceleration = False
        self.use_joint_vel = False
        self.extended = True # Decide whether extended Phase detection should be used (7 or 5 phases)

        # Thresholds TODO: User should be able to set the threshold
        self.phase_thresh_vel = 0.05  # Fraction of peak that needs to be passed for next Phase
        self.phase_thresh_pos = 0.05  # Fraction of peak that needs to be passed for next Phase

        # Filtering
        self.butterworth_cutoff = 10
        self.butterworth_order = 5

        # Smoothing TODO: Let User choose, which data to smoothen
        self.smooth_velocity = True
        self.smooth_distance = True
        self.smooth_trunk_displacement = False

        self.smoothing_divisor_vel = 4
        self.smoothing_divisor_dist = 4
        self.smoothing_divisor_trunk_displacement = 4
        self.reference_frame_trunk_displacement = 0

        # Directories
        # Dir containing all fies and subfolders for movement analysis
        self.dir_movement_analysis = os.path.realpath(os.path.join(self.dir_trial, "movement_analysis"))
        # Dir containing files for murphy measures
        self.dir_murphy_measures = os.path.realpath(os.path.join(self.dir_movement_analysis, "murphy_measures"))
        # Directory containing Kinematics based on raw .trc files
        self.dir_kin_trc = os.path.realpath(os.path.join(self.dir_movement_analysis, "kin_trc"))
        # Directory containing kinematics based on inverse Kinematics from P2S Function
        self.dir_kin_p2s = os.path.realpath(os.path.join(self.dir_movement_analysis, "kin_p2s"))
        # Dir containing kinematics based on the inverse kinematics tool from OpenSim
        self.dir_kin_ik_tool = os.path.realpath(os.path.join(self.dir_movement_analysis, "ik_tool"))
        # Dir containing kinematics calculated by OpenSim Analyzer Tool
        self.dir_anatool_results = os.path.realpath(os.path.join(self.dir_movement_analysis, "kin_opensim_analyzetool"))

        # Paths to Movement files
        self.path_opensim_ik = None

        # Opensim Analyze Tool
        self.path_opensim_ana_pos = None
        self.path_opensim_ana_vel = None
        self.path_opensim_ana_acc = None
        self.path_opensim_ana_ang_pos = None
        self.path_opensim_ana_ang_vel = None
        self.path_opensim_ana_ang_acc = None

        # Pose2Sim Inverse Kinematics
        self.path_p2s_ik_pos = None
        self.path_p2s_ik_vel = None
        self.path_p2s_ik_acc = None

        # Pose2Sim .trc files
        self.path_trc_pos = None
        self.path_trc_vel = None
        self.path_trc_acc = None

        # Movement Data
        # OpenSim Inverse Kinematics Tool
        self.opensim_ik = None
        self.opensim_ik_ang_pos = None
        self.opensim_ik_ang_vel = None
        self.opensim_ik_ang_acc = None

        # Opensim Analyze Tool
        self.opensim_ana_pos = None
        self.opensim_ana_vel = None
        self.opensim_ana_acc = None
        self.opensim_ana_ang_pos = None
        self.opensim_ana_ang_vel = None
        self.opensim_ana_ang_acc = None

        # Pose2Sim Inverse Kinematics
        self.p2s_ik_pos = None
        self.p2s_ik_vel = None
        self.p2s_ik_acc = None

        # Pose2Sim .trc files
        self.trc_pos = None
        self.trc_vel = None
        self.trc_acc = None

        # Movement Data used for analysis
        # Marker/Keypoint Position, Velocity and Acceleration
        self.marker_pos = None
        self.marker_vel = None
        self.marker_acc = None
        self.marker_source = None

        # joint Position, Velocity and Acceleration
        self.joint_pos = None
        self.joint_vel = None
        self.joint_acc = None
        self.joint_source = None

        # Murphy Measures - File Paths
        self.path_ang_joint_kin = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ang_joint_kin.csv"))
        self.path_ep_kin_movement_time = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_movement_time.csv"))
        self.path_ep_kin_movement_units = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_movement_units.csv"))
        self.path_ep_kin_peak_velocities = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_ep_kin_peak_velocities.csv"))
        self.path_mov_phases_timeframe = os.path.realpath(os.path.join(self.dir_murphy_measures, f"{self.identifier}_mov_phases_timeframe.csv"))

        # Murphy Measures - Data
        self.mov_phases_timeframe = None
        self.ang_joint_kin = None
        self.ep_kin_movement_time = None
        self.ep_kin_movement_units = None
        self.ep_kin_peak_velocities = None

        # OpenSim files and paths
        self.opensim_model = None
        self.opensim_model_scaled = None
        self.opensim_scaling = None
        self.opensim_inverse_kinematics = None
        self.opensim_analyze = None
        self.opensim_marker = None
        self.opensim_marker_filtered = None
        self.opensim_marker_scaling = None
        self.opensim_motion = None
        self.opensim_scaling_time_range = None
        self.opensim_IK_time_range = None
        self.opensim_ana_init_t = None
        self.opensim_ana_final_t = None
        self.opensim_dir_analyze_results = self.get_opensim_path(
            os.path.join(self.dir_movement_analysis, "kin_opensim_analyzetool"))

    def create_trial(self):
        """
        This function creates all the folders and their subfolders.
        """
        import shutil
        from .iDrinkSetup import write_default_configuration
        """Create empty Folder structure"""
        dirs = [
            self.dir_murphy_measures,
            self.dir_anatool_results,
            self.dir_kin_trc,
            self.dir_kin_p2s,
            self.dir_kin_ik_tool,
            self.dir_recordings,
            self.dir_rec_blurred,
            self.render_out,
            os.path.realpath(os.path.join(self.dir_trial, "pose")),
            os.path.realpath(os.path.join(self.dir_trial, "pose-3d")),
            os.path.realpath(os.path.join(self.dir_trial, "pose-associated")),
        ]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

        # Copy Geometry from default to trial folder
        dir_geom = os.path.realpath(os.path.join(self.dir_default, "Geometry"))
        new_dir_geom = os.path.realpath(os.path.join(self.dir_trial, "Geometry"))

        shutil.copytree(dir_geom, new_dir_geom, dirs_exist_ok=True)

        """Place empty config file"""

        empty_file = os.path.join(self.dir_default, "Config_empty.toml")
        if self.path_config is None:
            self.path_config = os.path.realpath(os.path.join(self.dir_trial, f"{self.identifier}_Config.toml"))

        write_default_configuration(empty_file, self.path_config)

    def check_if_affected(self):
        recnames = glob.glob(os.path.join(self.dir_recordings, "*unaffected*"))

        if recnames:
            self.affected = False
        else:
            self.affected = True

        return self.affected




    def load_configuration(self, load_default=False):
        from .iDrinkSetup import write_default_configuration

        def get_config():
            with open(self.path_config, 'r') as file:
                # Assuming TOML format, import the required module if necessary
                import toml
                self.config_dict = toml.load(file)


        try:
            if not load_default:
                get_config()
            else:
                empty_file = os.path.join(self.dir_default, "Config_empty.toml")
                self.path_config = os.path.realpath(os.path.join(self.dir_trial, f"{self.identifier}_Config.toml"))

                write_default_configuration(empty_file, self.path_config)
                get_config()

        except FileNotFoundError as e:
            print(f"Configuration file not found: {e}")
            print(f"Loading default configuration file.")

            empty_file = os.path.join(self.dir_default, "Config_empty.toml")
            self.path_config = os.path.realpath(os.path.join(self.dir_trial, f"{self.identifier}_Config.toml"))

            write_default_configuration(empty_file, self.path_config)
            get_config()
        except Exception as e:
            print(f"Another error occurred: {e}")
            print(f"Loading default configuration file.")

            empty_file = os.path.join(self.dir_default, "Config_empty.toml")
            self.path_config = os.path.realpath(os.path.join(self.dir_trial, f"{self.identifier}_Config.toml"))

            write_default_configuration(empty_file, self.path_config)
            get_config()

        return self.config_dict

    def save_configuration(self):
        """Saves config_digt to path in self.path_config"""
        import toml

        with open(self.path_config, 'w') as file:

            toml.dump(self.config_dict, file)
        return self.config_dict

    def load_calibration(self):
        with open(self.path_calib, 'r') as file:
            # Assuming TOML format, import the required module if necessary
            import toml
            self.calib = toml.load(file)
        return self.calib

    def get_time_range(self, path_trc_file, start_time=0, frame_range=[], as_string=False):
        """
        Input:
            - p2S_file: path to .trc file
            - start_time: default = 0
            - frame_range: [starting frame, ending frame] default = []
        Output:
            - time_range formatted for OpenSim .xml files.
        Gets time range based on Pose2Sim .trc file.
        """
        from trc import TRCData
        path_trc_file = os.path.realpath(os.path.join(self.dir_trial, path_trc_file))

        trc = TRCData()
        trc.load(filename=path_trc_file)
        time = np.array(trc["Time"])
        # If Frame Range is given, the time range is written using the frame range.
        if frame_range:
            frame = np.array(trc["Frame#"])
            if frame_range[0] == 0:
                frame_range[0] = 1
            if frame_range[1] > max(frame):
                frame_range[1] = max(frame)
            start_time = time[np.where(frame == frame_range[0])[0][0]]
            final_time = time[np.where(frame == frame_range[1])[0][0]]
            if as_string:
                return f"{start_time} {final_time}"
        # Otherwise, it will end at the end of the recording
        else:
            final_time = max(time)
            if as_string:
                return f"{start_time} {final_time}"
        # if as_sting is False, return list with start and end time
        return [start_time, final_time]

    def get_opensim_path(self, path_in):

        # TODO: Make this part more general. At the moment it is a bit too focused on OpenSim

        if self.dir_trial in path_in:
            filepath = path_in.split(self.dir_trial + "\\", 1)[1]
        else:
            return "dir_trial not in path"

        return filepath

    def get_filename(self):
        # Setup Appendix for Filename if needed
        ref = ""
        aff = ""
        if self.is_reference:
            ref = "_reference"
        if self.affected:
            aff = "_affected"
        filename = f"{self.identifier}{self.filenename_appendix}{aff}{ref}"

        return filename

    def find_file(self, directory, extension, flag=None):
        import glob
        if flag is not None:
            pattern = os.path.join(directory, f"{self.id_s}_*{self.id_p}_*{self.id_t}*{flag}*{extension}")
        else:
            pattern = os.path.join(directory, f"{self.id_s}_*{self.id_p}_*{self.id_t}*{extension}")

        try:
            filepath = glob.glob(pattern)[0]
        except Exception as e:
            print(e,"\n")
            print(f"File not found in {directory} with pattern {pattern}\n")

        return filepath

    def prepare_opensim(self):


        self.opensim_model = os.path.join(self.dir_default, f"iDrink_{self.pose_model}.osim")
        self.opensim_model_scaled = os.path.join(self.dir_trial, f"Scaled_{self.pose_model}.osim")

        self.opensim_scaling = os.path.join(self.dir_trial, f"Scaling_Setup_iDrink_{self.pose_model}.xml")
        self.opensim_inverse_kinematics = os.path.join(self.dir_trial, f"IK_Setup_iDrink_{self.pose_model}.xml")
        self.opensim_analyze = os.path.join(self.dir_trial, f"AT_Setup.xml")

        self.opensim_marker = self.get_opensim_path(self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc"))
        self.opensim_marker_filtered = self.get_opensim_path(
            self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc", flag="filt"))
        self.opensim_motion = os.path.splitext(
            self.get_opensim_path(self.find_file(os.path.join(self.dir_trial, "pose-3d"), ".trc", flag="filt")))[
                                  0] + ".mot"

        self.opensim_scaling_time_range = self.get_time_range(path_trc_file=self.opensim_marker_filtered,
                                                              frame_range=[0, 1], as_string=True)
        self.opensim_IK_time_range = self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=True)
        self.opensim_ana_init_t = str(
            self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=False)[0])
        self.opensim_ana_final_t = str(
            self.get_time_range(path_trc_file=self.opensim_marker_filtered, as_string=False)[1])



    def run_pose2sim(self):
        """Run Pose2Sim Pipeline"""
        from Pose2Sim import Pose2Sim
        # os.chdir(self.dir_trial)
        self.config_dict.get("project").update({"project_dir": self.dir_trial})
        self.config_dict['pose']['pose_framework'] = self.used_framework
        self.config_dict['pose']['pose_model'] = self.pose_model




        """self.config_dict['triangulation']['reproj_error_threshold_triangulation'] = 200
        self.config_dict['triangulation']['interp_if_gap_smaller_than'] = 400"""

        Pose2Sim.calibration(config=self.config_dict)


        # Change the config_dict so that it uses the correct skeleton
        if self.pose_model == "Coco17_UpperBody":
            self.config_dict['pose']['pose_model'] = 'COCO_17'

        Pose2Sim.poseEstimation(config=self.config_dict)
            
        self.config_dict['pose']['pose_model'] = self.pose_model

        Pose2Sim.synchronization(config=self.config_dict)
        Pose2Sim.personAssociation(config=self.config_dict)
        Pose2Sim.triangulation(config=self.config_dict)
        Pose2Sim.filtering(config=self.config_dict)
        if self.pose_model in ['BODY_25', 'BODY_25B']:
            print("Model supported for Marker augmentation.\n"
                  "Starting augmentation:")
            Pose2Sim.markerAugmentation(config=self.config_dict)
        else:
            print('Marker augmentation is only supported with OpenPose BODY_25 and BODY_25B models.\n'
                  'Augmentation will be skipped.')


