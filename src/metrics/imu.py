# this code is an adaptation from https://github.com/siva82kb/smoothness_from_imu/blob/master/scripts/smoothness.py
# further smoothness metrics can be found in this repository
# Adaptation: using quaternions instead of gravity vector to rotate the acceleration data

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import quaternion # https://quaternion.readthedocs.io/en/latest/
import pandas as pd

def normalize_quaternion(quat):
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / norm

def filter_data(data, cutoff, fs, order, filter_type):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def rotate_acc_by_quat(acc, quat):
    rotated_acc = np.zeros_like(acc)
    for i in range(len(acc)):
        # Get (and optionally normalize) the quaternion
        q = quaternion.from_float_array(quat[i])
        #q = quaternion.as_float_array(q)/np.linalg.norm(quaternion.as_float_array(q))
        
        
        # Convert acceleration vector to quaternion with zero real part
        acc_quat = quaternion.from_float_array([0, *acc[i]])
        
        # Rotate acceleration: q * acc * q^(-1)
        rotated = q * acc_quat * q.inverse()
        
        # Extract the vector part
        rotated_acc[i] = quaternion.as_float_array(rotated)[1:]
    
    return rotated_acc


def vecnorm(data, order, axis):
    return np.linalg.norm(data, ord=order, axis=axis)

def get_ldlj(*args):
    if len(args) == 4 and args[1] == 'accel':
        mov = args[0]
        mov_type = args[1]
        SampleRate = args[2]
        quat = args[3]
    elif len(args) == 3 and args[1] == 'vel':
        mov = args[0]
        mov_type = args[1]
        SampleRate = args[2]
    else:
        raise ValueError("Input with 4 arguments must be: movement(accelerometer time series), "
                         "movement type ('accel'), Sample Rate, and quaternion time series. "
                         "Input with 3 arguments must be: movement(velocity time series), "
                         "movement type ('vel'), and Sample Rate")
    
    dt = 1.0 / SampleRate

    if mov_type == 'accel':
        quat = normalize_quaternion(quat)
        mov = filter_data(mov, fs_cut, SampleRate, 4, 'low')
        mov = rotate_acc_by_quat(mov, quat)
        mov = mov - np.array([0, 0, 9.81])
        mov = (mov - np.mean(mov, axis=0)) 
        acc = vecnorm(mov, 2, axis=1)
    elif mov_type == 'vel':
        mov = mov.T
        acc = np.diff(mov, axis=0) / dt
        acc = np.vstack([acc[0, :], acc])
        acc = filter_data(acc, 5, SampleRate, 4, 'low')
        acc = vecnorm(acc, 2, axis=1)


    jerk = vecnorm(np.diff(acc, axis=0), 2, axis=0) / dt
    mjerk = np.sum(jerk ** 2) * dt

    T = len(acc)
    mdur = T * dt

    mamp = np.max(vecnorm(acc, 2, axis=0)) ** 2

    ldlj = -np.log(mdur) + np.log(mamp) - np.log(mjerk)

    return ldlj

def calculate_smoothness(data_path: str):
    # read in data (affected or unaffected - the more negative the less smooth)
    imu_data = pd.read_csv(data_path)
    timestamp_diff = imu_data['timestamp'].diff().dropna()

    # Calculate the average time difference
    average_time_diff = timestamp_diff.mean()
    # Calculate the sample rate in Hz (samples per second)
    sample_rate = 1 / (average_time_diff / 1e9)  # Convert nanoseconds to seconds
    # Euler angles are in columns named 'euler_x', 'euler_y', 'euler_z'
    euler_angles = imu_data[['euler_x', 'euler_y', 'euler_z']].values

    # Convert Euler angles (in degrees) to quaternions
    # Adjust 'xyz' and degrees=True according to your data's specifics
    rotations = R.from_euler('xyz', euler_angles, degrees=True)
    quat = rotations.as_quat()  # Returns quaternions in the format [x, y, z, w]
    acc = imu_data[['accel_x', 'accel_y', 'accel_z']].values
    # set parameters
    SampleRate = sample_rate
    global fs_cut
    fs_cut = SampleRate/4 ; # cut-off frequency for low-pass filter -> this is set random we need to check 
    # Get the LDLJ
    ldlj = get_ldlj(acc, 'accel', SampleRate, quat)
    return ldlj


# print(calculate_smoothness('/home/arashsm79/amyproject/sub-4a2/ses-rest/motion/sub-4a2_ses-rest_task-calib_tracksys-imu_sensor-handLeft_motion.csv'))