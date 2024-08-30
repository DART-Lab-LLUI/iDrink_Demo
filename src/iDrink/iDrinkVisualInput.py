# Import Libraries
import os
import re

import cv2
import numpy as np

def cut_videos_to_trials(video_files, trial):
    """
    Cuts the recorded videos into the individual trials.
    :param video_files: List of video files
    :param trial: Trial Object
    :return:

    Iterates through all the videos and only uses frames of the given trial.

    The video files are then stored in the recordings folder of the trial.

    If clean_video True, the video is checked for frame duplicates.
    """
    import queue
    from tqdm import tqdm
    from threading import Thread

    def frame_loader_cut(cap, frame_start, frame_end, frame_queue, ret_queue):
        
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start) this is not accurate https://github.com/opencv/opencv/issues/9053
        curr = 0
        while True:
            if curr == frame_start:
                break
            ret, frame = cap.read()
            curr += 1

        frame_count = frame_start
        while True:

            ret, frame = cap.read()
            if not ret or frame_count >= frame_end:
                break

            ret_queue.put(ret)
            frame_queue.put(frame)
            frame_count += 1

        ret_queue.put(None)
        frame_queue.put(None)


    video_files_new = []
    for video in video_files:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print("Error opening file.")
            return
        frame_queue = queue.Queue()
        ret_queue = queue.Queue()

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        match = re.search(r"cam\d+", video)
        if match:
            # If a match is found, extract the matched string and add it to the cam_names list
            cam = match.group()
        else:
            cam = "cam_unknown"
        file_extension = os.path.splitext(video)[1]

        video_new = os.path.realpath(os.path.join(trial.dir_recordings, f'{trial.identifier}_{cam}{file_extension}'))

        # Retrieve start and end frame of the trial
        frame_start = trial.onset
        frame_end = trial.offset
        total_frames_of_trial = int(frame_end - frame_start)

        thread_frame = Thread(target=frame_loader_cut, args=(cap, frame_start, frame_end, frame_queue, ret_queue))
        thread_frame.start()

        video_files_new.append(video_new)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(video_new, fourcc, fps, (width, height))

        pbar = tqdm(total=total_frames_of_trial, desc=f"Processing {trial.identifier} - {os.path.basename(video_new)}", unit="frame")

        # Run and write frames until last frame
        while True:
            ret = ret_queue.get()
            frame = frame_queue.get()
            if not ret:
                break

            out.write(frame)
            pbar.update(1)

        pbar.close()
        thread_frame.join()
        cap.release()
        out.release()

    return video_files_new
