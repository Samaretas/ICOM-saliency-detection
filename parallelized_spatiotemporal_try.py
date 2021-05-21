import os
import time
import numpy as np
from cv2 import cv2
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing import Pipe
from multiprocessing import freeze_support
from multiprocessing import set_start_method
from MotionDetectionModule import MotionDetection
from DynamicSaliencyLib import *

if __name__ == '__main__':
    video_name = './examples/moving.m4v'
    maps_path = './saliency maps parallelized/'

    frames = read_video_stream(video_name)
    frames = resize_frames(frames, 256)
    n_frames = len(frames)

    print(f'read {n_frames} frames, of size {frames[0].shape}')
    if not os.path.exists(maps_path):
        os.makedirs(maps_path)

    # motion
    s_m, r_m = Pipe()
    p_m = Process(target=motion, args=(s_m, r_m))

    # orientation
    s_o, r_o = Pipe()
    p_o = Process(target=orientation, args=(s_o, r_o))

    # intensity
    s_i, r_i = Pipe()
    p_i = Process(target=intensity, args=(s_i, r_i))

    # color
    send1, recv1 = Pipe()
    c1 = Process(target=rg_color_space, args=(send1, recv1))
    send2, recv2 = Pipe()
    c2 = Process(target=by_color_space, args=(send2, recv2))

    p_m.start()
    p_o.start()
    p_i.start()
    c1.start()
    c2.start()

    start = end = 0
    c_start = c_end = 0
    second_prev_frame_bw = None
    prev_frame_bw = None

    starting_frame = 0

    for i in range(starting_frame, n_frames):
        print(f"video {video_name}, progress {int((i/n_frames)*100)}%")
        start = time.time_ns()

        frame_bw = get_intensity(frames[i])

        if(i > 0):
            prev = prev_frame_bw if second_prev_frame_bw is None else prev_frame_bw + \
                second_prev_frame_bw
            s_m.send((prev_frame_bw, frame_bw))
        s_i.send(frame_bw)
        s_o.send(frame_bw)

        # color
        c_start = time.time_ns()

        (b, g, r) = get_color_channels(frames[i])
        send1.send((b, g, r))
        send2.send((b, g, r))
        rg_color_space_maps = recv1.recv()
        by_color_space_maps = recv2.recv()
        color_conspicuity_map = get_conspicuity_from_color_spaces(
            rg_color_space_maps, by_color_space_maps, frame_bw.shape)
        c_conspicuity = normalize_map(color_conspicuity_map)

        c_end = time.time_ns()

        saliency_map = c_conspicuity

        saliency_map += r_i.recv()
        saliency_map += r_o.recv()
        n = 3

        if i > 0:
            n = 4
            saliency_map += r_m.recv()

        saliency_map = saliency_map/n

        second_prev_frame_bw = prev_frame_bw
        prev_frame_bw = frame_bw

        end = time.time_ns()


        saliency_image = np.zeros_like(saliency_map)
        saliency_image = simple_normalization(
            saliency_map, 255).astype('uint8')

        frame_name = str(i+1)
        while len(frame_name) < 4:
            frame_name = "0"+frame_name
        cv2.imwrite(maps_path+frame_name+'.jpg', saliency_image)

    p_m.kill()
    p_o.kill()
    p_i.kill()
    c1.kill()
    c2.kill()