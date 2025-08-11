#!/usr/bin/env python3

import cv2
import time
import sys
from triton_bringup.omnicv import fisheyeImgConv
from tqdm import tqdm

input = '/home/john/Desktop/triton_footage/part2_equirectangular.mp4'
output = '/home/john/Desktop/triton_footage/part2_bottom.mp4'

mapper = fisheyeImgConv()
FOV = 150
Theta = -10
Phi = -90
Hd = 1600
Wd = 1600

cap = cv2.VideoCapture(input)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', etc.
out = cv2.VideoWriter(output, fourcc, fps, (Wd, Hd))

for i in tqdm(range(frame_count), desc="Processing Frames"):
    ret, frame_bgr = cap.read()
    if not ret:
        print(f"Warning: Could not read frame {i+1}/{frame_count}. Ending early.")
        break

    bottom_frame = mapper.eqruirect2persp(frame_bgr, FOV, Theta, Phi, Hd, Wd)
    out.write(bottom_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
    

    
