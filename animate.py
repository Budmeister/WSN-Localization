# https://github.com/Frederic-vW/fitzhugh-nagumo-2d/blob/main/fhn2d.py

import cv2
import numpy as np


def animate_video(fname, x):
    # BW
    y = 255 * (x-x.min()) / (x.max()-x.min())
    # BW inverted
    #y = 255 * ( 1 - (x-x.min()) / (x.max()-x.min()) )
    y = y.astype(np.uint8)
    nt, nx, ny = x.shape
    print(f"nt = {nt:d}, nx = {nx:d}, ny = {ny:d}")
    # write video using opencv
    frate = 30
    out = cv2.VideoWriter(fname, \
                          cv2.VideoWriter_fourcc(*'mp4v'), \
                          frate, (nx,ny))
    for i in range(0,nt):
        print(f"i = {i:d}/{nt:d}\r", end="")
        img = np.ones((nx, ny, 3), dtype=np.uint8)
        for j in range(3): img[:,:,j] = y[i,::-1,:]
        out.write(img)
    out.release()
    print("")
