# https://github.com/Frederic-vW/fitzhugh-nagumo-2d/blob/main/fhn2d.py
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Fith-Hugh Nagumo model on a 2D lattice
# FvW 03/2018

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def fhn2d(N, T, t0, dt, s, D, a, b, c, I0, stim, blocks):
    c1 = 1/c
    # initialize Fitz-Hugh Nagumo system
    v, w = np.zeros((N,N)), np.zeros((N,N))
    dv, dw = np.zeros((N,N)), np.zeros((N,N))
    sqrt_dt = np.sqrt(dt)
    X = np.zeros((T,N,N))
    X[0,:,:] = v
    #offset = 0 # Int(round(1*nt))
    # stimulation protocol
    I = np.zeros((t0+T,N,N))
    for st in stim:
        t_on, t_off = st[0]
        x0, x1 = st[1]
        y0, y1 = st[2]
        I[t0+t_on:t0+t_off, x0:x1, y0:y1] = I0
    # iterate
    for t in range(1, t0+T):
        if (t%100 == 0): print("    t = ", t, "\r", end="")
        # FHN equations
        dv = c1*(v - 1/3*v*v*v - w + I[t,:,:]) + D*L(v)
        dw = c*(v - a*w + b)
        # Ito stochastic integration
        v += (dv*dt + s*sqrt_dt*np.random.randn(N,N))
        w += (dw*dt)
        # dead block(s):
        for bl in blocks:
            #x0, x1 = bl[1]
            #y0, y1 = bl[2]
            v[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
            w[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
        if (t >= t0):
            X[t-t0,:,:] = v
    print("\n")
    return X


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


def L(x):
    # Laplace operator
    # periodic boundary conditions
    xU = np.roll(x, shift=-1, axis=0)
    xD = np.roll(x, shift=1, axis=0)
    xL = np.roll(x, shift=-1, axis=1)
    xR = np.roll(x, shift=1, axis=1)
    Lx = xU + xD + xL + xR - 4*x
    # non-periodic boundary conditions
    Lx[0,:] = 0.0
    Lx[-1,:] = 0.0
    Lx[:,0] = 0.0
    Lx[:,-1] = 0.0
    return Lx


def main():
    print("FitzHugh-Nagumo (FHN) lattice model\n")
    N = 128
    T = 1000
    t0 = 0
    dt = 0.1
    s = 0.02 # 0.02 # 0.10
    D = 1.0
    a = 0.5
    b = 0.7
    c = 0.3
    I = 1.0 # 1.0 # 0.5 # 1.0
    print("[+] Lattice size N: ", N)
    print("[+] Time steps T: ", T)
    print("[+] Warm-up steps t0: ", t0)
    print("[+] Integration time step dt: ", dt)
    print("[+] Noise std. dev.: ", s)
    print("[+] Diffusion coefficient D: ", D)
    print("[+] FHN parameter a: ", a)
    print("[+] FHN parameter b: ", b)
    print("[+] FHN parameter c: ", c)
    print("[+] Stimulation current I: ", I)
    # auxiliary variables
    n_2 = int(N/2) # 1/2 lattice size
    n_4 = int(N/4) # 1/4 lattice size
    n_5 = int(N/5) # 1/5 lattice size
    # stim protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
    stim = [ [[25,50], [1,N], [3,8]], [[130,150], [n_2-2,n_2+2], [10,25]] ]
    #stim = []
    # dead blocks, array of elementy [[x0,x1], [y0,y1]]
    blocks = [ [[2*n_4,3*n_4], [15,20]], [[2*n_4+10,3*n_4+10], [40,45]] ]
    # run simulation
    data = fhn2d(N, T, t0, dt, s, D, a, b, c, I, stim, blocks)
    print("[+] data dimensions: ", data.shape)

    # plot mean voltage
    m = np.mean(np.reshape(data, (T,N*N)), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(m, "-k")
    plt.show()

    # save data
    fname1 = f"fhn2d_I_{I:.4f}_s_{s:.4f}_D_{D:.4f}.npy"
    #np.save(fname1, data)
    #println("[+] Data saved as: ", fname1)
    fname2 = f"fhn2d_I_{I:.4f}_s_{s:.4f}_D_{D:.4f}.mp4"
    animate_video(fname2, data)
    print("[+] Data saved as: ", fname2)


if __name__ == "__main__":
    os.system("clear")
    main()