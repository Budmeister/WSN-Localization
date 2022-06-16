from pde import *
import numpy as np
import os
import json
from fhn2d import animate_video


def save_params(params, dir):
    with open(dir, "w") as file:
        json.dump(params, file)

class FHN(PDEBase):
    def __init__(
        self, N, n, T, dt, D, a, b, c, I0, stim, 
        bc="auto_periodic_neumann", w_h=1, w_l=1
    ):
        self.N = N
        self.n = n
        self.T = T
        self.dt = dt
        self.D = D
        self.a = a
        self.b = b
        self.c = c
        self.I = I0
        self.bc = bc
        self.I = np.zeros((T,n,n))
        for st in stim:
            t_on, t_off = st[0]
            x0, x1 = st[1]
            y0, y1 = st[2]
            self.I[t_on:t_off, x0:x1, y0:y1] = I0
        self.w_h = w_h
        self.w_l = w_l

    # def make_modify_after_step(self, state: FieldBase):
    #     def modify_after_step(state_data: np.ndarray, t=None) -> float:
    #         """no-op function"""
    #         return 0

    #     return modify_after_step

    def evolution_rate(self, state, t=0):
        v, w = state

        dv = (v - v ** 3 / 3 - w + self.I[int(t / self.dt), :, :]) / self.c + self.D * v.laplace(bc=self.bc)
        dw = self.c * (v - self.a*w + self.b) * ((self.w_h - self.w_l) / (1 + np.exp(-4 * v)) + self.w_l)
        return FieldCollection([dv, dw])

def run_FN2():
    T = 1000
    t0 = 0
    dt = 0.1
    dx = 0.6
    s = 0.02 # 0.02 # 0.10

    N = 128
    n = int(N / dx)
    D = 0.5
    a = 0.5
    b = 0.7
    c = 0.3
    I = 1.0 # 1.0 # 0.5 # 1.0
    w_h = 1
    w_l = 1
    stim = [ [[25, 40], [n // 2 - 5, n // 2 + 5], [n // 2 - 5, n // 2 + 5]] ]
    
    other_params = {
        "dx": dx
    }
    params = {
        "N": N,
        "n": n,
        "T": T,
        "dt": dt,
        "D": D,
        "a": a,
        "b": b,
        "c": c,
        "I0": I,
        # stim protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
        "stim": stim,

        "w_h": w_h,
        "w_l": w_l
    }
    eq = FHN(**params)

    def get_initial_state():
        # grid = UnitGrid((N, N))
        shape = (n, n)
        grid = CartesianGrid([(0, N), (0, N)], shape)
        data_v = np.zeros(shape)
        data_w = np.zeros(shape)
        # data_v[N // 2, N // 2] = I
        v0 = ScalarField(grid, data_v)
        w0 = ScalarField(grid, data_w)
        return grid, FieldCollection((v0, w0))

    grid, vw0 = get_initial_state()
    memory_storage = MemoryStorage()
    result = eq.solve(vw0, t_range=T * dt, dt=dt, tracker=[memory_storage.tracker(dt)])#, method=ExplicitTModSolver, adaptive=False)

    # Plotting
    result.plot()
    plot_kymographs(memory_storage)
    data = memory_storage.data
    data = np.array(data)
    vs = data[:, 0, :]
    ws = data[:, 1, :]
    return grid, {**params, **other_params}, memory_storage, data, vs, ws

def make_fn_video(dir, params, vs, ws):
    os.makedirs(dir, exist_ok=True)
    save_params(params, f"{dir}/params.json")
    animate_video(f"{dir}/vs.mp4", vs)
    animate_video(f"{dir}/ws.mp4", ws)
