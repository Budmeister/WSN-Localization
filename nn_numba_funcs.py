from numba import jit
import numpy as np

import mutual_information as mtin


@jit(nopython=True)
def dist2(r1: np.ndarray, r2: np.ndarray):
    diff = r1 - r2
    return np.dot(diff, diff)

@jit(nopython=True)
def dist(r1: np.ndarray, r2: np.ndarray):
    return np.sqrt(dist2(r1, r2))

@jit(nopython=True)
def get_fn_data(nodes, start_frame=400):
    data = mtin.default_fn_sol
    
    # signals = np.array([data[start_frame:, :, int(node[0]), int(node[1])] for node in nodes])
    # # signals.shape = (t, vw)
    # return signals
    signals = np.empty((len(nodes), len(data) - start_frame, 2))
    for i, node in enumerate(nodes):
        signals[i] = data[start_frame:, :, int(node[0]), int(node[1])]
    return signals

@jit(nopython=True)
def transmit_continuous(nodes, start_frame=4000):
    xs = get_fn_data(nodes, start_frame=start_frame)[:, :, 0]
    results = list(zip(nodes, xs))
    return results

@jit(nopython=True)
def mi_shift(xs, shifts):
    xs, ys = xs.T
    mis = np.empty(len(shifts))

    for ind, shift in enumerate(shifts):
        if shift < 0:
            length = len(xs) + shift
            new_xs = xs[:length]
            new_ys = ys[-length:]
        else:
            length = len(xs) - shift
            new_xs = xs[-length:]
            new_ys = ys[:length]
        
        # new_xy = np.array([new_xs, new_ys])
        assert len(new_xs) == len(new_ys)
        new_xy = np.empty((2, len(new_xs)))
        new_xy[0] = new_xs
        new_xy[1] = new_ys
        cor = np.corrcoef(new_xy)
        mi = -0.5 * np.log(np.linalg.det(cor))
        mis[ind] = mi
    return mis