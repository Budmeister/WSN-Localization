import numpy as np
from numba import jit
from scipy.signal import find_peaks

from tqdm import tqdm
import json

from WSN import WSN
from fn import FHN
import mutual_information as mtin
from nn_numba_funcs import *


dt = 0.01
T = 20000
mtin.default_fn_equ_params["dt"] = 0.01
mtin.default_fn_equ_params["T"] = 20_000
mtin.default_fn_equ_params['stim'] = [[[250, 350], [45, 65], [45, 65]]]
start_frame = int(40 / mtin.default_fn_equ_params["dt"])
mtin.default_fn_equ = FHN(**mtin.default_fn_equ_params)

if mtin.default_fn_sol is None:
    mtin.default_fn_sol = np.load("default_fn_sol_dt_0.01.npy")

# mtin.solve_default_fn_equ()

# np.save("default_fn_sol_dt_0.01.npy", mtin.default_fn_sol)


# Generate signals and find all peaks
# @jit(nopython=True)
def get_fn_peaks_inside_period(nodes, c, r0=None):
    if r0 is None:
        r0 = np.array([55, 55])
    results = transmit_continuous(nodes, start_frame=start_frame)
    # assert np.all(np.array([result[0] for result in results]) == nodes)
    # This is arbitrary, but I know this will work for this case
    max_tau = T // 5

    # self.printv("Finding period")
    avg_period = 0
    for i, sigi in results:
        shifts = np.arange(-max_tau, max_tau, 1)
        sigs = np.empty((len(sigi), 2))
        sigs[:, 0] = sigi
        sigs[:, 1] = sigi
        mis = mi_shift(sigs, shifts)

        max_peak = mis.max()
        peaks: np.ndarray
        peaks, _ = find_peaks(mis)
        peaks = peaks[np.argsort(mis[peaks])]
        peaks = peaks[-5:]
        peaks.sort()
        period1 = peaks[1] - peaks[0]
        period2 = peaks[2] - peaks[1]
        period = (period1 + period2) / 2
        period *= dt
        avg_period += period
    avg_period /= len(results)
    # self.printv("Period found to be", avg_period)

    all_peaks = np.empty((len(nodes - 1), 2))   # [(rn, rm, p, p0), ...]
    b = 0

    # For now, tree must be a list so that the neural network
    # sees the peaks in the same order every time
    tree = np.array([
        (0, i)
        for i in range(1, len(nodes))
    ])
    for i, j in tree:
        ri, sigi = results[i]
        rj, sigj = results[j]
        max_tau = int((2 * avg_period) / dt)
        shifts = np.arange(-max_tau, max_tau, 1)
        sigs = np.empty((len(sigi), 2))
        sigs[:, 0] = sigi
        sigs[:, 1] = sigj
        mis = mi_shift(sigs, shifts)

        max_peak = mis.max()
        inclusion_factor = 0.5
        peaks, _ = find_peaks(mis, height=max_peak * inclusion_factor)
        if len(peaks) == 0:
            p = (dist(ri, r0) - dist(rj, r0)) / c
            print(
                f"No peaks found between nodes {i} and {j} "
                f"at positions {ri} and {rj}.\n"
                f"The peak should be at time {p}."
            )
        peaks = shifts[peaks]
        # p0 = peak closest to 0
        p0 = peaks[np.argmin(np.abs(peaks))]

        # p = ideal peak
        p = (dist(ri, r0) - dist(rj, r0)) / c
        # all_peaks.append((p, p0))
        all_peaks[b] = (p, p0)
        b += 1

    return all_peaks, avg_period

# [setattr(WSN, attr, globals()[attr]) for attr in (
#     "get_fn_peaks_inside_period",
# )]

N = 12
c = 1.6424885622140555
wsn = WSN(100, N, D=142, std=0, c=c, verbose=False)
wsn.reset_anchors(range(N))

def get_input_and_output(wsn: WSN):
    all_peaks, period = get_fn_peaks_inside_period(wsn.nodes, wsn.c)
    # Input format:
    # [
    #   wsn.nodes.flatten() -- len = 2N,
    #   peaks -- len = N-1,
    #   period -- len = 1
    # ]

    # Output format:
    # peaks -- len = N-1
    input = []
    output = []
    for p, p0 in all_peaks:
        input.append(p0 / period)
        output.append(p / period)
    input.append(period)
    input = np.concatenate((wsn.nodes.flatten() / 100, input))
    output = np.array(output)

    return input, output

data_size = 2000
# input_shape = (data_size, 3 * N)
# output_shape = (data_size, N - 1)
input = [0] * data_size
output = [0] * data_size
for j in tqdm(range(data_size)):
    error = True
    while error:
        error = False
        try:
            wsn.reset_nodes()
            i, o = get_input_and_output(wsn)
            input[j] = i
            output[j] = o
        except ValueError as e:
            print(f"ValueError: {e}")
            error = True
    if j % 100 == 0 and j != 0:
        np.save(f"nn_saves/orig/input{j}.npy", np.array(input))
        np.save(f"nn_saves/orig/input{j}.npy", np.array(input))
        np.save(f"nn_saves/copy/output{j}.npy", np.array(output))
        np.save(f"nn_saves/copy/output{j}.npy", np.array(output))
input = np.array(input)
output = np.array(output)

np.save("nn_saves/orig/input2000.npy", input)
np.save("nn_saves/orig/output2000.npy", output)
np.save("nn_saves/copy/input2000.npy", input)
np.save("nn_saves/copy/output2000.npy", output)
