import numpy as np
from scipy.signal import find_peaks

from matplotlib import pyplot as plt

from tqdm import tqdm
import json

from WSN import *
import mutual_information as mtin


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
def get_fn_peaks_inside_period(self: WSN, r0=None, use_mst=False):
    if r0 is None:
        r0 = np.array([55, 55])
    results = self.transmit_continuous(signal_type="fn", start_frame=start_frame)
    assert np.all(np.array([result[0] for result in results]) == self.nodes)
    dt = mtin.default_fn_equ_params["dt"]
    T  = mtin.default_fn_equ_params["T"]
    # This is arbitrary, but I know this will work for this case
    max_tau = T // 5

    self.printv("Finding period")
    avg_period = 0
    for i, sigi in results:
        shifts = np.arange(-max_tau, max_tau, 1)
        mis = mi_shift(np.transpose([sigi, sigi]), shifts)

        mis = np.array(mis)
        max_peak = mis.max()
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
    self.printv("Period found to be", avg_period)

    all_peaks = []   # [(rn, rm, p, p0), ...]
    pair_to_ind = {} # {(n, m): i}

    # For now, tree must be a list so that the neural network
    # sees the peaks in the same order every time
    if use_mst:
        tree = list(self.find_MST())
    else:
        tree = [
            (None, 0, i)
            for i in range(1, len(self.nodes))
        ]
    for _, i, j in tree:
        ri, sigi = results[i]
        rj, sigj = results[j]
        max_tau = int((2 * avg_period) / dt)
        shifts = np.arange(-max_tau, max_tau, 1)
        mis = mtin.mi_shift(np.transpose([sigi, sigj]), shifts)

        mis = np.array(mis)
        max_peak = mis.max()
        inclusion_factor = 0.5
        peaks, _ = find_peaks(mis, height=max_peak * inclusion_factor)
        if self.verbose and np.random.uniform(0, 1) < 0.1:
            plt.plot(shifts, mis)
            plt.show(block=False)
        if len(peaks) == 0:
            plt.plot(shifts * dt, mis)
            plt.show(block=False)
            p = (dist(ri, r0) - dist(rj, r0)) / self.c
            raise ValueError(
                f"No peaks found between nodes {i} and {j} "
                f"at positions {ri} and {rj}.\n"
                f"The peak should be at time {p}."
            )
        peaks = shifts[peaks]
        # p0 = peak closest to 0
        p0 = peaks[np.argmin(np.abs(peaks))]

        # p = ideal peak
        p = (dist(ri, r0) - dist(rj, r0)) / self.c
        pair_to_ind[(i, j)] = len(all_peaks)
        all_peaks.append((ri, rj, p, p0))
    return all_peaks, pair_to_ind, avg_period

[setattr(WSN, attr, globals()[attr]) for attr in (
    "get_fn_peaks_inside_period",
)]

N = 12
c = 1.6424885622140555
wsn = WSN(100, N, D=142, std=0, c=c, verbose=False)
wsn.reset_anchors(range(N))

def get_input_and_output(wsn: WSN):
    all_peaks, pair_to_ind, period = wsn.get_fn_peaks_inside_period()
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
    for ri, rj, p, p0 in all_peaks:
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
input = np.array(input)
output = np.array(output)

np.save("nn_saves/input2000.npy", input)
np.save("nn_saves/output2000.npy", output)
