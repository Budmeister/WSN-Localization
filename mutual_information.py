from matplotlib import pyplot as plt
# from scipy.integrate import solve_ivp
from m_solve_ivp import solve_ivp
from scipy.signal import find_peaks
import numpy as np
from PartialSolution import *

if "DEBUG" not in globals():
    DEBUG = True

if DEBUG:
    def dprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    def dprint(*args, **kwargs):
        pass

def solve_ode_with_time_delay(A, B, time_delay, s0, t_range, dt):
    def F(t, s, partial_sol=None):
        # if t in t_eval:
        # partial_sol.add_val(t, s)
        # p = partial_sol(t-time_delay)
        dprint()
        if partial_sol is None or \
                partial_sol[0] is None or \
                partial_sol[1] is None or \
                len(partial_sol[0]) == 0 or \
                len(partial_sol[1]) == 0:
            dprint("partial_sol is None; returning initial condition")
            p = s0
        else:
            ts, ys = partial_sol
            ts = np.hstack(ts)
            ys = np.hstack(ys)
            ys = np.transpose(ys)
            dprint(ts, ys)
            if len(ts) != len(ys):
                dprint(f"Unequal lengths: len(t)={len(ts)}, len(y)={len(ys)}")
            known_vals = [(ts[i], ys[i]) for i in range(len(ys))] + [(t, s)]
            if "outer_sol" in globals():
                outer_sol.known_vals = known_vals
            p, _ = x_polate(known_vals, t - time_delay, extrapolate_right=True)
        retval = np.matmul(A, s) + np.matmul(B, p)
        dprint("p =", p)
        dprint("s =", s)
        dprint("retval =", retval)
        return retval
    
    t_eval = np.arange(*t_range, dt)
    sol = solve_ivp(F, t_range, s0, t_eval=t_eval, receive_partial_sol=True, args=())
    return sol.y

def mi_shift(xs, shifts, do_interactive_graph=False):
    xs, ys = xs.T
    mis = []
    if do_interactive_graph:
        from matplotlib import pyplot as plt
        plt.ion()
        fig1, ax1 = plt.subplots()

        xline, = ax1.plot(xs, label="x(t)")
        yline, = ax1.plot(ys, label="y(t-τ)")

        ax1.legend()

        fig2, ax2 = plt.subplots()
        miline, = ax2.plot(shifts * dt, np.concatenate((mis, np.zeros(len(shifts) - len(mis)))))
        ax2.set_ylim((-2, 6))

    for shift in shifts:
        if shift < 0:
            length = len(xs) + shift
            new_xs = xs[:length]
            new_ys = ys[-length:]
            if do_interactive_graph:
                xline.set_ydata(
                    np.concatenate((
                        new_xs, np.zeros(len(xs) - length)
                    ))
                )
                yline.set_ydata(
                    np.concatenate((
                        new_ys, np.zeros(len(xs) - length)
                    ))
                )
        else:
            length = len(xs) - shift
            new_xs = xs[-length:]
            new_ys = ys[:length]
            if do_interactive_graph:
                xline.set_ydata(
                    np.concatenate((
                        np.zeros(len(xs) - length), new_xs
                    ))
                )
                yline.set_ydata(
                    np.concatenate((
                        np.zeros(len(xs) - length), new_ys
                    ))
                )
            
        
        cor = np.corrcoef(np.array([new_xs, new_ys]))
        mi = -0.5 * np.log(np.linalg.det(cor))
        mis.append(mi)
        
        if do_interactive_graph:
            fig1.canvas.draw()
            fig1.canvas.flush_events()

            miline.set_ydata(np.concatenate((mis, np.zeros(len(shifts) - len(mis)))))
            fig2.canvas.draw()
            fig2.canvas.flush_events()
    return mis

def get_time_delay(xs, shifts, do_interactive_graph=False):
    mis = mi_shift(xs, shifts, do_interactive_graph=do_interactive_graph)
    mean = np.mean(mis)
    peaks, _ = find_peaks(mis, height=mean * 30)
    peaks = shifts[peaks]
    if len(peaks) != 1:
        from matplotlib import pyplot as plt
        print(f"Unable to determine peaks. Found peaks: {peaks}")
        plt.plot(shifts, mis)
        plt.show()
        raise ValueError(f"Unable to determine peaks. Found peaks: {peaks}")
    return peaks[0]


def get_ar_data(samples, b=None, c=None, tau=5, std=1):
    if b is None:
        b = np.random.uniform(-1, 1)
    if c is None:
        c = np.random.uniform(-1, 1)
    dprint(b, c)
    A = np.array(
        [[0, c],
         [b, 0]]
    )

    xs = np.empty(shape=(samples, 2))
    xs[:tau] = np.random.normal(size=(tau, 2), scale=std)
    
    for i in range(tau, samples):
        xs[i] = np.matmul(A, xs[i-tau]) + np.random.normal(size=2, scale=std)
    return xs

def get_multi_ar_data(samples, A, tau=5, std=1):
    if A is None:
        A = np.random.uniform(-1, 1, size=(5, 5))
        np.fill_diagonal(A, 0)
    dprint(A)

    xs = np.empty(shape=(samples, len(A)))
    xs[:tau] = np.random.normal(size=(tau, len(A)), scale=std)
    
    for i in range(tau, samples):
        xs[i] = np.matmul(A, xs[i-tau]) + np.random.normal(size=len(A), scale=std)
    return xs

def get_one_way_ar_data(samples, coeffs=None, taus=5, std=1):
    if coeffs is None:
        coeffs = 5
    if hasattr(coeffs, "__len__"):
        size = len(coeffs) + 1
    elif hasattr(taus, "__len__"):
        size = len(taus) + 1
    elif isinstance(coeffs, int):
        size = coeffs + 1
        coeffs = np.random.uniform(-1, 1, size=coeffs)
    else:
        raise ValueError("Either coeffs or taus should have a size or coeffs should be an int")
    temp = np.empty(size)
    temp[1:] = coeffs
    temp[0] = 0
    coeffs = temp
    temp = np.empty_like(coeffs, dtype=np.int32)
    temp[1:] = taus
    temp[0] = 0
    taus = temp
    
    dprint(coeffs)

    xs = np.empty(shape=(samples, size))
    xs[:np.max(taus), 1:] = np.random.normal(size=(np.max(taus), size - 1), scale=std)
    xs[:, 0] = np.random.normal(size=samples, scale=std)
    
    for i in range(np.min(taus), samples):
        # xs[i] = xs[[i - taus], range(len(coeffs))] + np.random.normal(size=len(coeffs), scale=std)
        for j in range(size):
            if i - taus[j] >= 0:
                xs[i, j] = coeffs[j] * xs[i-taus[j], 0] + np.random.normal(scale=std)
    return xs


if __name__ == "__main__":
    # "de" or "ar"
    correlation_type = "ar"
    do_interactive_graph = False

    if correlation_type == "de":
        A = np.array(
            [[-0.1, 0],
            [0, 0]]
        )
        B = np.array(
            [[0, 0],
            [-0.4, -0.1]]
        )
        time_delay = 5
        s0 = np.array([1, 0.5])
        outer_sol = PartialSolution((0, s0))

        t_range = (0, 40)
        dt = 0.1
        t_eval = np.arange(*t_range, dt)

        xs, ys = solve_ode_with_time_delay(A, B, time_delay, s0, t_range, dt)
        print([(a, b, len(a), len(b)) for (a, b) in ((np.array([kv[0] for kv in outer_sol.known_vals]), t_eval),)])
        plt.figure()
        plt.plot(t_eval, [outer_sol(t) for t in t_eval], label="partial_sol")
        plt.plot(t_eval, np.transpose((xs, ys)), label="real")
        plt.legend()
        plt.show(block=False)
        std = 0.1
        xs = xs + np.random.normal(size=xs.shape, scale=std)
        ys = ys + np.random.normal(size=xs.shape, scale=std)
    elif correlation_type == "ar":
        tau = 5
        dt = 1
        std = 1

        t_range = (0, 5000)
        size = t_range[1] - t_range[0]
        t_eval = np.arange(*t_range, dt)
        # xs = get_ar_data(b=1, samples=size, tau=tau, std=std)
        xs = get_one_way_ar_data(samples=size, coeffs=2, taus=(5, 1), std=std)
        xs = xs[:, 1:]

        # plt.figure()
        # plt.plot(t_eval, xs)
        # plt.show(block=False)

        # xs, ys = xs.T
    else:
        print("Invalid correlation type:", correlation_type)
        quit()
    
    num_shifts = 40
    ts: np.ndarray = t_eval
    mis = []
    shifts = np.arange(-num_shifts // 2, num_shifts // 2, 1)
    mis = mi_shift(xs, shifts, do_interactive_graph=do_interactive_graph)
    mean = np.mean(mis)
    peaks, _ = find_peaks(mis, height=mean * 6)
    print(shifts[peaks])

    if not do_interactive_graph:
        plt.figure()
        plt.plot(shifts, mis)
        plt.show()
