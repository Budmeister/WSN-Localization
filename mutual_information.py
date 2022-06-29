from matplotlib import pyplot as plt
# from scipy.integrate import solve_ivp
from m_solve_ivp import solve_ivp
from scipy.signal import find_peaks
import numpy as np
from PartialSolution import *
from tqdm import tqdm
from fn import *

if "DEBUG" not in globals():
    DEBUG = False

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
            p, _ = x_polate(known_vals, t - time_delay, extrapolate_right=True)
        retval = np.matmul(A, s) + np.matmul(B, p)
        dprint("p =", p)
        dprint("s =", s)
        dprint("retval =", retval)
        return retval
    
    t_eval = np.arange(*t_range, dt)
    sol = solve_ivp(F, t_range, s0, t_eval=t_eval, receive_partial_sol=True, args=())
    return sol.y

def mi_shift(xs, shifts, do_interactive_graph=False, dt=1):
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
        ax2.set_ylim((-2, 2))

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

def get_time_delay(xs, shifts, ensure_one_peak=True, closest_to_zero=False, do_interactive_graph=False, show_error=None, **kwargs):
    mis = mi_shift(xs, shifts, do_interactive_graph=do_interactive_graph)
    mean = np.mean(mis)
    peaks, properties = find_peaks(mis, height=(mean * 30 if ensure_one_peak else 0), **kwargs)
    peaks = shifts[peaks]
    if closest_to_zero:
        if len(peaks) == 0:
            if show_error:
                print(f"Unable to determine peaks. Found peaks: {peaks}")
                plt.plot(shifts, mis)
                plt.title(show_error)
                plt.show()
            raise ValueError(f"Unable to determine peaks. Found peaks: {peaks}")
        return peaks[np.argmin(abs(peaks))]
    elif ensure_one_peak and len(peaks) != 1:
        if show_error:
            print(f"Unable to determine peaks. Found peaks: {peaks}")
            plt.plot(shifts, mis)
            plt.title(show_error)
            plt.show()
        raise ValueError(f"Unable to determine peaks. Found peaks: {peaks}")
    return peaks[0] if ensure_one_peak else peaks[np.argmax(properties["peak_heights"])]


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

def get_one_way_osc_data(samples, taus, epsilon=0.25, f=None, a=4):
    if f is None:
        f = lambda x: a * x * (1 - x)
    taus = [0] + [tau for tau in taus]
    sigs = len(taus)
    x0 = np.random.uniform(size=sigs)
    xs = [x0]
    def g(x, y):
        return f(y) - f(x)
    def next(x, t, epsilon):
        y = np.array([
            xs[t-tau][i] if t-tau >= 0 else np.random.uniform()
            for i, tau in enumerate(taus)
        ])
        x0_t_tau = np.array([
            xs[t-tau][0] if t-tau >= 0 else np.random.uniform()
            for tau in taus
        ])
        retval = np.array([
            f(x[i]) + epsilon * g(x[i], x0_t_tau[i])
            for i in range(sigs)
        ])
        return retval
    
    x = x0
    for t in range(1, samples):
        x = next(x, t-1, epsilon)
        xs.append(x)
        
    xs = np.array(xs)
    return xs
    
default_fn_equ_params = {
    'N': 100,
    'n': 100,
    'T': 1000,
    'dt': 0.1,
    'D': 1,
    'a': 0.5,
    'b': 0.7,
    'c': 0.3,
    'I0': 1.0,
    # 'stim': [[[25, 40], [45, 65], [45, 65]]],
    'circular_stim': True,
    'stim': [[[25, 40], [55, 55], 10]],
    'w_h': 1,
    'w_l': 1,
    # 'noise': 0.02
}
def get_default_fn_initial_state():
    # grid = UnitGrid((N, N))
    N, n = [default_fn_equ_params.get(key) for key in ("N", "n")]
    shape = (n, n)
    grid = CartesianGrid([(0, N), (0, N)], shape)
    data_v = np.zeros(shape)
    data_w = np.zeros(shape)
    # data_v[N // 2, N // 2] = I
    v0 = ScalarField(grid, data_v)
    w0 = ScalarField(grid, data_w)
    return grid, FieldCollection((v0, w0))
default_fn_equ = FHN(**default_fn_equ_params)
default_fn_sol = None
def solve_default_fn_equ(resolve=False):
    global default_fn_sol
    if default_fn_sol is None or resolve:
        T, dt = [default_fn_equ_params.get(key) for key in ("T", "dt")]
        grid, vw0 = get_default_fn_initial_state()
        memory_storage = MemoryStorage()
        default_fn_equ.solve(vw0, t_range=T * dt, dt=dt, tracker=[memory_storage.tracker(dt)])
        default_fn_sol = np.array(memory_storage.data)
    # save_params(default_fn_equ_params, "./default_fn_equ_params.json")

def get_fn_data(samples, nodes, use_default_equ=True, initial_state=None, *args, **kwargs):
    if use_default_equ:
        solve_default_fn_equ()
        data = default_fn_sol
    else:
        eq = FHN(*args, **kwargs)
        if initial_state is None:
            grid, initial_state = get_default_fn_initial_state()
        dt, = [default_fn_equ_params.get(key) for key in ("dt")]
        memory_storage = MemoryStorage()
        eq.solve(initial_state, t_range=samples * dt, tracker=[memory_storage.tracker(dt)])
        data = np.array(memory_storage.data)
    
    signals = np.array([data[:, :, int(node[0]), int(node[1])] for node in nodes], copy=False)
    # signals.shape = (t, vw)
    return signals
        

def main():
    # "de" or "ar" or "osc"
    correlation_type = "osc"
    do_interactive_graph = False
    show_xs = True
    ylim = None

    if correlation_type == "de":
        A = np.array(
            [[-1, 0.5, 0],
             [-0.5, -1, 0],
             [0, 0, 0]]
        )
        B = np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0.1, 0, -0.1]]
        )
        time_delay = np.pi / 2
        s0 = np.array([1, 0, 0.5])
        outer_sol = PartialSolution((0, s0))

        t_range = (0, 40)
        dt = 0.1
        t_eval = np.arange(*t_range, dt)

        xs = solve_ode_with_time_delay(A, B, time_delay, s0, t_range, dt)
        # print([(a, b, len(a), len(b)) for (a, b) in ((np.array([kv[0] for kv in outer_sol.known_vals]), t_eval),)])
        std = 0.05
        xs = xs + np.random.normal(size=xs.shape, scale=std)
        xs = xs[[0, 2]].T
    elif correlation_type == "ar":
        tau = 5
        dt = 1
        std = 1

        t_range = (0, 1000)
        size = t_range[1] - t_range[0]
        t_eval = np.arange(*t_range, dt)
        # xs = get_ar_data(b=1, samples=size, tau=tau, std=std)
        xs = get_one_way_ar_data(samples=size, coeffs=2, taus=(10, 5), std=std)
        xs = xs[:, 1:]

        # plt.figure()
        # plt.plot(t_eval, xs)
        # plt.show(block=False)

        # xs, ys = xs.T
    elif correlation_type == "osc":
        sigs = 3
        # tau = 5
        taus = 0, 5, 10
        samples = 1000
        epsilon = 0.2
        # show_xs = False

        a = 4
        xs = get_one_way_osc_data(samples, taus, epsilon, a=a)
        xs = xs[:, -2:]

        dt = 1
        t_eval = np.arange(0, samples, dt)
        ylim = (0, 1)
        
    else:
        print("Invalid correlation type:", correlation_type)
        quit()
    
    if show_xs:
        plt.figure()
        plt.plot(t_eval, xs)
        plt.ylim(ylim)
        plt.show(block=False)
    
    num_shifts = 80
    ts: np.ndarray = t_eval
    mis = []
    shifts = np.arange(-num_shifts // 2, num_shifts // 2, 1)
    mis = mi_shift(xs, shifts, do_interactive_graph=do_interactive_graph, dt=dt)
    mean = np.mean(mis)
    peaks, _ = find_peaks(mis, height=mean * 6)
    print(shifts[peaks])

    if not do_interactive_graph:
        plt.figure()
        try:
            plt.plot([-time_delay, -time_delay], [np.max(mis), np.min(mis)], color="gray")
        except:
            dprint("time_delay not defined")
        plt.plot(shifts * dt, mis)
        plt.show()

if __name__ == "__main__":
    main()