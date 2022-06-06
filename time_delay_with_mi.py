from matplotlib import pyplot as plt
# from scipy.integrate import solve_ivp
from m_solve_ivp import solve_ivp
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
        # print(partial_sol.known_vals)
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

        t_range = (0, 1000)
        size = t_range[1] - t_range[0]
        t_eval = np.arange(*t_range, dt)
        xs = np.empty(shape=(size, 2))
        std = 1
        xs[:tau] = np.random.normal(size=(tau, 2), scale=std)
        
        b, c = np.random.uniform(-1, 1, size=2)
        dprint(b, c)
        A = np.array(
            [[0, c],
             [b, 0]]
        )
        for i in range(tau, size):
            xs[i] = np.matmul(A, xs[i-tau]) + np.random.normal(size=2, scale=std)

        plt.figure()
        plt.plot(t_eval, xs)
        plt.show(block=False)

        xs, ys = xs.T
    else:
        print("Invalid correlation type:", correlation_type)
        quit()
    
    num_shifts = 40
    ts: np.ndarray = t_eval
    mis = []
    shifts = np.arange(-num_shifts // 2, num_shifts // 2, 1)

    if do_interactive_graph:
        plt.ion()
        fig1, ax1 = plt.subplots()

        xline, = ax1.plot(ts, xs, label="x(t)")
        yline, = ax1.plot(ts, ys, label="y(t-τ)")

        ax1.legend()

        fig2, ax2 = plt.subplots()
        # mis = np.zeros(shape=shifts.shape)
        miline, = ax2.plot(shifts * dt, np.concatenate((mis, np.zeros(len(shifts) - len(mis)))))
        ax2.set_ylim((-2, 6))

    for shift in shifts:
        if shift < 0:
            length = len(ts) + shift
            new_xs = xs[:length]
            new_ys = ys[-length:]
            if do_interactive_graph:
                xline.set_ydata(
                    np.concatenate((
                        new_xs, np.zeros(len(ts) - length)
                    ))
                )
                yline.set_ydata(
                    np.concatenate((
                        new_ys, np.zeros(len(ts) - length)
                    ))
                )
        else:
            length = len(ts) - shift
            new_xs = xs[-length:]
            new_ys = ys[:length]
            if do_interactive_graph:
                xline.set_ydata(
                    np.concatenate((
                        np.zeros(len(ts) - length), new_xs
                    ))
                )
                yline.set_ydata(
                    np.concatenate((
                        np.zeros(len(ts) - length), new_ys
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

    if not do_interactive_graph:
        plt.figure()
        plt.plot(shifts, mis)
        plt.show()
