from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from PartialSolution import *


if __name__ == "__main__":
    partial_sol : PartialSolution

    A = np.array(
        [[-0.1, 0],
         [0, 0]]
    )
    B = np.array(
        [[0, 0],
         [4, -0.1]]
    )
    time_delay = 4
    s0 = np.array([3, 0.5])
    partial_sol = PartialSolution((0, s0))

    def event(t, s):
        partial_sol.add_val(t, s)
        return 1
    
    def F(t, s):
        # partial_sol.add_val(t, s)
        p = partial_sol(t-time_delay)
        return np.matmul(A, s) + np.matmul(B, p)

    t_range = (0, 10)
    dt = 0.1
    t_eval = np.arange(*t_range, dt)

    sol = solve_ivp(F, t_range, s0, t_eval=t_eval, events=event)
    plt.figure()
    plt.plot(t_eval, [partial_sol(t) for t in t_eval])
    plt.plot(t_eval, np.transpose(sol.y))
    plt.show(block=False)

    plt.ion()
    fig1, ax1 = plt.subplots()

    ts: np.ndarray = t_eval
    xs: np.ndarray = sol.y[0]
    ys: np.ndarray = sol.y[1]

    xline, = ax1.plot(ts, xs)
    yline, = ax1.plot(ts, ys)

    fig2, ax2 = plt.subplots()
    shifts = np.arange(-len(ts) + 1, len(ts) - 1, 1)
    # mis = np.zeros(shape=shifts.shape)
    mis = []
    miline, = ax2.plot(shifts * dt, np.concatenate((mis, np.zeros(len(shifts) - len(mis)))))
    ax2.set_ylim((-2, 6))

    for shift in shifts:
        if shift < 0:
            length = len(ts) + shift
            new_xs = xs[:length]
            new_ys = ys[-length:]
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
        
        fig1.canvas.draw()
        fig1.canvas.flush_events()

        miline.set_ydata(np.concatenate((mis, np.zeros(len(shifts) - len(mis)))))
        fig2.canvas.draw()
        fig2.canvas.flush_events()
    
    fig2.show(True)


