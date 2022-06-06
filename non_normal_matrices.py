from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from PartialSolution import *

if __name__ == "__main__":

    partial_sol : PartialSolution

    def get_nonnormal_A(a):
        return np.array(
            [[-0.1, a],
             [0, -0.1]]
        )

    # No time delay:
    # F = lambda t, s: np.array([s[0] ** 3 - s[0], s[1] ** 3 - s[1]]) + np.matmul(A, s)

    # With time delay:
    def F(t, s):
        partial_sol.add_val(t, s)
        return np.array([s[0] ** 3 - s[0], s[1] ** 3 - s[1]]) + np.matmul(A, partial_sol(t-time_delay))
    def F_lin(t, s):
        partial_sol.add_val(t, s)
        return np.matmul(A, partial_sol(t-time_delay))
    
    s0 = np.array([-0.1, 0.1])

    # Case A
    # partial_sol = PartialSolution((0, s0))
    # range_a = (0, 10)
    # A = np.array(
    #     [[-0.1, 0.05],
    #      [0.05, -0.1]]
    # )
    # t_eval = np.arange(*range_a, 0.01)
    # sol_a = solve_ivp(F, range_a, s0, t_eval=t_eval)
    partial_sol = PartialSolution((0, s0))
    time_delay = 0
    range_a = (0, 10)
    A = np.array(
        [[-1, 1],
         [1, -1]]
    )
    A = np.array(
        [[-0.2, 1],
         [-1, 0]]
    )
    t_eval = np.arange(*range_a, 0.0001)
    sol_a = solve_ivp(F_lin, range_a, s0, t_eval=t_eval)


    # Case B
    s0 = np.array([0.5, 1])
    partial_sol = PartialSolution((0, s0))
    range_b = (0, 40)
    # A = get_nonnormal_A(a=4.5)
    # A = np.array(
    #     [[-1, 0],
    #      [0, -1]]
    # )
    A = np.array(
        [[0, 1],
         [-1, 0]]
    )
    A = get_nonnormal_A(-0.4)
    time_delay = 0
    t_eval = np.arange(*range_b, 0.0001)
    sol_b = solve_ivp(F_lin, range_b, s0, t_eval=t_eval)


    # Case C
    partial_sol = PartialSolution((0, s0))
    range_c = (0, 40)
    # A = get_nonnormal_A(a=10)
    # A = np.array(
    #     [[-1, -1],
    #      [1, -1]]
    # )
    # A = np.array(
    #     [[0, 1],
    #      [-1, 0]]
    # )
    time_delay = 1
    t_eval = np.arange(*range_c, 0.0001)
    sol_c = solve_ivp(F_lin, range_c, s0, t_eval=t_eval)

    def plot_boundaries(range):
        plt.plot(range, np.full_like(range, 1), color="gray")
        plt.plot(range, np.full_like(range, -1), color="gray")
        plt.ylim((-2, 2))


    plt.subplot(1, 3, 1)
    plt.plot(sol_a.t, sol_a.y[0])
    plt.plot(sol_a.t, sol_a.y[1])
    plt.xlabel("delay=0")

    plot_boundaries(range_a)

    plt.subplot(1, 3, 2)
    plt.plot(sol_b.t, sol_b.y[0])
    plt.plot(sol_b.t, sol_b.y[1])
    plt.xlabel("delay=0.2")

    plot_boundaries(range_b)

    plt.subplot(1, 3, 3)
    plt.plot(sol_c.t, sol_c.y[0])
    plt.plot(sol_c.t, sol_c.y[1])
    plt.xlabel("delay=0.5")

    plot_boundaries(range_c)


    verify_partial_sol = False
    plt.show(block=False)

    plt.figure()
    plt.plot(*sol_a.y)
    plt.plot(*sol_b.y)
    plt.plot(*sol_c.y)
    plt.show()

    if verify_partial_sol:
        plt.figure()
        sol = np.transpose([partial_sol(t) for t in sol_c.t])
        plt.plot(sol_c.t, sol[0])
        plt.plot(sol_c.t, sol[1])
        plot_boundaries(range_c)
        plt.show()
