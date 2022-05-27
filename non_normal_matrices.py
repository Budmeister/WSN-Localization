from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    def get_nonnormal_A(a):
        return np.array(
            [[-0.1, a],
             [0, -0.1]]
        )
    F = lambda t, s: np.array([s[0] ** 3 - s[0], s[1] ** 3 - s[1]]) + np.matmul(A, s)
    s0 = [-0.5, 0.5]

    A = np.array(
        [[-0.1, 0.05],
         [0.05, -0.1]]
    )
    t_eval = np.arange(0, 10, 0.01)
    sol_a = solve_ivp(F, (0, 10), s0, t_eval=t_eval)


    A = get_nonnormal_A(a=4.5)
    sol_b = solve_ivp(F, (0, 10), s0, t_eval=t_eval)


    A = get_nonnormal_A(a=10)
    t_eval = np.arange(0, 0.7, 0.01)
    sol_c = solve_ivp(F, (0, 0.7), s0, t_eval=t_eval)

    def plot_boundaries(range):
        plt.plot(range, np.full_like(range, 1), color="gray")
        plt.plot(range, np.full_like(range, -1), color="gray")
        plt.ylim((-2, 2))


    plt.subplot(1, 3, 1)
    plt.plot(sol_a.t, sol_a.y[0])
    plt.plot(sol_a.t, sol_a.y[1])

    plot_boundaries((0, 10))

    plt.subplot(1, 3, 2)
    plt.plot(sol_b.t, sol_b.y[0])
    plt.plot(sol_b.t, sol_b.y[1])

    plot_boundaries((0, 10))

    plt.subplot(1, 3, 3)
    plt.plot(sol_c.t, sol_c.y[0])
    plt.plot(sol_c.t, sol_c.y[1])

    plot_boundaries((0, 0.7))


    plt.show()
