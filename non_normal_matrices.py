from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    class PartialSolution:
        def __init__(self, initial_value=None):
            self.known_vals = []
            self.i = 0
            if initial_value is not None:
                self.add_val(*initial_value)
        
        def add_val(self, t, s):
            for i in range(len(self.known_vals)):
                if self.known_vals[i][0] > t:
                    self.known_vals.insert(i, (t, s))
                    break
                elif self.known_vals[i][0] == t:
                    self.known_vals[i] = (t, s)
                    break
            else:
                self.known_vals.append((t, s))
        
        def __call__(self, t):
            if len(self.known_vals) == 1:
                return self.known_vals[0][1]
            if self.i >= len(self.known_vals) - 1:
                self.i = len(self.known_vals) - 2
            if self.i < 0:
                self.i = 0
            while self.known_vals[self.i+1][0] <= t:
                self.i += 1
                if self.i == len(self.known_vals) - 1:
                    return self.known_vals[-1][1]
            while self.known_vals[self.i][0] > t:
                self.i -= 1
                if self.i == -1:
                    return self.known_vals[0][1]
            diff_s = self.known_vals[self.i+1][1] - self.known_vals[self.i][1]
            diff_t = self.known_vals[self.i+1][0] - self.known_vals[self.i][0]
            t -= self.known_vals[self.i][0]
            return t / diff_t * diff_s + self.known_vals[self.i][1]

    partial_sol : PartialSolution

    def get_nonnormal_A(a):
        return np.array(
            [[-0.1, a],
             [0, -0.1]]
        )
    time_delay = 0.001

    # No time delay:
    # F = lambda t, s: np.array([s[0] ** 3 - s[0], s[1] ** 3 - s[1]]) + np.matmul(A, s)

    # With time delay:
    def F(t, s):
        partial_sol.add_val(t, s)
        return np.array([s[0] ** 3 - s[0], s[1] ** 3 - s[1]]) + np.matmul(A, partial_sol(t-time_delay))
    
    s0 = np.array([-0.5, 0.5])

    # Case A
    partial_sol = PartialSolution((0, s0))
    A = np.array(
        [[-0.1, 0.05],
         [0.05, -0.1]]
    )
    t_eval = np.arange(0, 10, 0.01)
    sol_a = solve_ivp(F, (0, 10), s0, t_eval=t_eval)


    # Case B
    partial_sol = PartialSolution((0, s0))
    A = get_nonnormal_A(a=4.5)
    sol_b = solve_ivp(F, (0, 10), s0, t_eval=t_eval)


    # Case C
    partial_sol = PartialSolution((0, s0))
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


    verify_partial_sol = False
    plt.show(block=not verify_partial_sol)

    if verify_partial_sol:
        plt.figure()
        sol = np.transpose([partial_sol(t) for t in sol_c.t])
        plt.plot(sol_c.t, sol[0])
        plt.plot(sol_c.t, sol[1])
        plot_boundaries((0, 0.7))
        plt.show()
