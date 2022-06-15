from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from scipy.integrate import solve_ivp

from pde import *


def get_fn_ode_data(a, b, R, I, tau, t_span, *args, **kwargs):
    vw0 = np.array([0, 0])
    def F(t, vw):
        v, w = vw
        return np.array([
            v - v**3 / 3 - w + R * I,
            (v + a - b * w) / tau
        ])
    sol = solve_ivp(F, t_span, vw0, *args, **kwargs)
    t = sol.t
    vw = sol.y
    return t, vw.T

# https://py-pde.readthedocs.io/en/latest/examples_gallery/pde_coupled.html#sphx-glr-examples-gallery-pde-coupled-py
class FitzhughNagumoPDE(PDEBase):
    """FitzHugh–Nagumo model with diffusive coupling"""

    def __init__(self, stimulus=0.5, τ=10, a=0, b=0, bc="auto_periodic_neumann"):
        self.bc = bc
        self.stimulus = stimulus
        self.τ = τ
        self.a = a
        self.b = b

    def evolution_rate(self, state, t=0):
        v, w = state  # membrane potential and recovery variable

        v_t = v.laplace(bc=self.bc) + v - v**3 / 3 - w + self.stimulus
        w_t = (v + self.a - self.b * w) / self.τ

        return FieldCollection([v_t, w_t])

def main():
    # t, vw = get_fn_ode_data(
    #     a=0.7,
    #     b=0.8,
    #     R=1,
    #     I=0.5,
    #     tau=12.5,
    #     t_span=(0, 200),
    #     t_eval=np.arange(0, 200, 0.1)
    # )
    # plt.plot(t, vw)
    # plt.show()
    # plt.plot(*vw.T)
    # plt.show()
    grid = UnitGrid([32, 32])
    state = FieldCollection.scalar_random_uniform(2, grid)

    memory_storage = MemoryStorage()
    eq = FitzhughNagumoPDE(a=0.7, b=0.8)
    result = eq.solve(state, t_range=100, dt=0.01, tracker=[memory_storage.tracker(0.1)])
    result.plot()
    # plot_kymographs(memory_storage)
    data = memory_storage.data
    data = np.array(data)
    vs = data[:, 0, :]
    ws = data[:, 1, :]

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(vs[0])
    plt.show(block=False)
    from time import sleep
    for a in range(0, 1000, 20):
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # xs = range(data.shape[2])
        # ts = range(data.shape[0])
        # xs, ts = np.meshgrid(xs, ts)
        # xs, ys = np.meshgrid(xs, xs)
        # ax.plot_surface(xs, ys, vs[a], cmap=cm.coolwarm,
        #                 linewidth=0, antialiased=False)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # # A StrMethodFormatter is used automatically
        # ax.zaxis.set_major_formatter('{x:.02f}')
        im.set_data(vs[a])
        sleep(0.1)



if __name__ == "__main__":
    main()
