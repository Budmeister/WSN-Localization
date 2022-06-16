import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    num_points = 10000
    fig = plt.figure()
    # "gaussian" or "dice"
    mi_type = "dice"

    if mi_type == "gaussian":
        s = 0.5
        r = 0.9
        mean = np.array([0.5, 0.5])
        K = np.array(
            [[s ** 2, r * s ** 2],
            [r * s ** 2, s ** 2]]
        )

        data = np.random.multivariate_normal(mean, K, size=num_points)
        plt.scatter(data[:, 0], data[:, 1])

        cor = np.corrcoef(data, rowvar=False)
        print(cor)
        mi = -0.5 * np.log2(np.linalg.det(cor))
        print(f"Mutual information: {mi}")
        plt.xlabel(f"MI: {mi}")
        plt.xlim(np.array([-8 / 3, 8 / 3]) + mean[0])
        plt.ylim(np.array([-2, 2]) + mean[1])
        plt.show()
    elif mi_type == "dice":
        N = 1000
        M = 100
        to_plot_size = int(N * 1.1)
        to_plot = np.empty(shape=(to_plot_size, 3))
        for c_offset in tqdm(range(to_plot_size)):
            data = np.array([
                (x, x % M, x % (M+c_offset)) for x in (np.random.randint(0, N) for _ in range(num_points))
            ])
            # plt.scatter(data[:, 0], data[:, 1])

            cor = np.corrcoef(data, rowvar=False)
            to_plot[c_offset] = (cor[0, 1], cor[0, 2], cor[1, 2])
        plt.plot(range(to_plot_size), to_plot[:, 0], label="Cor(a;b)")
        plt.plot(range(to_plot_size), to_plot[:, 1], label="Cor(a;c)")
        plt.plot(range(to_plot_size), to_plot[:, 2], label="Cor(b;c)")
        plt.xlabel("c_offset")
        plt.legend()
        plt.show()
        # print(cor)

