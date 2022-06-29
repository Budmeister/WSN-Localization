from WSN import *

def err(est, real, anchors):
    if len(est) != len(real):
        raise ValueError()
    num_nodes = len(est)
    num_determined = 0
    euclid_dist = 0
    for n in range(num_nodes):
        est_node = est[n]
        real_node = real[n]
        if n in anchors or np.all(est_node == np.zeros(2)):
            continue
        num_determined += 1
        diff = est_node - real_node
        euclid_dist += np.dot(diff, diff)
    euclid_dist = np.sqrt(euclid_dist)
    return num_determined, euclid_dist

def test_fn_clusters(
    num_clusters,
    cluster_size_range,
    cluster_spacing_range,
    iterations,
    true_center,
    true_speed
):
    import multiprocessing as mp
    from time import time

    name = mp.current_process().name
    print(f"Process {name} running with num_clusters={num_clusters}.")

    wsn = WSN(100, N=0, D=142, std=0, verbose=False)
    wsn.num_clusters = num_clusters

    all_errs = np.zeros((len(cluster_size_range), len(cluster_spacing_range), 2))

    for i, cluster_size in enumerate(cluster_size_range):
        starttime = time()
        print(f"Process {name} running with cluster_size={cluster_size}.")
        wsn.cluster_size = cluster_size
        for j, cluster_spacing in enumerate(cluster_spacing_range):
            wsn.cluster_spacing = cluster_spacing
            avg_errs = np.zeros(2)
            for it in range(iterations):
                wsn.reset_clusters()
                wsn.reset_anchors()
                est_pos, = wsn.localize_continuous(method="cTDOA", signal_type="fn", return_full_array=True)
                distance_err = dist(est_pos[:2], true_center)
                speed = np.sqrt(est_pos[2])
                speed_err = np.abs(speed - true_speed)

                avg_errs += np.array([distance_err, speed_err])
            avg_errs /= iterations
            all_errs[i, j] = avg_errs
        print(f"Process {name} finished running with cluster_size={cluster_size} after {round(time() - starttime, 1)}s.")
    return all_errs

def main():
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import multiprocessing as mp

    iterations = 500
    num_clusters_range = range(3, 11)
    cluster_size_range = range(3, 5)
    cluster_spacing_range = range(3, 11)

    true_center = np.array([55, 55])
    true_speed = 1.6424885622140555

    with mp.Pool(len(num_clusters_range)) as pool:
        results = pool.starmap(test_fn_clusters, (
            (
                num_clusters,
                cluster_size_range,
                cluster_spacing_range,
                iterations,
                true_center,
                true_speed
            ) for num_clusters in num_clusters_range)
        )
        errs = np.array(results)
        np.save("./cTDOA_errs.npy", errs)

if __name__ == "__main__":
    main()
