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


def main():
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    wsn = WSN(100, 11, D=200)

    num_trials = 100

    noise_factors = np.arange(start=0, stop=0.3, step=0.001)
    successes = []
    errors = []
    for nf in tqdm(noise_factors):
        wsn.std = nf
        avg_success = 0
        avg_error = 0
        for _ in range(num_trials):
            wsn.reset_nodes()
            wsn.reset_anchors()
            est_pos = wsn.localize(method="TDOA")
            success, error = err(est_pos, wsn.nodes, wsn.anchor_nodes)
            avg_success += success
            avg_error += error
        avg_success /= num_trials
        avg_error /= num_trials
        successes.append(avg_success)
        errors.append(avg_error)
    
    plt.figure()
    plt.plot(noise_factors, successes)
    plt.plot(noise_factors, errors)
    plt.show()

if __name__ == "__main__":
    main()
