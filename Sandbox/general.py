from typing import Iterable
import numpy as np
import random

def dist2(node1: np.ndarray, node2: np.ndarray):
    diff = node2 - node1
    return np.dot(diff, diff)

def dist(node1: np.ndarray, node2: np.ndarray):
    return np.sqrt(dist2(node1, node2))

def norm(node: np.ndarray):
    mag = np.sqrt(np.dot(node, node))
    if mag != 0:
        return node / mag
    else:
        raise ValueError()

class WSN:
    def __init__(self, size, N, Fc=2.4, D=30, c=3e8):
        self.size = size
        self.N = N
        self.Fc = Fc * 1e9
        self.D = D
        self.c = c
        self.nodes = np.random.default_rng().random(size=(N, 2)) * size
    
    def localize(self, method, *args, **kwargs):
        if method == "TOA":
            return self.localize_TOA(*args, **kwargs)

    # Get the results of transmitting from the given node
    def transmit(self, i, r0, known_nodes):
        results = []
        for j in known_nodes:
            if i == j:
                continue
            rn = self.nodes[j]
            pn = dist(r0, rn)
            if pn > self.D:
                continue
            tn0 = pn / self.c   # time = distance / rate

            results.append(
                (
                    tn0,
                    norm(rn - r0),
                    rn
                )
            )
        return results

    def localize_TOA(self, epochs=10, anchor_nodes=10):
        est_pos = np.zeros(shape=(self.N, 2))
        if anchor_nodes is not None and isinstance(anchor_nodes, Iterable) and not isinstance(anchor_nodes, np.ndarray):
            known_nodes = np.array(anchor_nodes)
        elif isinstance(anchor_nodes, int):
            known_nodes = np.arange(anchor_nodes)
            anchor_nodes = np.copy(known_nodes)
        # Set anchor nodes
        for n in known_nodes:
            est_pos[n] = self.nodes[n]
        Nmin = 3

        for e in range(epochs):
            for i, r0 in enumerate(self.nodes):
                if i in anchor_nodes:
                    # Do not try to estimate the location of an anchor node
                    continue
                # results holds the known nodes within range of r0
                results = self.transmit(i, r0, known_nodes)
                if len(results) < Nmin:
                    continue
                t10, no1, r1 = results[0]
                H = np.transpose([rn - r1 for n, (tn0, no, rn, *_) in enumerate(results) if n != 0])
                norm_r1_2 = np.dot(r1, r1)
                t10_2 = t10 ** 2
                b = np.transpose([
                    np.dot(rn, rn) - norm_r1_2 - self.c ** 2 * (tn0 ** 2 - t10_2)
                    for n, (tn0, no, rn, *_) in enumerate(results) if n != 0
                ]) * 0.5
                r0_est = np.matmul(np.linalg.pinv(H), b)
                est_pos[i] = r0_est
        return est_pos




if __name__ == "__main__":
    wsn = WSN(100, 100)
    est_pos = wsn.localize_TOA()
    [print((wsn.nodes[i], est_pos[i])) for i in range(100)]
