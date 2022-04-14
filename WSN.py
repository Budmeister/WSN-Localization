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
    def __init__(self, size, N, Fc=2.4e9, D=30, c=3e8, Pt=1):
        self.size = size
        self.N = N
        self.Fc = Fc
        self.D = D
        self.c = c
        self.Pt = Pt
        self.nodes = np.array([])
        self.anchor_nodes = set()
        self.known_nodes = set()

        self.t0 = 0

    def reset_nodes(self):
        self.nodes = np.random.default_rng().random(size=(self.N, 2)) * self.size
        print(self.N)

    def reset_anchors(self, anchor_nodes=10):
        if anchor_nodes is not None and isinstance(anchor_nodes, Iterable) and not isinstance(anchor_nodes, np.ndarray):
            self.known_nodes = set(anchor_nodes)
            self.anchor_nodes = anchor_nodes
        elif isinstance(anchor_nodes, int):
            self.known_nodes = set(range(anchor_nodes))
            self.anchor_nodes = set(self.known_nodes)
        else:
            raise ValueError("Invalid type for anchor_nodes. Should be Iterable or int.")

    # Get the results of transmitting from the given node
    def transmit(self, i, r0, t0=0):
        results = []
        for j in self.known_nodes:
            if i == j:
                continue
            rn = self.nodes[j]
            pn = dist(r0, rn)
            if pn > self.D:
                continue
            # time = distance / rate
            tn0 = pn / self.c + t0
            # Rearranged equation (8)
            Pr = self.Pt / (pn * 4 * np.pi * self.Fc / self.c) ** 2

            results.append(
                (
                    rn,
                    tn0,
                    norm(rn - r0),
                    Pr
                )
            )
        return results

    def get_TDOA_H_and_b(self, results, *args, **kwargs):
        r1, t10, no1, *_ = results[0]

        # No transpose, because "rn - r1" are being inserted as rows
        H = np.array([rn - r1 for n, (rn, *_) in enumerate(results) if n != 0])
        H = np.concatenate((H, np.transpose([[self.c * (tn0 - t10) for n, (rn, tn0, *_) in enumerate(results) if n != 0]])), 1)
        norm_r1_2 = np.dot(r1, r1)
        t10_2 = t10 ** 2
        # b = np.array([
        #     [np.dot(rn, rn) - norm_r1_2 - self.c ** 2 * ((tn0 + time_offset) ** 2 - t10_2)]
        #     for n, (rn, tn0, no, *_) in enumerate(results) if n != 0
        # ]) * 0.5
        b = np.array([
            # [np.dot(rn, rn) - norm_r1_2 - self.c ** 2 * ((tn0 - t10) ** 2 + 2 * t10 * (tn0 - t10))]
            # [(self.c * (tn0 - t10)) **2 - norm_r1_2 - np.dot(rn, rn)]
            [np.dot(rn, rn) - norm_r1_2 - self.c ** 2 * (tn0 - t10) ** 2]
            for n, (rn, tn0, no, *_) in enumerate(results) if n != 0
        ]) * 0.5
        return H, b

    def get_TOA_H_and_b(self, results, *args, **kwargs):
        r1, t10, no1, *_ = results[0]

        # No transpose, because "rn - r1" are being inserted as rows
        H = np.array([rn - r1 for n, (rn, *_) in enumerate(results) if n != 0])
        norm_r1_2 = np.dot(r1, r1)
        t10_2 = t10 ** 2
        b = np.array([
            # [np.dot(rn, rn) - norm_r1_2 - self.c ** 2 * ((tn0 - t10) ** 2 + 2 * t10 * (tn0 - t10))]
            [norm_r1_2 - np.dot(rn, rn) + self.c ** 2 * (tn0 ** 2 - t10_2)]
            for n, (rn, tn0, no, *_) in enumerate(results) if n != 0
        ]) * 0.5
        return H, b
    
    def power_to_dist(self, Pr):
        return self.c / (4 * np.pi * self.Fc) * np.sqrt(self.Pt / Pr)

    def get_RSS_H_and_b(self, results, *args, **kwargs):
        r1, t10, no1, Pr1, *_ = results[0]

        # No transpose, because "rn - r1" are being inserted as rows
        H = np.array([rn - r1 for n, (rn, *_) in enumerate(results) if n != 0])
        norm_r1_2 = np.dot(r1, r1)
        P1_2 = self.power_to_dist(Pr1) ** 2
        b = np.array([
            [np.dot(rn, rn) - norm_r1_2 - (self.power_to_dist(Prn) ** 2 - P1_2)]
            for n, (rn, tn0, no, Prn, *_) in enumerate(results) if n != 0
        ]) * 0.5
        return H, b

    def get_LAA_H_and_b(self, results, *args, **kwargs):
        pass


    def localize(self, method="TOA", epochs=10):
        est_pos = np.zeros(shape=(self.N, 2))

        # LAA currently unimplemented
        if method == "LAA":
            return est_pos

        # Set anchor nodes
        for n in self.known_nodes:
            est_pos[n] = self.nodes[n]
        method_data = {
            "TOA": (3, self.get_TOA_H_and_b),
            "TDOA": (3, self.get_TDOA_H_and_b),
            "RSS": (3, self.get_RSS_H_and_b),
            "LAA": (4, self.get_LAA_H_and_b)
        }
        if method not in method_data:
            raise ValueError(f"Unsupported method: {method}")
        Nmin, get_H_and_b = method_data[method]

        for e in range(epochs):
            for i, r0 in enumerate(self.nodes):
                if i in self.anchor_nodes:
                    # Do not try to estimate the location of an anchor node
                    continue
                # results holds the known nodes within range of r0
                results = self.transmit(i, r0, t0=self.t0 * 1e-8 if method == "TDOA" else 0)
                if len(results) < Nmin:
                    continue
                
                H, b = get_H_and_b(results)
                r0_est = np.matmul(np.linalg.pinv(H), b)
                est_pos[i] = r0_est.flatten()[:2]   # TDOA returns a triple
                self.known_nodes.add(i)
        return est_pos




if __name__ == "__main__":
    wsn = WSN(size=100, N=300)
    wsn.reset_nodes()
    wsn.reset_anchors(10)
    est_pos = wsn.localize("TOA")
    [print((wsn.nodes[i], est_pos[i])) for i in range(100)]
