from typing import Iterable
import numpy as np
from mutual_information import *
import mutual_information as mtin
# Import as mtin to access dynamic global variables from mtin

from heapq import heappush, heappop

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
    def __init__(
        self,
        size,
        N,
        Fc=2.4e9,
        D=30,
        c=100,
        Pt=1,
        std=0,
        samples=1000,
        dt=0.01,
        ar_b=0.5,
        ar_std=1,
        epsilon=0.25,
        num_clusters=3,
        cluster_size=3,
        cluster_spacing=4,
        verbose=True
    ):
        self.size = size
        self.N = N
        self.Fc = Fc
        self.D = D
        self.c = c
        self.Pt = Pt
        self.nodes = np.zeros((0, 2))
        self.anchor_nodes = set()
        self.known_nodes = set()
        self.num_anchors = 0
        self.std = std

        self.samples = samples
        self.message_length = None
        self.dt = dt

        # ar
        self.ar_b = ar_b
        self.ar_std = ar_std

        # osc
        self.epsilon = epsilon

        # clusters
        self.num_clusters = num_clusters
        self.cluster_size = cluster_size
        self.cluster_spacing = cluster_spacing

        self.t0 = 0
        self.verbose = verbose
        self.MST = None
    
    def printv(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def reset_nodes(self):
        self.nodes = np.random.default_rng().random(size=(self.N, 2)) * self.size
        self.MST = None
        self.printv(self.N)
    
    def reset_clusters(self):
        clusters = np.random.uniform(0, self.size, size=(self.num_clusters, 2))
        self.N = self.num_clusters * self.cluster_size
        self.num_anchors = self.N
        self.nodes = np.random.normal(
            loc=[cluster for cluster in clusters for _ in range(self.cluster_size)],
            scale=self.cluster_spacing,
            size=(self.N, 2)
        )
        self.nodes = np.max(
            [self.nodes, np.full_like(self.nodes, 0)], axis=0
        )
        self.nodes = np.min(
            [self.nodes, np.full_like(self.nodes, self.size-1)], axis=0
        )
        self.MST = None
        self.printv(self.num_clusters, self.cluster_size)
    
    def clear_nodes(self):
        self.N = 0
        self.num_anchors = 0
        self.nodes = np.zeros((0, 2))
        self.MST = None
        self.printv(0)

    def reset_anchors(self, anchor_nodes=None):
        self.MST = None
        if anchor_nodes is None:
            anchor_nodes = self.num_anchors
        if isinstance(anchor_nodes, Iterable) and not isinstance(anchor_nodes, np.ndarray):
            self.known_nodes = set(anchor_nodes)
            self.anchor_nodes = set(anchor_nodes)
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
            tn0 = pn / self.c + t0 + np.random.normal(scale=self.std)
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
    
    def transmit_continuous(self, i=None, r0=None, signal_type="ar", return_taus=False):
        signal_types = {
            "ar": lambda: get_one_way_ar_data(self.samples, self.ar_b, taus, self.ar_std),
            "osc": lambda: get_one_way_osc_data(self.samples, taus, self.epsilon, a=4),
            "fn": lambda: get_fn_data(self.samples, receivers, use_default_equ=True)
        }
        get_data = signal_types[signal_type]
        if signal_type == "ar" or signal_type == "osc":
            if r0 is None or i is None:
                raise ValueError("r0 and i should be provided for ar and osc signal types")
            receivers = []
            taus = []
            for j in self.known_nodes:
                if j == i:
                    continue
                rn = self.nodes[j]
                pn = dist(r0, rn)
                tn0 = pn / self.c
                tau = int(tn0 / self.dt)
                if pn > self.D:
                    continue
                receivers.append(rn)
                taus.append(tau)
            xs = get_data()
            results = list(zip(receivers, xs.T[1:]))
            if return_taus:
                return results, taus
            else:
                return results
        elif signal_type == "fn":
            if return_taus:
                raise ValueError("Unable to return taus for fn signal type")
            receivers = []
            for j in self.known_nodes:
                rn = self.nodes[j]
                receivers.append(rn)
            xs = get_data()[:, :, 0]
            results = list(zip(receivers, xs))
            return results
        else:
            raise ValueError(f"Invalid type {signal_type}")

    def get_TDOA_H_and_b(self, results, *args, **kwargs):
        r1, t10, *_ = results[0]

        # No transpose, because "rn - r1" are being inserted as rows
        H = np.array([rn - r1 for n, (rn, *_) in enumerate(results) if n != 0])
        H = np.concatenate((H, np.transpose([[self.c * (tn0 - t10) for n, (rn, tn0, *_) in enumerate(results) if n != 0]])), 1)
        norm_r1_2 = np.dot(r1, r1)
        t10_2 = t10 ** 2
        b = np.array([
            # [np.dot(rn, rn) - norm_r1_2 - self.c ** 2 * ((tn0 - t10) ** 2 + 2 * t10 * (tn0 - t10))]
            [np.dot(rn, rn) - norm_r1_2 - self.c ** 2 * (tn0 - t10) ** 2]
            for n, (rn, tn0, *_) in enumerate(results) if n != 0
        ]) * 0.5
        return H, b

    def find_MST(self, locations=None):
        if locations is None:
            locations = self.nodes
        if self.MST is not None and locations is self.nodes:
            return self.MST
        MST = set()
        subtrees = []
        subtrees_to_remove = []
        edges = []
        nodes = list(range(len(locations)))
        V = len(nodes)
        for ii, i in enumerate(nodes[:-1]):
            for j in nodes[ii+1:]:
                ri = locations[i]
                rj = locations[j]
                d = dist(ri, rj)
                edge = (d, i, j)
                heappush(edges, edge)

        counter = 0
        while counter < V - 1:
            edge = heappop(edges)
            subtree : set = None
            legal = True
            added = False
            for s in subtrees:
                edge1 = edge[1] in s
                edge2 = edge[2] in s
                if edge1 and edge2:
                    legal = False
                    break
                if edge1 or edge2:
                    s.add(edge[1])
                    s.add(edge[2])
                    MST.add(edge)
                    added = True
                    if subtree is None:
                        subtree = s
                    else:
                        subtree.update(s)
                        subtrees_to_remove.append(s)
                        break
            if not legal:
                continue
            if not added:
                subtrees.append({edge[1], edge[2]})
                MST.add(edge)
                counter += 1
                continue

            s = 0
            j = 0
            while j < len(subtrees) and s < len(subtrees_to_remove):
                if subtrees[j] is subtrees_to_remove[s]:
                    subtrees.pop(j)
                    s += 1
                else:
                    j += 1
            counter += 1
        # assert len(subtrees) == 1
        if locations is self.nodes:
            self.MST = MST
        return MST

    def get_mTDOA_H_and_b_with_tree(self, results, *args, **kwargs):
        r1, t10, *_ = results[0]
        t10_2 = t10 ** 2

        Hb = [self.get_mTDOA_H_and_b_row(results[i], results[j]) for _, i, j in self.find_MST([r for r, *_ in results])]
        H = np.zeros((len(Hb), 4))
        b = np.zeros(len(Hb))

        # Hb = [(Hrow0, b0), (Hrow1, b1), ...]
        # H = [Hrow0, Hrow1, ...]
        # b = [b0, b1, ...]
        # This line unpacks the tuples from Hb into H and b
        [0 for i, (H[i], b[i]) in enumerate(Hb)]
        
        return H, b

    
    def get_mTDOA_H_and_b_row(self, res_1, res_n):
        r1, t10, *_ = res_1
        rn, tn0, *_ = res_n
        Hn = np.array([*(2 * (rn - r1)), -2 * (tn0 - t10), tn0 ** 2 - t10 ** 2])
        bn = np.dot(rn, rn) - np.dot(r1, r1)
        return Hn, bn


    def get_mTDOA_H_and_b(self, results, *args, **kwargs):
        r1, t10, *_ = results[0]
        t10_2 = t10 ** 2
        
        # No transpose, because "rn - r1" are being inserted as rows
        H = np.array([rn - r1 for n, (rn, *_) in enumerate(results) if n != 0])
        if H.ndim == 1:
            raise ValueError
        H = np.concatenate((2 * H, np.array([[-2 * (tn0 - t10), tn0 ** 2 - t10 ** 2] for n, (rn, tn0, *_) in enumerate(results) if n != 0])), 1)
        # H = np.concatenate((2 * H, np.array([[tn0 ** 2 - t10 ** 2] for n, (rn, tn0, *_) in enumerate(results) if n != 0])), 1)
        norm_r1_2 = np.dot(r1, r1)
        b = np.array([
            # [np.dot(rn, rn) - norm_r1_2 - self.c ** 2 * ((tn0 - t10) ** 2 + 2 * t10 * (tn0 - t10))]
            [np.dot(rn, rn) - norm_r1_2]
            for n, (rn, tn0, *_) in enumerate(results) if n != 0
        ])
        return H, b
    
    def get_cTDOA_H_and_b(self, results, max_tau=200, *args, **kwargs):
        num_clusters = self.num_clusters
        cluster_size = self.cluster_size
        nodes = np.array([node for node, *_ in results])
        clusters = np.array([nodes[c * cluster_size:(c+1) * cluster_size] for c in range(num_clusters)])
        if clusters.ndim != 3:
            raise ValueError("Not enough nodes")
        dt = default_fn_equ_params["dt"]
        H = np.zeros((num_clusters * (cluster_size-1), num_clusters + 3))
        b = np.zeros(num_clusters * (cluster_size-1))
        shifts = np.arange(-max_tau, max_tau, 1)
        for c in range(num_clusters):
            rc = clusters[c][0]
            sigc = results[c * cluster_size][1]
            for n in range(1, cluster_size):
                    rn = clusters[c][n]
                    sign = results[c * cluster_size + n][1]
                    # Generate time
                    r0 = np.array([55, 55])
                    # tn = -(dist(rn, r0) - dist(rc, r0)) / self.c
                    tn = get_time_delay(np.transpose([sign, sigc]), shifts, closest_to_zero=True, ensure_one_peak=True, show_error=True) * dt
                    
                    cc = 1.6424885622140555
                    center = np.array([55, 55])
                    real_dt = (dist(rn, center) - dist(rc, center)) / cc
                    self.printv(tn, real_dt, rc, rn)

                    Hn = np.zeros(num_clusters + 3)
                    Hn[:2] = 2 * (rn - rc)
                    Hn[2] = tn ** 2
                    Hn[c + 3] = -2 * tn
                    H[c * (cluster_size - 1) + n - 1] = Hn
                    b[c * (cluster_size - 1) + n - 1] = rn @ rn - rc @ rc
        return H, b

    def get_TOA_H_and_b(self, results, *args, **kwargs):
        r1, t10, *_ = results[0]

        # No transpose, because "rn - r1" are being inserted as rows
        H = np.array([rn - r1 for n, (rn, *_) in enumerate(results) if n != 0])
        norm_r1_2 = np.dot(r1, r1)
        t10_2 = t10 ** 2
        b = np.array([
            [norm_r1_2 - np.dot(rn, rn) + self.c ** 2 * (tn0 ** 2 - t10_2)]
            for n, (rn, tn0, *_) in enumerate(results) if n != 0
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

    def localize_fn_continuous(self, method="mTDOA", return_full_array=False, *args, **kwargs):
        self.dt = default_fn_equ_params["dt"]
        methods = {
            "mTDOA": self.get_mTDOA_H_and_b,
            "cTDOA": self.get_cTDOA_H_and_b,
            "TDOA": self.get_TDOA_H_and_b
        }
        get_H_and_b = methods[method]
        # max_tau = min(int(self.D / self.dt), self.samples) - 1
        max_tau = 200
        shifts = np.arange(-max_tau, max_tau, 1)

        if method == "cTDOA":
            results = self.transmit_continuous(signal_type="fn")
            H, b = get_H_and_b(results, max_tau=max_tau)
            est_pos = np.linalg.pinv(H) @ b
            self.printv(est_pos)
            if return_full_array:
                est_pos = est_pos.flatten()
                return est_pos.reshape((1, len(est_pos)))
            return est_pos.flatten()[:2].reshape((1, 2))


        nodes = np.array([self.nodes[n] for n in self.known_nodes])
        MST = self.find_MST(nodes)
        nodes = [tuple(node) for node in nodes]
        tree_dict = {node: [] for node in nodes}
        for d, n1, n2 in MST:
            tree_dict[nodes[n1]].append(n2)
            tree_dict[nodes[n2]].append(n1)
        r0 = nodes[0]
        results = self.transmit_continuous(signal_type="fn")
        results = {tuple(result[0]): result[1] for result in results}

        delays = {r0: 0}
        def get_child_delays(root):
            children = tree_dict[root]
            sigr = results[root]
            for child in children:
                rc = nodes[child]
                if rc in delays:
                    continue
                sigc = results[rc]
                delay = get_time_delay(np.transpose([sigc, sigr]), shifts=shifts, ensure_one_peak=False)
                delays[rc] = delays[root] + delay * self.dt
                get_child_delays(rc)
        get_child_delays(r0)

        results = [(np.array(rn), delays[rn]) for rn in delays]
        H, b = get_H_and_b(results)
        est_pos = np.linalg.pinv(H) @ b
        self.printv(est_pos)
        if return_full_array:
            est_pos = est_pos.flatten()
            return est_pos.reshape((1, len(est_pos)))
        return est_pos.flatten()[:2].reshape((1, 2))



    def localize_continuous(self, method="mTDOA", signal_type="osc", epochs=1, show_error="Error", return_full_array=False):
        if signal_type == "fn":
            return self.localize_fn_continuous(method=method, return_full_array=return_full_array)
        methods = {
            "mTDOA": self.get_mTDOA_H_and_b,
            "TDOA": self.get_TDOA_H_and_b
        }
        get_H_and_b = methods[method]

        est_pos = np.zeros(shape=(self.N, 2)) if not return_full_array else [[0, 0] for _ in range(self.N)]
        for n in self.known_nodes:
            est_pos[n] = self.nodes[n]
        
        if self.message_length is not None:
            self.samples = int(self.message_length / self.dt)

        Nmin = 3
        for e in range(epochs):
            for i, r0 in enumerate(self.nodes):
                if i in self.anchor_nodes:
                    # Do not try to estimate the location of an anchor node
                    continue
                results = self.transmit_continuous(i, r0, signal_type=signal_type)
                # results, taus = self.transmit_continuous(i, r0, signal_type=signal_type, return_taus=True)
                if len(results) < Nmin:
                    continue
                new_results = [(results[0][0], 0)]
                sig0 = results[0][1]
                max_tau = min(int(self.D / self.c / self.dt), self.samples) - 1
                shifts = np.arange(-max_tau, max_tau, 1)
                for _, result in enumerate(results, 1):
                    xs = np.transpose([sig0, result[1]])
                    tau = get_time_delay(xs, shifts, show_error=show_error)
                    new_results.append((result[0], -tau * self.dt))
                
                H, b = get_H_and_b(new_results)
                r0_est = np.matmul(np.linalg.pinv(H), b)
                pos = r0_est.flatten()
                if not return_full_array:
                    pos = est_pos[:2]   # TDOA returns a triple
                est_pos[i] = pos
                self.known_nodes.add(i)
        est_pos = np.asarray(est_pos)
        return est_pos




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
            "TDOA": (4, self.get_TDOA_H_and_b),
            "mTDOA": (5, self.get_mTDOA_H_and_b),
            "tmTDOA": (5, self.get_mTDOA_H_and_b_with_tree),
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
                results = self.transmit(i, r0, t0=self.t0 * 1e-8 if method == "TDOA" or method == "mTDOA" else 0)
                if len(results) < Nmin:
                    continue
                
                H, b = get_H_and_b(results)
                r0_est = np.matmul(np.linalg.pinv(H), b)
                est_pos[i] = r0_est.flatten()[:2]   # TDOA returns a triple
                self.known_nodes.add(i)
        return est_pos




def main():
    wsn = WSN(size=100, N=300)
    wsn.reset_nodes()
    wsn.reset_anchors(10)
    est_pos = wsn.localize("TOA")
    [print((wsn.nodes[i], est_pos[i])) for i in range(100)]

if __name__ == "__main__":
    main()
