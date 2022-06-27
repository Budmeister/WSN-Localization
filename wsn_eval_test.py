from WSN import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from wsn_eval import *

def plot_edges(wsn : WSN):
    edges = []
    nodes = list(wsn.anchor_nodes)
    V = len(nodes)
    for ii, i in enumerate(nodes[:-1]):
        for j in nodes[ii+1:]:
            ri = wsn.nodes[i]
            rj = wsn.nodes[j]
            d = dist(ri, rj)
            edge = (d, i, j)
            heappush(edges, edge)
    
    for edge in edges:
        n1 = wsn.nodes[edge[1]]
        n2 = wsn.nodes[edge[2]]
        plt.plot([n1[0], n2[0]], [n1[1], n2[1]], c="orange")
    plt.scatter(wsn.nodes[:, 0], wsn.nodes[:, 1], c="blue")
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.show()

wsn = WSN(100, 7, std=0, D=142)
# wsn.reset_nodes()
wsn.nodes = np.array([[25.29637085, 81.0270733 ],
                        [21.00637125, 34.16074866],
                        [40.46185305, 34.6855283 ],
                        [48.27998251, 45.75517048],
                        [16.25076286, 21.58334057],
                        [30.48841588, 97.56500214],
                        [76.96954032, 12.42933106]])
wsn.reset_anchors({1,2,3,4,5,6})
MST = wsn.find_MST()

