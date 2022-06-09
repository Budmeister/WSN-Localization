from WSN import *

def main():
    wsn = WSN(120, 5, c=100)
    wsn.std = 0
    wsn.nodes = np.array([
        [8, 5],
        [4, 9],
        [3, 3],
        [10, 10],
        [6, 6]
    ])
    wsn.reset_anchors(4)
    est_pos = wsn.localize("TDOA", epochs=1)
    print(est_pos)
    

if __name__ == "__main__":
    main()