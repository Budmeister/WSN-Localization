from tkinter import *

import numpy as np

class WSNAreaWidget(Canvas):
    def __init__(self, root, width, height, zoom=1, *args, **kw):
        super(WSNAreaWidget, self).__init__(root, width=width, height=height, *args, **kw)
        self.width = width
        self.height = height
        self.zoom = zoom

        self._bg = self.create_rectangle(0, 0, width, height, fill="white")
        self._node_radius = 10

        self._node_elements = []
        self._est_val_elements = []

    def node2pix(self, node: np.ndarray):
        return tuple(node * self.zoom)

    def clear_nodes(self):
        [self.delete(node_element) for node_element in self._node_elements]

    def set_nodes(self, nodes: np.ndarray, anchor_nodes: np.ndarray):
        for i, node in enumerate(nodes):
            pix = self.node2pix(node)
            self._node_elements.append(
                self.create_oval(
                    pix[0] - self._node_radius,
                    pix[1] - self._node_radius,
                    pix[0] + self._node_radius,
                    pix[1] + self._node_radius,
                    fill="",
                    outline="black" if i not in anchor_nodes else "blue"
                )
            )
        self._node_elements = self._node_elements

    def clear_est_values(self):
        [self.delete(est_val_element[0], est_val_element[1]) for est_val_element in self._est_val_elements]

    def set_est_values(self, est_vals: np.ndarray):
        for est_val in est_vals:
            pix = self.node2pix(est_val)
            self._est_val_elements.append(
                (
                    self.create_line(
                        pix[0] - self._node_radius,
                        pix[1] - self._node_radius,
                        pix[0] + self._node_radius,
                        pix[1] + self._node_radius,
                        fill="black"
                    ),
                    self.create_line(
                        pix[0] + self._node_radius,
                        pix[1] - self._node_radius,
                        pix[0] - self._node_radius,
                        pix[1] + self._node_radius,
                        fill="black"
                    )
                )
            )
        self._est_val_elements = self._est_val_elements

