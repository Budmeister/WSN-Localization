from tkinter import *

import numpy as np

class WSNAreaWidget(Canvas):
    def __init__(self, root, width, height, zoom=1, *args, **kw):
        super(WSNAreaWidget, self).__init__(root, width=width, height=height, *args, **kw)
        self.width = width
        self.height = height
        self.zoom = zoom

        self._bg = self.create_rectangle(0, 0, width, height, fill="white")
        self._bg_image = None
        self._bg_image_obj = None
        self._node_radius = 10

        self._node_elements = []
        self._est_val_elements = []
        self._other_val_elements = []

        self.default_color = "black"
        self.default_color_with_bg = "red"
        self.other_value_color = "orange"
        self.anchor_color ="blue"


    def set_bg_image(self, image):
        self.clear_bg_image()
        self._bg_image = self.create_image(250, 250, image=image, anchor=CENTER)
        self._bg_image_obj = image
        self.tag_lower(self._bg_image)
        # self.tag_lower(self._bg)
        self.delete(self._bg)

    def clear_bg_image(self):
        self.delete(self._bg_image)
        self._bg_image = None
        self._bg_image_obj = None

    def node2pix(self, node: np.ndarray):
        return tuple(node * self.zoom)

    def clear_nodes(self):
        [self.delete(node_element) for node_element in self._node_elements]
        self._node_elements = []

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
                    outline=
                        (self.default_color_with_bg if self._bg_image is not None else self.default_color)
                        if i not in anchor_nodes else self.anchor_color
                )
            )
        self._node_elements = self._node_elements

    def clear_est_values(self):
        [self.delete(est_val_element[0], est_val_element[1]) for est_val_element in self._est_val_elements]
        self._est_val_elements = []

    def set_est_values(self, est_vals: np.ndarray):
        for est_val in est_vals:
            pix = self.node2pix(est_val)
            color = self.default_color_with_bg if self._bg_image is not None else self.default_color
            self._est_val_elements.append(
                (
                    self.create_line(
                        pix[0] - self._node_radius,
                        pix[1] - self._node_radius,
                        pix[0] + self._node_radius,
                        pix[1] + self._node_radius,
                        fill=color
                    ),
                    self.create_line(
                        pix[0] + self._node_radius,
                        pix[1] - self._node_radius,
                        pix[0] - self._node_radius,
                        pix[1] + self._node_radius,
                        fill=color
                    )
                )
            )
    
    def clear_other_values(self):
        [self.delete(other_val_element[0], other_val_element[1]) for other_val_element in self._other_val_elements]
        self._other_val_elements = []
    
    def set_other_values(self, other_vals: np.ndarray, color=None):
        for other_val in other_vals:
            pix = self.node2pix(other_val)
            if color is None:
                color = self.other_value_color
            else:
                self.other_value_color = color
            self._other_val_elements.append(
                (
                    self.create_line(
                        pix[0] - self._node_radius,
                        pix[1] - self._node_radius,
                        pix[0] + self._node_radius,
                        pix[1] + self._node_radius,
                        fill=color
                    ),
                    self.create_line(
                        pix[0] + self._node_radius,
                        pix[1] - self._node_radius,
                        pix[0] - self._node_radius,
                        pix[1] + self._node_radius,
                        fill=color
                    )
                )
            )

