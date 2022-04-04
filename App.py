from tkinter import *
from WSN import *
from WSNAreaWidget import WSNAreaWidget
from global_funcs import *

class App:
    def __init__(self):
        self.running = False

    def reset_nodes(self):
        self.wsn_area.clear_nodes()
        self.wsn_area.clear_est_values()
        self.wsn.reset_nodes()
        self.wsn_area.set_nodes(self.wsn.nodes, self.wsn.anchor_nodes)

    def localize(self, method):
        est_pos = self.wsn.localize(method)
        self.wsn_area.clear_est_values()
        self.wsn_area.set_est_values(est_pos)

    def show(self):
        if self.running:
            return

        self.root = Tk()
        self.wsn = WSN(size=100, N=100)

        self.wsn_area = WSNAreaWidget(self.root, 800, 800, zoom=8)
        self.wsn_area.grid(row=0, column=0, columnspan=4)

        big_w = 11
        big_h = 3

        reset_nodes_btn = Button(self.root, text="Reset Nodes", command=self.reset_nodes, width=big_w, height=big_h)
        reset_nodes_btn.grid(row=1, column=0)

        localize_TOA_btn = Button(self.root, text="Localize TOA", command=lambda: self.localize("TOA"), width=big_w, height=big_h)
        localize_TOA_btn.grid(row=1, column=1)

        center_window(self.root)
        
        self.running = True
        while self.running:
            self.root.update()

if __name__ == "__main__":
    app = App()
    app.show()