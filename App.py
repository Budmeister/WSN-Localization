from tkinter import *
from WSN import *
from WSNAreaWidget import WSNAreaWidget
from global_funcs import *
from wsn_eval import *

class App:
    def __init__(self):
        self.running = False

    def on_exit(self):
        self.running = False

    def reset_nodes(self):
        self.wsn_area.clear_nodes()
        self.wsn_area.clear_est_values()
        self.wsn.reset_nodes()
        self.wsn.reset_anchors()
        self.wsn_area.set_nodes(self.wsn.nodes, self.wsn.anchor_nodes)

    def localize(self, method):
        self.wsn.reset_anchors()
        est_pos = self.wsn.localize(method)
        self.wsn_area.clear_est_values()
        self.wsn_area.set_est_values(est_pos)
        success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
        print(success, error)
    
    def localize_TDOA_continuous(self, signal_type):
        self.wsn.reset_anchors()
        est_pos = self.wsn.localize_TDOA_continuous(signal_type=signal_type)
        self.wsn_area.clear_est_values()
        self.wsn_area.set_est_values(est_pos)
        success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
        print(success, error)

    def show(self):
        if self.running:
            return

        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.wsn = WSN(size=100, N=100)

        zoom = 5
        self.wsn_area = WSNAreaWidget(self.root, 100 * zoom, 100 * zoom, zoom=zoom)
        self.wsn_area.grid(row=0, column=0, columnspan=5)

        big_w = 11
        big_h = 3

        reset_nodes_btn = Button(self.root, text="Reset Nodes", command=self.reset_nodes, width=big_w, height=big_h)
        reset_nodes_btn.grid(row=1, column=0)

        localize_TOA_btn = Button(self.root, text="Localize TOA", command=lambda: self.localize("TOA"), width=big_w, height=big_h)
        localize_TOA_btn.grid(row=1, column=1)

        localize_TDOA_btn = Button(self.root, text="Localize TDOA", command=lambda: self.localize("TDOA"), width=big_w, height=big_h)
        localize_TDOA_btn.grid(row=1, column=2)

        localize_RSS_btn = Button(self.root, text="Localize RSS", command=lambda: self.localize("RSS"), width=big_w, height=big_h)
        localize_RSS_btn.grid(row=1, column=3)

        localize_LAA_btn = Button(self.root, text="Localize TDOA\nContinuous", command=lambda: self.localize_TDOA_continuous("ar"), width=big_w, height=big_h)
        localize_LAA_btn.grid(row=1, column=4)

        option_columns = [
            [
                ("N", 0, 100, 100),
                # ("Fc", 1e8, 1e10, 2.4e9),
                ("num_anchors", 0, 10, 10),
                ("D", 0, 200, 30),
                ("dt", 1, 100, 100, 1, 0.0001),
            ],
            [
                # ("c", 1e8, 3e8, 3e8),
                ("samples", 0, 1000, 100),
                # ("t0", 0, 50, 0, 1),
                ("ar_b", 0.0, 1.0, 0.9, 0.01),
                ("ar_std", 0, 3, 1, 0.01),
            ]
        ]
        def t0_command(num):
            self.wsn.t0 = float(num)
            self.localize("TDOA")
        for c, option_column in enumerate(option_columns):
            gridy = 2
            gridx = c * 2
            for name, min, max, default, *others in option_column:
                if len(others) > 0:
                    resolution = others[0]
                else:
                    resolution = None
                if len(others) > 1:
                    multiplier = others[1]
                else:
                    multiplier = 1
                s_lbl = Label(self.root, text=name)
                s_lbl.grid(row=gridy, column=gridx)
                slider = Scale(self.root, from_=min, to=max, length=50 * zoom, tickinterval=(max - min) / 2, orient=HORIZONTAL)
                slider.set(default)
                option = Option(self.wsn, slider, name, min, max, default, multiplier)
                slider.config(command=option.command, resolution=resolution)
                if name == "t0":
                    slider.config(command=t0_command)
                slider.grid(row=gridy, column=gridx+1)
                gridy += 1
            

        center_window(self.root)
        
        self.running = True
        while self.running:
            self.root.update()


class Option:
    def __init__(self, wsn, slider, name, min, max, default, multiplier):
        self.wsn = wsn
        self.slider = slider
        self.name = name
        self.min = min
        self.max = max
        self.default = default
        self.multiplier = multiplier

    def command(self, num):
        setattr(self.wsn, self.name, self.slider.get() * self.multiplier)

if __name__ == "__main__":
    app = App()
    app.show()