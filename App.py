from tkinter import *
from WSN import *
from WSNAreaWidget import WSNAreaWidget
from global_funcs import *
from wsn_eval import *
from tqdm import tqdm

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
    
    def localize_single_TDOA(self, signal_type):
        self.wsn.reset_anchors()
        est_pos = self.wsn.localize_TDOA_continuous(signal_type=signal_type)
        self.wsn_area.clear_est_values()
        self.wsn_area.set_est_values(est_pos)
        success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
        print(success, error)
        
    # Run continuous TDOA on a set of dts and display the errors on a graph
    def localize_TDOA_continuous(self, signal_type):
        self.wsn.reset_anchors()
        dt_to_show = self.wsn.dt
        dt_range = (4, 100)
        multiplier = 0.0001
        errors = np.zeros(dt_range[1] - dt_range[0])
        for dt in tqdm(range(*dt_range)):
            self.wsn.dt = dt * multiplier
            try:
                est_pos = self.wsn.localize_TDOA_continuous(signal_type=signal_type, show_error=False)
                success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
                errors[dt-dt_range[0]] = error
                # if dt * multiplier == dt_to_show:
                # self.wsn_area.clear_est_values()
                # self.wsn_area.set_est_values(est_pos)
            except ValueError:
                errors[dt-dt_range[0]] = -1
        plt.plot(range(*dt_range), errors)
        plt.show()
    
    def test_mTDOA_vs_continuous(self, signal_type):
        self.wsn.reset_anchors()

        # Continuous
        dts = (10, 50)
        cont_succs = []
        cont_errs = []
        span = 5
        multiplier = 0.0001
        for base_dt in tqdm(dts):
            avg_success = 0
            avg_err = 0
            for dt in range(base_dt - span, base_dt + span):
                self.wsn.dt = dt * multiplier
                est_pos = self.wsn.localize_TDOA_continuous(signal_type=signal_type, show_error=True)
                success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
                avg_success += success
                avg_err += error
            avg_success /= span * 2
            cont_succs.append(avg_success)
            avg_err /= span * 2
            cont_errs.append(avg_err)

        # mTDOA
        num_trials = 100
        mtdoa_success = 0
        mtdoa_err = 0
        
        for _ in range(num_trials):
            est_pos = self.wsn.localize("TDOA", epochs=1)
            success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
            mtdoa_success += success
            mtdoa_err += error
        mtdoa_success /= num_trials
        mtdoa_err /= num_trials
        
        self.wsn_area.clear_est_values()
        self.wsn_area.set_est_values(est_pos)

        print()
        print("Continuous results:")
        [print(f"dt={dt/10}ms, success={success}, error={error}") for dt, success, error in zip(dts, cont_succs, cont_errs)]
        print()
        print(f"mTDOA results:")
        print(f"success={mtdoa_success}")
        print(f"error={mtdoa_err}")
        print()


        

    def show(self):
        if self.running:
            return

        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.wsn = WSN(size=100, N=100)

        zoom = 5
        self.wsn_area = WSNAreaWidget(self.root, 100 * zoom, 100 * zoom, zoom=zoom)
        self.wsn_area.grid(row=0, column=0, columnspan=7)

        big_w = 11
        big_h = 3

        button_rows = [
            [
                ("Reset Nodes", self.reset_nodes),
                ("Localize TOA", lambda: self.localize("TOA")),
                ("Localize TDOA", lambda: self.localize("TDOA")),
                ("Localize\nSingle TDOA\nOsc", lambda: self.localize_single_TDOA("osc")),
                ("Test mTDOA\nvs continuous\nOsc", lambda: self.test_mTDOA_vs_continuous("osc")),
                ("Localize\nSingle TDOA\nAR", lambda: self.localize_single_TDOA("ar")),
                ("Test mTDOA\nvs continuous\nAR", lambda: self.test_mTDOA_vs_continuous("ar")),
            ]
        ]

        gridy = 0
        for r, button_row in enumerate(button_rows):
            gridx = 0
            gridy = r + 1
            for name, command, *others in button_row:
                button = Button(self.root, text=name, command=command, width=big_w, height=big_h)
                button.grid(row=gridy, column=gridx)
                gridx += 1
        gridy += 1

        # reset_nodes_btn = Button(self.root, text="Reset Nodes", command=self.reset_nodes, width=big_w, height=big_h)
        # reset_nodes_btn.grid(row=1, column=0)

        # localize_TOA_btn = Button(self.root, text="Localize TOA", command=lambda: self.localize("TOA"), width=big_w, height=big_h)
        # localize_TOA_btn.grid(row=1, column=1)

        # localize_TDOA_btn = Button(self.root, text="Localize TDOA", command=lambda: self.localize("TDOA"), width=big_w, height=big_h)
        # localize_TDOA_btn.grid(row=1, column=2)

        # localize_RSS_btn = Button(self.root, text="Localize\nSingle TDOA", command=lambda: self.localize_single_TDOA("osc"), width=big_w, height=big_h)
        # localize_RSS_btn.grid(row=1, column=3)

        # localize_LAA_btn = Button(self.root, text="Test mTDOA\nvs continuous", command=lambda: self.test_mTDOA_vs_continuous("osc"), width=big_w, height=big_h)
        # localize_LAA_btn.grid(row=1, column=4)

        option_columns = [
            [
                ("N", 0, 100, 5),
                # ("Fc", 1e8, 1e10, 2.4e9),
                ("num_anchors", 0, 10, 4),
                ("D", 0, 142, 142),
                ("dt", 1, 500, 100, 1, 0.0001),
            ],
            [
                # ("c", 1e8, 3e8, 3e8),
                ("message_length", 0, 10, 10, 0.01),
                # ("samples", 0, 1000, 100),
                # ("t0", 0, 50, 0, 1),
                ("ar_b", 0.0, 1.0, 0.9, 0.01),
                # ("ar_std", 0, 3, 1, 0.01),
                ("std", 0, 100, 10, 1, 0.001),
                ("epsilon", 0.0, 1.0, 0.25, 0.01),
            ]
        ]
        def t0_command(num):
            self.wsn.t0 = float(num)
            self.localize("TDOA")
        option_start_y = gridy
        for c, option_column in enumerate(option_columns):
            gridy = option_start_y
            gridx = c * 3
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
                option.command(default)
                if name == "t0":
                    slider.config(command=t0_command)
                slider.grid(row=gridy, column=gridx+1, columnspan=2)
                gridy += 1
            

        center_window(self.root)
        
        self.running = True
        plt.figure()
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