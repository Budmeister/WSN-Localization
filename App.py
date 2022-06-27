from tkinter import *
from WSN import *
from WSNAreaWidget import WSNAreaWidget
from global_funcs import *
from wsn_eval import *
from tqdm import tqdm
from PIL import Image, ImageTk

class App:
    def __init__(self):
        self.running = False

    def on_exit(self):
        self.running = False

    def reset_nodes(self):
        self.wsn_area.clear_nodes()
        self.wsn_area.clear_est_values()
        self.wsn_area.clear_other_values()
        self.wsn.reset_nodes()
        self.wsn.reset_anchors()
        self.wsn_area.set_nodes(self.wsn.nodes, self.wsn.anchor_nodes)
    
    def reset_clusters(self):
        self.wsn_area.clear_nodes()
        self.wsn_area.clear_est_values()
        self.wsn_area.clear_other_values()
        self.wsn.reset_clusters()
        self.wsn.reset_anchors()
        self.wsn_area.set_nodes(self.wsn.nodes, self.wsn.anchor_nodes)

    def localize(self, method):
        self.wsn.reset_anchors()
        est_pos = self.wsn.localize(method)
        self.wsn_area.clear_est_values()
        self.wsn_area.set_est_values(est_pos)
        center = np.mean(self.wsn.nodes, axis=0)
        self.wsn_area.set_other_values(np.reshape(center, (1, 2)))
        success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
        print(success, error)
    
    def localize_single_continuous(self, method="mTDOA", signal_type="osc"):
        self.wsn.reset_anchors()
        est_pos = self.wsn.localize_continuous(method=method, signal_type=signal_type)
        self.wsn_area.clear_est_values()
        self.wsn_area.set_est_values(est_pos)
        center = np.mean(self.wsn.nodes, axis=0)
        self.wsn_area.set_other_values(np.reshape(center, (1, 2)))
        success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
        print(success, error)
        
    # Run continuous TDOA on a set of dts and display the errors on a graph
    def localize_several_continuous(self, method="mTDOA", signal_type="osc"):
        self.wsn.reset_anchors()
        dt_to_show = self.wsn.dt
        dt_range = (4, 100)
        multiplier = 0.0001
        errors = np.zeros(dt_range[1] - dt_range[0])
        for dt in tqdm(range(*dt_range)):
            self.wsn.dt = dt * multiplier
            try:
                est_pos = self.wsn.localize_continuous(method=method, signal_type=signal_type, show_error=False)
                success, error = err(est_pos, self.wsn.nodes, self.wsn.anchor_nodes)
                errors[dt-dt_range[0]] = error
                # if dt * multiplier == dt_to_show:
                # self.wsn_area.clear_est_values()
                # self.wsn_area.set_est_values(est_pos)
            except ValueError:
                errors[dt-dt_range[0]] = -1
        plt.plot(range(*dt_range), errors)
        plt.show()
    
    def test_TDOA_vs_continuous(self, signal_type):
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
                est_pos = self.wsn.localize_continuous(signal_type=signal_type, show_error=True)
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

    def show_fn_solution(self):
        mtin.solve_default_fn_equ()
        width, height = self.wsn_area.width, self.wsn_area.height
        data: np.ndarray = mtin.default_fn_sol[-1, 0].flatten()
        x, n = data.max(), data.min()
        data = (data - n) / (x - n)
        data = (data * 255).astype(np.int32)
        data = [(v, v, v) for v in data]
        zoom = 5
        image = Image.new("RGB", (width // zoom, height // zoom))
        image.putdata(data)
        image = image.resize((width, height), resample=Image.BICUBIC)
        image = ImageTk.PhotoImage(image)
        # image = ImageTk.PhotoImage(Image.open("image.png"))
        self.wsn_area.set_bg_image(image)

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
                ("Reset Clusters", self.reset_clusters),
                ("Set FN BG", self.show_fn_solution),
                # ("Localize TOA", lambda: self.localize("TOA")),
                # ("Localize TDOA", lambda: self.localize("TDOA")),
                ("Localize\ntmTDOA", lambda: self.localize("tmTDOA")),
                # ("Localize\nSingle TDOA\nOsc", lambda: self.localize_single_continuous(method="TDOA", signal_type="osc")),
                # ("Test TDOA\nvs continuous\nOsc", lambda: self.test_TDOA_vs_continuous(method="TDOA", signal_type="osc")),
                ("Localize\nSingle cTDOA\nFN", lambda: self.localize_single_continuous(method="cTDOA", signal_type="fn")),
                ("Test TDOA\nvs continuous\nFN", lambda: self.test_TDOA_vs_continuous(method="mTDOA", signal_type="fn")),
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

        option_columns = [
            [
                # ("N", 0, 100, 5),
                # ("num_anchors", 0, 10, 4),
                ("cluster_size", 1, 10, 3),
                ("num_clusters", 1, 10, 3),
                ("D", 0, 142, 142),
                ("dt", 1, 500, 100, 1, 0.0001),
            ],
            [
                # ("c", 1e8, 3e8, 3e8),
                ("message_length", 0, 10, 10, 0.01),
                # ("samples", 0, 1000, 100),
                # ("t0", 0, 50, 0, 1),
                # ("ar_b", 0.0, 1.0, 0.9, 0.01),
                # ("ar_std", 0, 3, 1, 0.01),
                ("std", 0, 100, 10, 1, 0.001),
                # ("epsilon", 0.0, 1.0, 0.25, 0.01),
                ("cluster_spacing", 1, 10, 10),
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
                s_lbl = Label(self.root, text=name + ((" " + str(multiplier)) if multiplier != 1 else ""))
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