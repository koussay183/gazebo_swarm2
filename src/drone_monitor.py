"""
PFE - Swarm Drones MARL — Gazebo Harmonic edition
Moniteur v5 — Trajectoires + Vision RL + Contrôle + Rapports
Tkinter tourne dans le thread principal (requis sur Linux).
Camera tab: position/battery info panels (no PyBullet, Gazebo GUI handles 3D view).
"""
import threading
import math
import time

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TK = True
except Exception:
    HAS_TK = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ROS2 imports — optional (wrapped so the monitor still works without ROS2)
try:
    import rclpy
    from rclpy.node import Node as RosNode
    from geometry_msgs.msg import Pose as RosPose
    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False

# Config constants — must be importable from the project root
try:
    from config import (
        NUM_DRONES, DRONE_COLORS, DRONE_COLOR_NAMES,
        OBSTACLES, PATROL_ZONE, SENSOR_RANGE,
        ENCIRCLE_RADIUS, BIRD_SAFETY_DISTANCE, MAX_SPEED,
    )
except ImportError:
    # Sensible defaults if config not found
    NUM_DRONES         = 4
    DRONE_COLORS       = [[1,0,0,1],[0,0,1,1],[0,0.8,0,1],[1,1,0,1]]
    DRONE_COLOR_NAMES  = ["Rouge","Bleu","Vert","Jaune"]
    OBSTACLES          = []
    PATROL_ZONE        = {"x_min": -25, "x_max": 25, "y_min": -25, "y_max": 25}
    SENSOR_RANGE       = 12.0
    ENCIRCLE_RADIUS    = 5.0
    BIRD_SAFETY_DISTANCE = 3.5
    MAX_SPEED          = 2.5


class DroneMonitor:
    """Tkinter-based mission monitor for the Gazebo swarm simulation."""

    # ──────────────────────────────────────────────────────────────────────
    #  Construction
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self, sim):
        self.sim     = sim
        self.running = False

        # Per-drone trajectory history {drone_id: {x: [...], y: [...]}}
        self.histories = {i: {"x": [], "y": []} for i in range(NUM_DRONES)}

        # Thread-safe report queue
        self._report_queue = []
        self._report_lock  = threading.Lock()
        self._report_text  = None   # tk.Text widget, created in _build_reports

        # Thread-safe snapshot complet (écrit par notify_step @ 50 Hz)
        self._data_lock      = threading.Lock()
        self._drone_snap     = []
        self._intruders_snap = []
        self._birds_snap     = []
        self._sim_time       = 0.0
        self._phase          = "INIT"
        self._formation      = "v_shape"
        self._locked         = False
        self._leader_id      = 0

        # Trajectory plot state
        self._traj_first   = True

    # ──────────────────────────────────────────────────────────────────────
    #  Public API called from the simulation thread
    # ──────────────────────────────────────────────────────────────────────

    def notify_step(self, drones, sim_time: float):
        """Appelé par _sim_loop à chaque step (50 Hz). Snapshot thread-safe complet :
        drones, intrus, oiseaux, phase — tout ce que le moniteur affiche."""
        snap = []
        for d in drones:
            snap.append({
                "id":      d.id,
                "x":       float(d.x),
                "y":       float(d.y),
                "z":       float(d.z),
                "battery": float(d.battery),
                "state":   str(d.state),
                "tx":      float(getattr(d, "tx", 0.0)),
                "ty":      float(getattr(d, "ty", 0.0)),
                "is_leader": d.id == getattr(
                    getattr(self.sim, "election", None), "leader_id", -1),
                "marked":  bool(getattr(d, "is_marked", False)),
            })
        # Snapshot intrus
        intruders_snap = [
            {"pos": list(i["pos"]), "detected": i.get("detected", False),
             "formation": i.get("formation", "?")}
            for i in getattr(self.sim, "intruders", [])
        ]
        # Snapshot oiseaux
        birds_snap = [
            {"pos": list(b["pos"])}
            for b in getattr(self.sim, "birds", [])
        ]
        phase = getattr(self.sim, "phase", "INIT")
        try:
            formation = self.sim.formation_mgr.formation
            locked    = bool(getattr(self.sim, "formation_locked", False))
            leader_id = self.sim.election.leader_id
        except Exception:
            formation, locked, leader_id = "v_shape", False, 0

        with self._data_lock:
            self._drone_snap    = snap
            self._sim_time      = sim_time
            self._phase         = phase
            self._intruders_snap = intruders_snap
            self._birds_snap    = birds_snap
            self._formation     = formation
            self._locked        = locked
            self._leader_id     = leader_id

    def add_mission_report(self, report_text: str):
        """Add a mission report text to the queue. Thread-safe."""
        with self._report_lock:
            self._report_queue.append(report_text)

    # ──────────────────────────────────────────────────────────────────────
    #  Entry point
    # ──────────────────────────────────────────────────────────────────────

    def start_main(self):
        """Launch the monitor. Blocks in mainloop() until window is closed."""
        if not HAS_TK:
            print("  [Monitor] Tkinter unavailable — running in console mode")
            try:
                while True:
                    try:
                        self.sim.step()
                    except Exception:
                        break
                    time.sleep(1.0 / 240.0)
            except KeyboardInterrupt:
                pass
            return
        self.running = True
        self._build()   # calls mainloop() — blocks here

    def start(self):
        """Legacy non-blocking start (runs build in daemon thread)."""
        if not HAS_TK:
            return
        self.running = True
        threading.Thread(target=self._build, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────
    #  UI construction
    # ──────────────────────────────────────────────────────────────────────

    def _build(self):
        self.root = tk.Tk()
        self.root.title("Moniteur Drones — Gazebo Harmonic")
        self.root.geometry("970x800")
        self.root.configure(bg="#0d1117")

        # Open the remote control as a Toplevel (same Tkinter event loop)
        if hasattr(self.sim, "remote") and self.sim.remote is not None:
            rc_win = tk.Toplevel(self.root)
            self.sim.remote.build(rc_win)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#0d1117")
        style.configure("TNotebook.Tab", background="#161b22", foreground="white",
                        padding=[10, 5], font=("Consolas", 10))
        style.map("TNotebook.Tab", background=[("selected", "#1f6feb")])

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        f1 = tk.Frame(nb, bg="#0d1117"); nb.add(f1, text="  Cameras  ")
        self._build_cam(f1)

        f2 = tk.Frame(nb, bg="#0d1117"); nb.add(f2, text="  Trajectoires  ")
        self._build_traj(f2)

        f3 = tk.Frame(nb, bg="#0d1117"); nb.add(f3, text="  Vision RL  ")
        self._build_vision(f3)

        f4 = tk.Frame(nb, bg="#0d1117"); nb.add(f4, text="  Controle  ")
        self._build_params(f4)

        f5 = tk.Frame(nb, bg="#0d1117"); nb.add(f5, text="  Rapports  ")
        self._build_reports(f5)

        self._tick()
        self._tick_traj()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()

    # ──────────────────────────────────────────────────────────────────────
    #  Tab: Cameras
    # ──────────────────────────────────────────────────────────────────────

    def _build_cam(self, parent):
        """Camera tab: show per-drone status cards (no PyBullet camera, Gazebo provides 3D view)."""
        header = tk.Label(
            parent,
            text="Vue 3D : utilisez l'interface graphique Gazebo",
            fg="#ffaa00", bg="#0d1117",
            font=("Consolas", 11, "bold"),
        )
        header.pack(fill=tk.X, padx=5, pady=(8, 2))

        info = tk.Label(
            parent,
            text="Les cameras embarquees sont disponibles via les topics ROS2 /drone_i/camera\n"
                 "Ce panneau affiche l'etat en temps reel de chaque drone.",
            fg="#8b949e", bg="#0d1117",
            font=("Consolas", 9),
            justify=tk.LEFT,
        )
        info.pack(fill=tk.X, padx=5, pady=(0, 6))

        # Grid of drone info cards
        grid = tk.Frame(parent, bg="#0d1117")
        grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        COLS = 2
        self._cam_cards  = {}   # drone_id -> dict of labels
        self._cam_frames = {}

        for i in range(NUM_DRONES):
            col = i % COLS
            row = i // COLS
            color_hex = self._hex(DRONE_COLORS[i])

            card = tk.Frame(grid, bg="#161b22", relief=tk.RIDGE, bd=2)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            grid.columnconfigure(col, weight=1)
            grid.rowconfigure(row, weight=1)

            # Title
            tk.Label(card,
                     text=f"  DRONE {i}  [{DRONE_COLOR_NAMES[i]}]",
                     fg=color_hex, bg="#161b22",
                     font=("Consolas", 11, "bold")).pack(anchor="w", padx=4, pady=(6, 2))

            # Gazebo note
            tk.Label(card,
                     text="[Vue Gazebo GUI]",
                     fg="#444c56", bg="#161b22",
                     font=("Consolas", 8, "italic")).pack(anchor="w", padx=4)

            sep = tk.Frame(card, bg="#30363d", height=1)
            sep.pack(fill=tk.X, padx=4, pady=4)

            # Info rows
            cards = {}
            for key, label_text in [
                ("pos",     "Position :"),
                ("alt",     "Altitude  :"),
                ("battery", "Batterie  :"),
                ("state",   "Etat      :"),
                ("target",  "Cible     :"),
                ("dist",    "Dist. cible:"),
            ]:
                row_f = tk.Frame(card, bg="#161b22")
                row_f.pack(fill=tk.X, padx=4, pady=1)
                tk.Label(row_f, text=label_text, fg="#8b949e", bg="#161b22",
                         font=("Consolas", 9), width=13, anchor="w").pack(side=tk.LEFT)
                val_lbl = tk.Label(row_f, text="---", fg="white", bg="#161b22",
                                   font=("Consolas", 9), anchor="w")
                val_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                cards[key] = val_lbl

            # Battery bar canvas
            bat_canvas = tk.Canvas(card, height=12, bg="#161b22",
                                   highlightthickness=0)
            bat_canvas.pack(fill=tk.X, padx=4, pady=(2, 6))
            cards["bat_canvas"] = bat_canvas

            self._cam_cards[i]  = cards
            self._cam_frames[i] = card

    def _update_cam(self):
        """Refresh camera tab cards from latest drone snapshot."""
        with self._data_lock:
            snap  = list(self._drone_snap)
            phase = self._phase

        for d in snap:
            i = d["id"]
            if i not in self._cam_cards:
                continue
            cards = self._cam_cards[i]

            # Position
            cards["pos"].config(text=f"({d['x']:.1f}, {d['y']:.1f})")

            # Altitude
            cards["alt"].config(text=f"{d['z']:.2f} m")

            # Battery
            bat = d["battery"]
            bat_color = ("#ff4444" if bat < 20 else "#ffdd00" if bat < 50 else "#44ff88")
            cards["battery"].config(text=f"{bat:.1f} %", fg=bat_color)

            # State
            state_str = d["state"]
            s_color = "#ff4444" if "FAIL" in state_str or "GROUND" in state_str else \
                      "#44ff88" if "NAVIG" in state_str or "HOVER" in state_str else \
                      "#4488ff"
            cards["state"].config(text=state_str, fg=s_color)

            # Target
            cards["target"].config(text=f"({d['tx']:.1f}, {d['ty']:.1f})")

            # Distance to target
            dist = math.sqrt((d["x"] - d["tx"])**2 + (d["y"] - d["ty"])**2)
            cards["dist"].config(text=f"{dist:.1f} m")

            # Battery bar
            bc = cards["bat_canvas"]
            bc.delete("all")
            w = bc.winfo_width() or 200
            fill_w = int(w * bat / 100.0)
            bc.create_rectangle(0, 0, w, 12, fill="#1a1a2e", outline="")
            bc.create_rectangle(0, 0, fill_w, 12, fill=bat_color, outline="")
            bc.create_text(w // 2, 6, text=f"{bat:.0f}%",
                           fill="white", font=("Consolas", 7))

    # ──────────────────────────────────────────────────────────────────────
    #  Tab: Trajectoires
    # ──────────────────────────────────────────────────────────────────────

    def _build_traj(self, parent):
        if not HAS_MPL:
            tk.Label(parent,
                     text="Matplotlib non disponible\npip3 install matplotlib",
                     fg="#f85149", bg="#0d1117",
                     font=("Consolas", 12)).pack(pady=50)
            self.traj_canvas = None
            return
        self._traj_first = True
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.fig.patch.set_facecolor("#0d1117")
        self.ax.set_facecolor("#161b22")
        self.traj_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.traj_canvas, parent)
        toolbar.update()
        self.traj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_traj(self):
        if not HAS_MPL or not hasattr(self, "traj_canvas") or self.traj_canvas is None:
            return

        traj_colors = [
            "#ff4444", "#4488ff", "#44cc44", "#ffff44",
            "#ff44ff", "#44ffff", "#ff8844", "#8844ff",
        ]

        with self._data_lock:
            snap  = list(self._drone_snap)
            phase = self._phase

        for d in snap:
            i = d["id"]
            state = d["state"]
            if "FAIL" not in state and "GROUND" not in state and "IDLE" not in state:
                self.histories[i]["x"].append(d["x"])
                self.histories[i]["y"].append(d["y"])
                if len(self.histories[i]["x"]) > 500:
                    self.histories[i]["x"].pop(0)
                    self.histories[i]["y"].pop(0)

        if self._traj_first:
            saved_xlim = (PATROL_ZONE["x_min"] - 1, PATROL_ZONE["x_max"] + 1)
            saved_ylim = (PATROL_ZONE["y_min"] - 1, PATROL_ZONE["y_max"] + 1)
            self._traj_first = False
        else:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()

        self.ax.clear()
        self.ax.set_facecolor("#161b22")
        self.ax.set_aspect("equal")
        self.ax.set_xlim(saved_xlim)
        self.ax.set_ylim(saved_ylim)
        self.ax.grid(True, alpha=0.15, color="#555")
        self.ax.set_title(
            f"Trajectoires [MAPPO] — {phase}",
            color="white", fontsize=11,
        )
        self.ax.tick_params(colors="#888")

        # Draw obstacles
        for obs in OBSTACLES:
            self.ax.add_patch(plt.Rectangle(
                (obs["x"] - obs["sx"] / 2, obs["y"] - obs["sy"] / 2),
                obs["sx"], obs["sy"], color="#444", alpha=0.5,
            ))
            self.ax.text(obs["x"], obs["y"], obs["name"][:6],
                         color="#777", fontsize=5, ha="center", va="center")

        # Draw drone trajectories
        for d in snap:
            i  = d["id"]
            c  = traj_colors[i % len(traj_colors)]
            hx = self.histories[i]["x"]
            hy = self.histories[i]["y"]
            if not hx:
                continue
            self.ax.plot(hx, hy, color=c, alpha=0.5, linewidth=1)
            self.ax.plot(hx[-1], hy[-1], "o", color=c, markersize=7)
            self.ax.annotate(
                f"D{i}", (hx[-1] + 0.5, hy[-1] + 0.5),
                color=c, fontsize=7, fontweight="bold",
            )
            # Dashed line to current target
            state = d["state"]
            if "FAIL" not in state and "GROUND" not in state and \
               "LAND" not in state and "IDLE" not in state:
                self.ax.plot(
                    [hx[-1], d["tx"]], [hy[-1], d["ty"]],
                    "--", color=c, alpha=0.3, linewidth=0.8,
                )

        # Draw intruders — depuis snapshot thread-safe
        with self._data_lock:
            intruders_snap = list(self._intruders_snap)
            birds_snap     = list(self._birds_snap)

        for intr in intruders_snap:
            ix, iy = intr["pos"]
            self.ax.plot(ix, iy, "x", color="red", markersize=14, markeredgewidth=3)
            self.ax.annotate("INTRUS", (ix + 1, iy + 1),
                             color="red", fontsize=8, fontweight="bold")
            if intr.get("detected", False):
                circle = plt.Circle(
                    (ix, iy), ENCIRCLE_RADIUS,
                    fill=False, color="red", linestyle="--", alpha=0.5,
                )
                self.ax.add_patch(circle)

        # Draw birds — depuis snapshot thread-safe
        for bird in birds_snap:
            bx, by, bz = bird["pos"]
            self.ax.plot(bx, by, "^", color="#ffaa00", markersize=7, alpha=0.85)
            self.ax.annotate(f"{bz:.1f}m", (bx + 0.4, by + 0.4),
                             color="#ffaa00", fontsize=6)

        self.traj_canvas.draw()

    # ──────────────────────────────────────────────────────────────────────
    #  Tab: Vision RL
    # ──────────────────────────────────────────────────────────────────────

    def _build_vision(self, parent):
        tk.Label(parent, text="CE QUE CHAQUE DRONE VOIT",
                 fg="#58a6ff", bg="#0d1117",
                 font=("Consolas", 12, "bold")).pack(pady=5)

        self.vision_labels  = {}
        self.penalty_labels = {}
        for i in range(NUM_DRONES):
            outer = tk.Frame(parent, bg="#161b22", relief=tk.RIDGE, bd=1)
            outer.pack(fill=tk.X, padx=5, pady=1)

            row1 = tk.Frame(outer, bg="#161b22")
            row1.pack(fill=tk.X)
            c = self._hex(DRONE_COLORS[i])
            tk.Label(row1,
                     text=f" D{i} [{DRONE_COLOR_NAMES[i][:4]}]",
                     fg=c, bg="#161b22",
                     font=("Consolas", 9, "bold"),
                     width=14, anchor="w").pack(side=tk.LEFT, padx=3)
            lbl = tk.Label(row1, text="...", fg="white", bg="#161b22",
                           font=("Consolas", 8), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            self.vision_labels[i] = lbl

            pen_lbl = tk.Label(outer, text="", fg="#ff6644", bg="#161b22",
                               font=("Consolas", 8), anchor="w")
            pen_lbl.pack(fill=tk.X, padx=20, pady=(0, 2))
            self.penalty_labels[i] = pen_lbl

        self.vis_global = tk.Label(parent, text="", fg="#8b949e",
                                   bg="#0d1117", font=("Consolas", 9))
        self.vis_global.pack(fill=tk.X, padx=5, pady=5)

    def _update_vision(self):
        with self._data_lock:
            snap           = list(self._drone_snap)
            phase          = self._phase
            intruders_snap = list(self._intruders_snap)
            birds_snap     = list(self._birds_snap)
            lid            = self._leader_id

        rewards   = getattr(getattr(self.sim, "rl", None), "last_rewards", {})
        penalties = getattr(getattr(self.sim, "rl", None), "last_penalty_reasons", {})

        active = 0
        for d in snap:
            i = d["id"]
            state = d["state"]

            if "FAIL" in state or "GROUND" in state:
                self.vision_labels[i].config(text="HORS SERVICE", fg="#f85149")
                self.penalty_labels[i].config(text="")
                continue
            if "LAND" in state or "IDLE" in state:
                self.vision_labels[i].config(text="Au sol", fg="#484f58")
                self.penalty_labels[i].config(text="")
                continue

            active += 1
            dt   = math.sqrt((d["x"] - d["tx"])**2 + (d["y"] - d["ty"])**2)
            ldr  = "[L]" if d.get("is_leader") else "   "

            intr_str = ""
            for it in intruders_snap:
                ix, iy = it["pos"]
                di = math.sqrt((d["x"] - ix)**2 + (d["y"] - iy)**2)
                if di < SENSOR_RANGE:
                    intr_str = f" INTRUS{di:.0f}m"

            bird_str = ""
            for bird in birds_snap:
                bx, by, bz = bird["pos"]
                db = math.sqrt((d["x"]-bx)**2 + (d["y"]-by)**2 + (d["z"]-bz)**2)
                if db < BIRD_SAFETY_DISTANCE * 2:
                    bird_str = f" OIS{db:.0f}m"
                    break

            rew_s   = ""
            rew_col = "#58a6ff" if "NAVIG" in state else "#d29922"
            if i in rewards:
                r = rewards[i]
                rew_s   = f" R{r:+.1f}"
                rew_col = "#44ffaa" if r >= 0 else "#ff6644"

            self.vision_labels[i].config(
                text=(f"{ldr} ({d['x']:.1f},{d['y']:.1f},{d['z']:.1f}) "
                      f"{dt:.1f}m bat:{d['battery']:.0f}%{rew_s}{intr_str}{bird_str}"),
                fg=rew_col,
            )

            reasons = penalties.get(i, [])
            self.penalty_labels[i].config(
                text="  Penalites: " + "  |  ".join(reasons) if reasons else "",
            )

        ni = len(intruders_snap)
        nd = sum(1 for it in intruders_snap if it.get("detected", False))
        with self._data_lock:
            formation = self._formation
        self.vis_global.config(
            text=(f"Mode:MAPPO  Form:{formation}"
                  f"  Actifs:{active}/{NUM_DRONES}  Intrus:{nd}/{ni}  Phase:{phase}"),
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Tab: Contrôle
    # ──────────────────────────────────────────────────────────────────────

    def _build_params(self, parent):
        tk.Label(parent, text="CONTROLE", fg="#d29922", bg="#0d1117",
                 font=("Consolas", 12, "bold")).pack(pady=10)

        # Speed slider
        f = tk.Frame(parent, bg="#0d1117"); f.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(f, text="Vitesse drones:", fg="white", bg="#0d1117",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.spd = tk.DoubleVar(value=MAX_SPEED)
        tk.Scale(
            f, from_=0.5, to=8.0, resolution=0.5, orient=tk.HORIZONTAL,
            variable=self.spd, bg="#161b22", fg="white", length=250,
            command=lambda v: self._set_speed(float(v)),
        ).pack(side=tk.LEFT, padx=10)

        # Altitude slider
        f_alt = tk.Frame(parent, bg="#0d1117"); f_alt.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(f_alt, text="Altitude (m):", fg="#00ccff", bg="#0d1117",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.alt_var = tk.DoubleVar(value=5.0)
        tk.Scale(
            f_alt, from_=2.0, to=25.0, resolution=1.0, orient=tk.HORIZONTAL,
            variable=self.alt_var, bg="#161b22", fg="#00ccff", length=250,
            command=lambda v: self._set_altitude(float(v)),
        ).pack(side=tk.LEFT, padx=10)

        # Target entry
        f2 = tk.Frame(parent, bg="#0d1117"); f2.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(f2, text="Target X:", fg="white", bg="#0d1117",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.tx_entry = tk.Entry(f2, width=5, bg="#161b22", fg="white",
                                  font=("Consolas", 10))
        self.tx_entry.insert(0, "10"); self.tx_entry.pack(side=tk.LEFT, padx=3)
        tk.Label(f2, text="Y:", fg="white", bg="#0d1117",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.ty_entry = tk.Entry(f2, width=5, bg="#161b22", fg="white",
                                  font=("Consolas", 10))
        self.ty_entry.insert(0, "10"); self.ty_entry.pack(side=tk.LEFT, padx=3)
        tk.Button(
            f2, text="GO", bg="#e67e22", fg="white",
            font=("Consolas", 9, "bold"),
            command=lambda: self.sim.cmd_target(
                float(self.tx_entry.get()), float(self.ty_entry.get())),
        ).pack(side=tk.LEFT, padx=5)

        # Route entry
        f3 = tk.Frame(parent, bg="#0d1117"); f3.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(f3, text="Route:", fg="white", bg="#0d1117",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.rt_entry = tk.Entry(f3, width=35, bg="#161b22", fg="white",
                                  font=("Consolas", 10))
        self.rt_entry.insert(0, "10,10 -10,10 -10,-10 10,-10")
        self.rt_entry.pack(side=tk.LEFT, padx=3)
        tk.Button(f3, text="MAP", bg="#2980b9", fg="white",
                  font=("Consolas", 9, "bold"),
                  command=self._do_route).pack(side=tk.LEFT, padx=5)

        # Swarm commands row
        f4 = tk.Frame(parent, bg="#0d1117"); f4.pack(fill=tk.X, padx=20, pady=8)
        btn_cfg = [
            ("TAKEOFF",  "#27ae60", lambda: self.sim.cmd_takeoff()),
            ("LAND",     "#e74c3c", lambda: self.sim.cmd_land()),
            ("STOP",     "#f39c12", lambda: self.sim.cmd_stop()),
            ("PATROL",   "#3498db", lambda: self.sim.cmd_patrol()),
        ]
        for label, color, cmd in btn_cfg:
            tk.Button(f4, text=label, command=cmd,
                      bg=color, fg="white", font=("Consolas", 9, "bold"),
                      relief=tk.FLAT, padx=8, pady=4,
                      ).pack(side=tk.LEFT, padx=4)

        # RL status label
        self.rl_lbl = tk.Label(parent, text="", fg="#8b949e", bg="#0d1117",
                                font=("Consolas", 9))
        self.rl_lbl.pack(padx=20, pady=5)

    def _set_speed(self, val: float):
        try:
            import config
            config.MAX_SPEED = val
        except Exception:
            pass

    def _set_altitude(self, val: float):
        try:
            self.sim.cmd_altitude(val)
        except Exception:
            pass

    def _do_route(self):
        try:
            raw = self.rt_entry.get().strip().split()
            wps = []
            for pair in raw:
                parts = pair.split(",")
                if len(parts) == 2:
                    wps.append((float(parts[0]), float(parts[1])))
            if wps:
                self.sim.cmd_route(wps)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    #  Tab: Rapports
    # ──────────────────────────────────────────────────────────────────────

    def _build_reports(self, parent):
        tk.Label(parent,
                 text="RAPPORTS DE MISSION — LEADER DRONE",
                 fg="#d29922", bg="#0d1117",
                 font=("Consolas", 11, "bold")).pack(pady=(10, 4))

        tk.Label(parent,
                 text="Chaque rapport est genere automatiquement a l'arrivee a la cible.",
                 fg="#8b949e", bg="#0d1117",
                 font=("Consolas", 9)).pack(pady=(0, 8))

        frame = tk.Frame(parent, bg="#0d1117")
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._report_text = tk.Text(
            frame,
            bg="#0d1117", fg="#e6edf3",
            font=("Consolas", 9),
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
        )
        self._report_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._report_text.yview)

        # Color tags
        self._report_text.tag_configure("header",    foreground="#ffd700", font=("Consolas", 9, "bold"))
        self._report_text.tag_configure("label",     foreground="#8b949e")
        self._report_text.tag_configure("value",     foreground="#3fb950")
        self._report_text.tag_configure("formation", foreground="#ff7b72", font=("Consolas", 9, "bold"))
        self._report_text.tag_configure("sep",       foreground="#30363d")

        tk.Button(parent, text="Effacer les rapports",
                  bg="#21262d", fg="#8b949e",
                  font=("Consolas", 9),
                  command=self._clear_reports,
                  ).pack(pady=(0, 8))

        self._append_report_line("En attente du premier rapport de mission...\n", "label")

    def _clear_reports(self):
        if self._report_text is None:
            return
        self._report_text.config(state=tk.NORMAL)
        self._report_text.delete("1.0", tk.END)
        self._report_text.config(state=tk.DISABLED)

    def _append_report_line(self, text: str, tag: str = "value"):
        if self._report_text is None:
            return
        self._report_text.config(state=tk.NORMAL)
        self._report_text.insert(tk.END, text, tag)
        self._report_text.see(tk.END)
        self._report_text.config(state=tk.DISABLED)

    def _flush_report_queue(self):
        if not self._report_queue:
            return
        with self._report_lock:
            pending = list(self._report_queue)
            self._report_queue.clear()
        for report in pending:
            self._render_report(report)

    def _render_report(self, report_text: str):
        if self._report_text is None:
            return
        self._report_text.config(state=tk.NORMAL)
        for line in report_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("=") or stripped.startswith("-"):
                self._report_text.insert(tk.END, line + "\n", "sep")
            elif "RAPPORT MISSION" in stripped:
                self._report_text.insert(tk.END, line + "\n", "header")
            elif "Formation prise" in line:
                idx    = line.find(":") + 1
                prefix = line[:idx]
                value  = line[idx:]
                self._report_text.insert(tk.END, prefix, "label")
                self._report_text.insert(tk.END, value + "\n", "formation")
            elif ":" in line and not stripped.startswith("=") and not stripped.startswith("-"):
                idx = line.find(":") + 1
                self._report_text.insert(tk.END, line[:idx], "label")
                self._report_text.insert(tk.END, line[idx:] + "\n", "value")
            else:
                self._report_text.insert(tk.END, line + "\n", "label")
        self._report_text.see(tk.END)
        self._report_text.config(state=tk.DISABLED)

    # ──────────────────────────────────────────────────────────────────────
    #  Simulation loop (Gazebo version)
    # ──────────────────────────────────────────────────────────────────────

    def _schedule_sim(self):
        """Keep-alive only — la simulation tourne dans _sim_loop (thread séparé).
        Ne pas appeler step() ici : cela créerait une double boucle avec races."""
        pass

    # ──────────────────────────────────────────────────────────────────────
    #  Main UI tick
    # ──────────────────────────────────────────────────────────────────────

    def _tick(self):
        """Mise à jour rapide : états drones, vision RL, rapports (200ms)."""
        if not self.running:
            return
        try:
            self._update_cam()
            self._update_vision()
            self._flush_report_queue()

            # Update RL status label (control tab)
            try:
                rl_trained = self.sim.rl.is_trained()
                import config as _cfg
                spd = _cfg.MAX_SPEED
                self.rl_lbl.config(
                    text=(f"MAPPO: {'RESEAU OK' if rl_trained else 'MODE DEMO'}  |  "
                          f"Speed: {spd:.1f} m/s  |  Sim: Gazebo Harmonic"),
                    fg="#3fb950" if rl_trained else "#d29922",
                )
            except Exception:
                pass

        except tk.TclError:
            self.running = False
            return
        except Exception:
            pass

        if self.running:
            try:
                self.root.after(50, self._tick)   # 20 Hz — temps réel
            except tk.TclError:
                self.running = False

    def _tick_traj(self):
        """Mise à jour trajectoire matplotlib (400ms = 2.5 Hz)."""
        if not self.running:
            return
        try:
            self._update_traj()
        except tk.TclError:
            self.running = False
            return
        except Exception:
            pass
        if self.running:
            try:
                self.root.after(400, self._tick_traj)
            except tk.TclError:
                self.running = False

    # ──────────────────────────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────────────────────────

    def _hex(self, rgba) -> str:
        return (f"#{int(rgba[0] * 255):02x}"
                f"{int(rgba[1] * 255):02x}"
                f"{int(rgba[2] * 255):02x}")

    def _close(self):
        self.running = False
        try:
            self.root.destroy()
        except Exception:
            pass
