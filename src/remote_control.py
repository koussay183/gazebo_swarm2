"""
PFE - Swarm Drones MARL — Gazebo Harmonic edition
Telecommande Graphique — s'ouvre comme Toplevel dans la boucle Tkinter du moniteur.
PyBullet references removed; camera follow prints a message instead.
"""
import tkinter as tk
from tkinter import ttk
import math

from config import NUM_DRONES


class RemoteControl:
    """Interface telecommande graphique pour controler le swarm."""

    def __init__(self, sim):
        self.sim     = sim
        self.root    = None
        self.running = False

    def start(self):
        """Legacy no-op — la fenetre est maintenant ouverte par DroneMonitor."""
        pass

    def build(self, window):
        """Peuple la fenetre Toplevel avec tous les widgets de telecommande.
        Appele depuis le thread principal via DroneMonitor._build()."""
        self.root = window
        self.running = True
        self.root.title("Telecommande Drone Swarm — Gazebo")
        self.root.geometry("540x1100")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")

        # ══════ HEADER ══════
        header = tk.Frame(self.root, bg="#16213e", height=50)
        header.pack(fill=tk.X, padx=5, pady=5)
        header.pack_propagate(False)
        tk.Label(
            header, text="TELECOMMANDE DRONE SWARM [Gazebo]",
            font=("Arial", 13, "bold"), fg="#e94560", bg="#16213e",
        ).pack(expand=True)

        # ══════ STATUS BAR ══════
        sf = tk.Frame(self.root, bg="#0f3460")
        sf.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_phase = tk.Label(sf, text="Phase: INIT", font=("Consolas", 10),
                                   fg="#00ff88", bg="#0f3460", anchor="w")
        self.lbl_phase.pack(side=tk.LEFT, padx=10)
        self.lbl_formation = tk.Label(sf, text="Formation: v_shape",
                                       font=("Consolas", 10), fg="#00ccff",
                                       bg="#0f3460", anchor="e")
        self.lbl_formation.pack(side=tk.RIGHT, padx=10)

        sf2 = tk.Frame(self.root, bg="#0f3460")
        sf2.pack(fill=tk.X, padx=5, pady=2)
        self.lbl_leader = tk.Label(sf2, text="Leader: D0", font=("Consolas", 10),
                                    fg="#ffd700", bg="#0f3460", anchor="w")
        self.lbl_leader.pack(side=tk.LEFT, padx=10)
        self.lbl_speed = tk.Label(sf2, text="Speed: 4.0 m/s", font=("Consolas", 10),
                                   fg="#ff6b6b", bg="#0f3460", anchor="e")
        self.lbl_speed.pack(side=tk.RIGHT, padx=10)

        # ══════ COMMANDES SWARM ══════
        self._section("COMMANDES SWARM")
        bf1 = tk.Frame(self.root, bg="#1a1a2e")
        bf1.pack(fill=tk.X, padx=10, pady=2)
        self._btn(bf1, "TAKEOFF", "#27ae60", lambda: self.sim.cmd_takeoff())
        self._btn(bf1, "LAND",    "#e74c3c", lambda: self.sim.cmd_land())
        self._btn(bf1, "STOP",    "#f39c12", lambda: self.sim.cmd_stop())
        self._btn(bf1, "PATROL",  "#3498db", lambda: self.sim.cmd_patrol())

        # ══════ FORMATIONS ══════
        self._section("FORMATIONS")
        ff1 = tk.Frame(self.root, bg="#1a1a2e")
        ff1.pack(fill=tk.X, padx=10, pady=2)
        for f in ["v_shape", "circle", "diamond", "triangle"]:
            self._btn(ff1, f, "#8e44ad",
                      lambda fn=f: self._formation(fn), w=9)

        ff2 = tk.Frame(self.root, bg="#1a1a2e")
        ff2.pack(fill=tk.X, padx=10, pady=2)
        for f in ["line", "grid", "column", "wedge"]:
            self._btn(ff2, f, "#8e44ad",
                      lambda fn=f: self._formation(fn), w=9)

        ff3 = tk.Frame(self.root, bg="#1a1a2e")
        ff3.pack(fill=tk.X, padx=10, pady=2)
        for f in ["echelon_left", "echelon_right"]:
            self._btn(ff3, f, "#8e44ad",
                      lambda fn=f: self._formation(fn), w=15)

        ff4 = tk.Frame(self.root, bg="#1a1a2e")
        ff4.pack(fill=tk.X, padx=10, pady=2)
        self._btn(ff4, "AUTO (cerveau)", "#2c3e50",
                  lambda: self._formation_auto(), w=20)
        self.lbl_form_lock = tk.Label(ff4, text="AUTO", font=("Consolas", 9),
                                      fg="#888888", bg="#1a1a2e")
        self.lbl_form_lock.pack(side=tk.LEFT, padx=6)

        # ══════ VITESSE & ALTITUDE ══════
        self._section("VITESSE & ALTITUDE DU SWARM")
        vf = tk.Frame(self.root, bg="#1a1a2e")
        vf.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(vf, text="Vit(m/s):", fg="white", bg="#1a1a2e",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=2.5)
        tk.Scale(vf, from_=0.5, to=10.0, resolution=0.5,
                 orient=tk.HORIZONTAL, variable=self.speed_var,
                 command=self._on_speed_change,
                 bg="#1a1a2e", fg="white", troughcolor="#0f3460",
                 highlightthickness=0, length=280,
                 font=("Consolas", 9),
                 ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        af = tk.Frame(self.root, bg="#1a1a2e")
        af.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(af, text="Alt(m):", fg="#00ccff", bg="#1a1a2e",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self.alt_var = tk.DoubleVar(value=5.0)
        tk.Scale(af, from_=2.0, to=25.0, resolution=1.0,
                 orient=tk.HORIZONTAL, variable=self.alt_var,
                 command=self._on_alt_change,
                 bg="#1a1a2e", fg="#00ccff", troughcolor="#0f3460",
                 highlightthickness=0, length=280,
                 font=("Consolas", 9),
                 ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.lbl_swarm_params = tk.Label(
            self.root,
            text="Vit: 2.5 m/s  |  Alt: 5.0 m",
            font=("Consolas", 9), fg="#aaaaaa", bg="#1a1a2e",
        )
        self.lbl_swarm_params.pack(fill=tk.X, padx=10, pady=1)

        # ══════ INTRUS ══════
        self._section("INTRUS MOBILE")
        inf = tk.Frame(self.root, bg="#1a1a2e")
        inf.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(inf, text="X:", fg="white", bg="#1a1a2e",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.intruder_x = tk.Entry(inf, width=5, font=("Consolas", 10),
                                    bg="#0f3460", fg="white", insertbackground="white")
        self.intruder_x.insert(0, "10")
        self.intruder_x.pack(side=tk.LEFT, padx=3)
        tk.Label(inf, text="Y:", fg="white", bg="#1a1a2e",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.intruder_y = tk.Entry(inf, width=5, font=("Consolas", 10),
                                    bg="#0f3460", fg="white", insertbackground="white")
        self.intruder_y.insert(0, "5")
        self.intruder_y.pack(side=tk.LEFT, padx=3)
        self._btn(inf, "LANCER", "#c0392b", self._launch_intruder, w=10)
        self._btn(inf, "CLEAR",  "#27ae60",
                  lambda: self.sim.cmd_clear_intruder(), w=8)

        inf2 = tk.Frame(self.root, bg="#1a1a2e")
        inf2.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(inf2, text="Mouvement:", fg="#ff6b6b", bg="#1a1a2e",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self.intruder_formation = tk.StringVar(value="random")
        for fm in ["random", "zigzag", "circle", "evasive"]:
            tk.Radiobutton(inf2, text=fm, variable=self.intruder_formation,
                           value=fm, bg="#1a1a2e", fg="white",
                           selectcolor="#0f3460",
                           activebackground="#1a1a2e", activeforeground="white",
                           font=("Consolas", 8)).pack(side=tk.LEFT, padx=2)

        inf3 = tk.Frame(self.root, bg="#1a1a2e")
        inf3.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(inf3, text="Vitesse intrus:", fg="#ff6b6b", bg="#1a1a2e",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self.ispeed_var = tk.DoubleVar(value=1.5)
        tk.Scale(inf3, from_=0.5, to=5.0, resolution=0.5, orient=tk.HORIZONTAL,
                 variable=self.ispeed_var, command=self._on_ispeed_change,
                 bg="#1a1a2e", fg="white", troughcolor="#0f3460",
                 highlightthickness=0, length=250,
                 font=("Consolas", 9)).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # ══════ PANNES ══════
        self._section("PANNES / KILL")
        pf = tk.Frame(self.root, bg="#1a1a2e")
        pf.pack(fill=tk.X, padx=10, pady=2)
        self._btn(pf, "KILL LEADER", "#e74c3c",
                  lambda: self.sim.cmd_kill(self.sim.election.leader_id), w=14)
        tk.Label(pf, text="Kill D:", fg="white", bg="#1a1a2e",
                 font=("Consolas", 10)).pack(side=tk.LEFT, padx=3)
        self.kill_id = tk.Entry(pf, width=3, font=("Consolas", 10),
                                 bg="#0f3460", fg="white", insertbackground="white")
        self.kill_id.insert(0, "0")
        self.kill_id.pack(side=tk.LEFT)
        self._btn(pf, "KILL", "#e74c3c", lambda: self._kill_drone(), w=6)
        self._btn(pf, "FIX",  "#27ae60", lambda: self._fix_drone(),  w=6)

        pf2 = tk.Frame(self.root, bg="#1a1a2e")
        pf2.pack(fill=tk.X, padx=10, pady=2)
        self._btn(pf2, "FIX + TAKEOFF", "#16a085", self._fix_and_takeoff, w=15)

        # ══════ PATROUILLE ══════
        self._section("PATROUILLE — ALTITUDE & RAYON")
        paf = tk.Frame(self.root, bg="#1a1a2e")
        paf.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(paf, text="Alt (m):", fg="white", bg="#1a1a2e",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self.palt_var = tk.DoubleVar(value=8.0)
        tk.Scale(paf, from_=2.0, to=25.0, resolution=1.0, orient=tk.HORIZONTAL,
                 variable=self.palt_var, command=self._on_palt_change,
                 bg="#1a1a2e", fg="white", troughcolor="#0f3460",
                 highlightthickness=0, length=260, font=("Consolas", 8),
                 ).pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)

        prf = tk.Frame(self.root, bg="#1a1a2e")
        prf.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(prf, text="Rayon(m):", fg="white", bg="#1a1a2e",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self.prad_var = tk.DoubleVar(value=20.0)
        tk.Scale(prf, from_=3.0, to=23.0, resolution=1.0, orient=tk.HORIZONTAL,
                 variable=self.prad_var, command=self._on_prad_change,
                 bg="#1a1a2e", fg="white", troughcolor="#0f3460",
                 highlightthickness=0, length=260, font=("Consolas", 8),
                 ).pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)

        # ══════ TRAJECTOIRE ══════
        self._section("TRAJECTOIRE DU SWARM")

        tgt_f = tk.Frame(self.root, bg="#1a1a2e")
        tgt_f.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(tgt_f, text="X:", fg="white", bg="#1a1a2e",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.tgt_x = tk.Entry(tgt_f, width=5, font=("Consolas", 10),
                               bg="#0f3460", fg="white", insertbackground="white")
        self.tgt_x.insert(0, "10")
        self.tgt_x.pack(side=tk.LEFT, padx=3)
        tk.Label(tgt_f, text="Y:", fg="white", bg="#1a1a2e",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.tgt_y = tk.Entry(tgt_f, width=5, font=("Consolas", 10),
                               bg="#0f3460", fg="white", insertbackground="white")
        self.tgt_y.insert(0, "10")
        self.tgt_y.pack(side=tk.LEFT, padx=3)
        self._btn(tgt_f, "GO", "#e67e22", self._do_target, w=6)

        rt_f = tk.Frame(self.root, bg="#1a1a2e")
        rt_f.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(rt_f, text="Route:", fg="#00ccff", bg="#1a1a2e",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        self.route_entry = tk.Entry(rt_f, width=28, font=("Consolas", 9),
                                     bg="#0f3460", fg="white", insertbackground="white")
        self.route_entry.insert(0, "10,10 -10,10 -10,-10 10,-10")
        self.route_entry.pack(side=tk.LEFT, padx=3)
        self._btn(rt_f, "GO", "#2980b9", self._do_route, w=4)

        # ══════ MAPPO ══════
        self._section("MAPPO")
        rlf = tk.Frame(self.root, bg="#1a1a2e")
        rlf.pack(fill=tk.X, padx=10, pady=2)
        self._btn(rlf, "CAM FOLLOW (Gazebo)", "#2980b9",
                  self._toggle_camfollow, w=22)

        self.lbl_rl_mode = tk.Label(self.root, text="Mode: MAPPO | Sim: Gazebo Harmonic",
                                     font=("Consolas", 9, "bold"),
                                     fg="#1abc9c", bg="#1a1a2e")
        self.lbl_rl_mode.pack(fill=tk.X, padx=10, pady=1)
        self.lbl_rl_info = tk.Label(self.root, text="",
                                     font=("Consolas", 8), fg="#888888", bg="#1a1a2e")
        self.lbl_rl_info.pack(fill=tk.X, padx=10, pady=1)
        self.lbl_rl_reward = tk.Label(self.root, text="",
                                       font=("Consolas", 8), fg="#00ccff", bg="#1a1a2e")
        self.lbl_rl_reward.pack(fill=tk.X, padx=10, pady=1)

        # ══════ ETAT DES DRONES ══════
        self._section("ETAT DES DRONES")
        self.drone_labels = []
        dsf = tk.Frame(self.root, bg="#16213e")
        dsf.pack(fill=tk.X, padx=5, pady=2)
        drone_colors = ["#ff4444", "#4488ff", "#44cc44", "#ffff44",
                        "#ff44ff", "#44ffff", "#ff8844", "#8844ff"]
        for i in range(NUM_DRONES):
            row = tk.Frame(dsf, bg="#16213e")
            row.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(row, text=f"D{i}", font=("Consolas", 9, "bold"),
                     fg=drone_colors[i], bg="#16213e", width=3).pack(side=tk.LEFT)
            lbl = tk.Label(row,
                           text="IDLE  | |||||||||||| 100% | (0.0, 0.0, 0.3)",
                           font=("Consolas", 8), fg="#aaaaaa", bg="#16213e",
                           anchor="w")
            lbl.pack(side=tk.LEFT, expand=True, fill=tk.X)
            self.drone_labels.append(lbl)

        # Statut intrus
        self.lbl_intruder = tk.Label(self.root, text="",
                                      font=("Consolas", 10, "bold"),
                                      fg="#ff4444", bg="#1a1a2e")
        self.lbl_intruder.pack(fill=tk.X, padx=10, pady=3)

        # Start display update loop
        self._update_display()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────── ACTIONS ───────────

    def _formation(self, name: str):
        """Apply a formation and lock it (brain cannot override)."""
        self.sim.cmd_formation(name)
        if hasattr(self, "lbl_form_lock"):
            self.lbl_form_lock.config(text=f"[LOCK] {name}", fg="#e94560")

    def _formation_auto(self):
        """Return formation control to the autonomous brain."""
        self.sim.formation_locked = False
        try:
            self.sim.brain._last_formation = None
        except Exception:
            pass
        if hasattr(self, "lbl_form_lock"):
            self.lbl_form_lock.config(text="AUTO", fg="#888888")
        print("  [RC] Formation : retour mode AUTO (cerveau)")

    def _launch_intruder(self):
        try:
            x  = float(self.intruder_x.get())
            y  = float(self.intruder_y.get())
            fm = self.intruder_formation.get()
            self.sim.cmd_intruder(x, y, fm)
        except ValueError:
            pass

    def _kill_drone(self):
        try:
            self.sim.cmd_kill(int(self.kill_id.get()))
        except (ValueError, IndexError):
            pass

    def _fix_drone(self):
        try:
            self.sim.cmd_unmark(int(self.kill_id.get()))
        except (ValueError, IndexError):
            pass

    def _fix_and_takeoff(self):
        try:
            did = int(self.kill_id.get())
            self.sim.cmd_unmark(did)
            self.root.after(500, lambda: self.sim.drones[did].takeoff())
        except (ValueError, IndexError):
            pass

    def _on_speed_change(self, val):
        import config
        config.MAX_SPEED = float(val)
        if hasattr(self, "lbl_swarm_params"):
            self.lbl_swarm_params.config(
                text=f"Vit: {config.MAX_SPEED:.1f} m/s  |  Alt: {config.CRUISE_ALTITUDE:.1f} m")

    def _on_alt_change(self, val):
        self.sim.cmd_altitude(float(val))
        import config
        if hasattr(self, "lbl_swarm_params"):
            self.lbl_swarm_params.config(
                text=f"Vit: {config.MAX_SPEED:.1f} m/s  |  Alt: {config.CRUISE_ALTITUDE:.1f} m")

    def _on_ispeed_change(self, val):
        self.sim.intruder_speed = float(val)

    def _on_palt_change(self, val):
        self.sim.patrol_altitude = float(val)

    def _on_prad_change(self, val):
        rad = float(val)
        self.sim.patrol_radius = rad
        try:
            self.sim.patrol_wps = self.sim._gen_patrol_wps()
            self.sim.patrol_idx = 0
        except Exception:
            pass

    def _do_target(self):
        try:
            x = float(self.tgt_x.get())
            y = float(self.tgt_y.get())
            self.sim.cmd_target(x, y)
        except ValueError:
            pass

    def _do_route(self):
        try:
            raw = self.route_entry.get().strip().split()
            wps = []
            for pair in raw:
                parts = pair.split(",")
                if len(parts) == 2:
                    wps.append((float(parts[0]), float(parts[1])))
            if wps:
                self.sim.cmd_route(wps)
        except Exception:
            pass

    def _toggle_camfollow(self):
        """In Gazebo mode, camera follow is handled by the Gazebo GUI, not Python."""
        follow = not getattr(self.sim, "camera_follow", False)
        self.sim.camera_follow = follow
        print(f"  [RC] Camera follow: {'ON' if follow else 'OFF'} "
              f"(use Gazebo GUI Entity Tree to track a drone)")

    # ─────────── MISE A JOUR AFFICHAGE ───────────

    def _update_display(self):
        if not self.running:
            return
        try:
            import config
            from drone import DroneState

            self.lbl_phase.config(text=f"Phase: {self.sim.phase}")
            fm     = self.sim.formation_mgr.formation
            locked = getattr(self.sim, "formation_locked", False)
            self.lbl_formation.config(
                text=f"Form: {fm} {'[L]' if locked else '[A]'}")
            if hasattr(self, "lbl_form_lock"):
                if locked:
                    self.lbl_form_lock.config(text=f"[LOCK] {fm}", fg="#e94560")
                else:
                    self.lbl_form_lock.config(text="AUTO", fg="#888888")
            self.lbl_leader.config(
                text=f"Leader: D{self.sim.election.leader_id}")
            self.lbl_speed.config(
                text=f"Vit: {config.MAX_SPEED:.1f} m/s | Alt: {config.CRUISE_ALTITUDE:.1f} m")

            drone_colors = ["#ff4444", "#4488ff", "#44cc44", "#ffff44",
                            "#ff44ff", "#44ffff", "#ff8844", "#8844ff"]
            for i, d in enumerate(self.sim.drones):
                bat_len = int(d.battery / 10)
                bar     = "|" * bat_len + "." * (10 - bat_len)
                state   = str(d.state)[:7]
                ldr     = "[L]" if d.id == self.sim.election.leader_id else "   "

                if d.state in [DroneState.FAILED, DroneState.GROUNDED]:
                    color = "#ff4444"
                elif getattr(d, "is_marked", False):
                    color = "#ffaa00"
                elif d.state in [DroneState.HOVERING, DroneState.NAVIGATING]:
                    color = "#44ff44"
                elif d.state in [DroneState.TAKEOFF, DroneState.LANDING]:
                    color = "#4488ff"
                else:
                    color = "#666666"

                txt = (f"{ldr}{state:<7} |{bar}| {d.battery:4.0f}% "
                       f"({d.x:5.1f},{d.y:5.1f},{d.z:4.1f})")
                self.drone_labels[i].config(text=txt, fg=color)

            if self.sim.intruders:
                intr = self.sim.intruders[0]
                ix, iy = intr["pos"]
                det  = "DETECTE !" if intr["detected"] else "en mouvement..."
                n_i  = len(self.sim.intruders)
                pfx  = f"{n_i} INTRUS" if n_i > 1 else "INTRUS"
                self.lbl_intruder.config(
                    text=f"[!] {pfx} ({ix:.1f},{iy:.1f}) [{intr['formation']}] — {det}")
            else:
                self.lbl_intruder.config(text="")

            info    = self.sim.rl.get_last_info()
            trained = "RESEAU MAPPO" if info["trained"] else "MODE DEMO"
            self.lbl_rl_mode.config(
                text=f"MAPPO [{trained}]  drones={info['drones']}  | Gazebo Harmonic")
            self.lbl_rl_info.config(
                text=f"Formation: {self.sim.formation_mgr.formation}", fg="#1abc9c")
            self.lbl_rl_reward.config(
                text=f"Recompenses: {self.sim.rl.get_reward_summary()}",
                fg="#00ccff")

        except Exception:
            pass

        if self.running:
            self.root.after(50, self._update_display)

    # ─────────── HELPERS ───────────

    def _section(self, title: str):
        sf = tk.Frame(self.root, bg="#1a1a2e")
        sf.pack(fill=tk.X, padx=10, pady=(8, 2))
        tk.Label(sf, text=f"--- {title} ---",
                 font=("Arial", 9, "bold"),
                 fg="#555577", bg="#1a1a2e").pack(anchor="w")

    def _btn(self, parent, text: str, color: str, command, w=None):
        btn = tk.Button(
            parent, text=text, command=command,
            bg=color, fg="white", activebackground=color,
            font=("Arial", 9, "bold"), relief=tk.FLAT,
            cursor="hand2", padx=5, pady=3,
        )
        if w:
            btn.config(width=w)
        btn.pack(side=tk.LEFT, padx=2, pady=2)
        btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=self._lighten(c)))
        btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))
        return btn

    def _lighten(self, hex_color: str) -> str:
        r = min(255, int(hex_color[1:3], 16) + 30)
        g = min(255, int(hex_color[3:5], 16) + 30)
        b = min(255, int(hex_color[5:7], 16) + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_close(self):
        self.running = False
        if self.root:
            self.root.destroy()
