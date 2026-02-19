import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.spatial.transform import Rotation
from time import perf_counter

from drone_dynamics import SimulationResult


class MatplotlibQuadViewer:
    def __init__(self, result: SimulationResult, body_axis_len: float = 0.25):
        self.result = result
        self.steps = len(result.time)
        self.body_axis_len = body_axis_len

        self.current_frame = 0
        self.playhead = 0.0
        self.is_playing = False
        self.playback_speed = 1.0
        self.last_tick_wall = None

        self.fig = None
        self.time_slider = None
        self.speed_slider = None
        self.play_button = None
        self.timer = None

    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.20, hspace=0.35)

        self.ax_world = self.fig.add_subplot(221, projection="3d")
        self.ax_world.set_title("Quadcopter 3D Simulation")
        self.ax_world.set_xlabel("X [m]")
        self.ax_world.set_ylabel("Y [m]")
        self.ax_world.set_zlabel("Z [m]")
        self.ax_world.zaxis.label.set_color("tab:blue")
        self.ax_world.tick_params(axis="z", colors="tab:blue")
        world_points = [self.result.pos]
        if self.result.desired_path is not None and len(self.result.desired_path) > 0:
            world_points.append(self.result.desired_path)
        world_cloud = np.vstack(world_points)
        mins = np.min(world_cloud, axis=0)
        maxs = np.max(world_cloud, axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(1.5, 0.6 * np.max(maxs - mins))
        self.ax_world.set_xlim(center[0] - radius, center[0] + radius)
        self.ax_world.set_ylim(center[1] - radius, center[1] + radius)
        self.ax_world.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
        triad_len = 0.20 * radius
        triad_origin = np.array(
            [
                center[0] - 0.80 * radius,
                center[1] - 0.80 * radius,
                max(0.0, center[2] - 0.80 * radius),
            ]
        )
        self.ax_world.quiver(*triad_origin, triad_len, 0.0, 0.0, color="r", arrow_length_ratio=0.15, lw=1.5)
        self.ax_world.quiver(*triad_origin, 0.0, triad_len, 0.0, color="g", arrow_length_ratio=0.15, lw=1.5)
        self.ax_world.quiver(*triad_origin, 0.0, 0.0, triad_len, color="b", arrow_length_ratio=0.15, lw=1.5)
        self.ax_world.text(*(triad_origin + np.array([triad_len * 1.1, 0.0, 0.0])), "+X", color="r", fontsize=8)
        self.ax_world.text(*(triad_origin + np.array([0.0, triad_len * 1.1, 0.0])), "+Y", color="g", fontsize=8)
        self.ax_world.text(*(triad_origin + np.array([0.0, 0.0, triad_len * 1.1])), "+Z", color="b", fontsize=8)

        self.path_line, = self.ax_world.plot([], [], [], "k-", lw=1.6, alpha=0.8, label="actual")
        self.desired_path_line = None
        self.waypoint_points = None
        self.active_waypoint_point = None
        if self.result.desired_path is not None and len(self.result.desired_path) > 0:
            self.desired_path_line, = self.ax_world.plot(
                self.result.desired_path[:, 0],
                self.result.desired_path[:, 1],
                self.result.desired_path[:, 2],
                color="tab:purple",
                lw=1.4,
                ls="--",
                alpha=0.9,
                label="desired",
            )
            self.waypoint_points = self.ax_world.scatter(
                self.result.desired_path[:, 0],
                self.result.desired_path[:, 1],
                self.result.desired_path[:, 2],
                c="tab:purple",
                s=24,
                alpha=0.8,
            )
            self.active_waypoint_point = self.ax_world.scatter([], [], [], c="tab:red", s=42, marker="x")
        self.body_x, = self.ax_world.plot([], [], [], "r-", lw=2, label="body x")
        self.body_y, = self.ax_world.plot([], [], [], "g-", lw=2, label="body y")
        self.body_z, = self.ax_world.plot([], [], [], "b-", lw=2, label="body z")
        self.point = self.ax_world.scatter([], [], [], c="k", s=20)
        self.ax_world.legend(loc="upper left")

        self.ax_model = self.fig.add_subplot(222, projection="3d")
        self.ax_model.set_title("Drone Attitude Model")
        self.ax_model.set_xlabel("x")
        self.ax_model.set_ylabel("y")
        self.ax_model.set_zlabel("z")
        self.ax_model.zaxis.label.set_color("tab:blue")
        self.ax_model.tick_params(axis="z", colors="tab:blue")
        model_range = 0.5
        self.ax_model.set_xlim(-model_range, model_range)
        self.ax_model.set_ylim(-model_range, model_range)
        self.ax_model.set_zlim(-model_range, model_range)
        self.ax_model.set_box_aspect((1, 1, 1))
        model_triad_len = 0.18
        self.ax_model.quiver(0.0, 0.0, 0.0, model_triad_len, 0.0, 0.0, color="r", arrow_length_ratio=0.15, lw=1.5)
        self.ax_model.quiver(0.0, 0.0, 0.0, 0.0, model_triad_len, 0.0, color="g", arrow_length_ratio=0.15, lw=1.5)
        self.ax_model.quiver(0.0, 0.0, 0.0, 0.0, 0.0, model_triad_len, color="b", arrow_length_ratio=0.15, lw=1.5)
        self.ax_model.text(model_triad_len * 1.1, 0.0, 0.0, "+X", color="r", fontsize=8)
        self.ax_model.text(0.0, model_triad_len * 1.1, 0.0, "+Y", color="g", fontsize=8)
        self.ax_model.text(0.0, 0.0, model_triad_len * 1.1, "+Z", color="b", fontsize=8)

        arm_len_model = 0.25
        rotor_offset = arm_len_model * 0.7
        self.nose_offset = np.array([arm_len_model * 0.45, 0.0, 0.0])
        self.arm_x_local = np.array([[-arm_len_model, 0.0, 0.0], [arm_len_model, 0.0, 0.0]])
        self.arm_y_local = np.array([[0.0, -arm_len_model, 0.0], [0.0, arm_len_model, 0.0]])
        self.rotors_local = np.array(
            [
                [rotor_offset, rotor_offset, 0.0],
                [rotor_offset, -rotor_offset, 0.0],
                [-rotor_offset, rotor_offset, 0.0],
                [-rotor_offset, -rotor_offset, 0.0],
            ]
        )

        self.model_arm_x, = self.ax_model.plot([], [], [], "tab:blue", lw=3)
        self.model_arm_y, = self.ax_model.plot([], [], [], "tab:orange", lw=3)
        self.model_body = self.ax_model.scatter([], [], [], c="k", s=28)
        self.model_rotors = self.ax_model.scatter([], [], [], c="tab:red", s=26)
        self.model_nose = self.ax_model.scatter([], [], [], c="tab:green", s=36)
        self.thrust_lines = [self.ax_model.plot([], [], [], color="purple", lw=2)[0] for _ in range(4)]

        self.ax_pos = self.fig.add_subplot(212)
        self.ax_pos.set_title("Position vs Time")
        self.ax_pos.set_xlabel("Time [s]")
        self.ax_pos.set_ylabel("Position [m]")
        self.ax_pos.plot(self.result.time, self.result.pos[:, 0], "r-", lw=1.5, label="x")
        self.ax_pos.plot(self.result.time, self.result.pos[:, 1], "g-", lw=1.5, label="y")
        self.ax_pos.plot(self.result.time, self.result.pos[:, 2], "b-", lw=1.5, label="z")
        self.pos_cursor = self.ax_pos.axvline(0.0, color="k", ls="--", lw=1.0, alpha=0.7)
        self.ax_pos.set_xlim(self.result.time[0], self.result.time[-1])
        pos_min = np.min(self.result.pos)
        pos_max = np.max(self.result.pos)
        pad = 0.1 * max(1.0, pos_max - pos_min)
        self.ax_pos.set_ylim(pos_min - pad, pos_max + pad)
        self.ax_pos.grid(alpha=0.25)
        self.ax_pos.legend(loc="upper left")

        self.airspeed = np.linalg.norm(self.result.vel, axis=1)
        self.ax_speed = self.ax_pos.twinx()
        self.ax_speed.set_ylabel("Airspeed [m/s]", color="tab:orange")
        self.ax_speed.tick_params(axis="y", colors="tab:orange")
        self.ax_speed.plot(self.result.time, self.airspeed, color="tab:orange", lw=1.4, label="airspeed")
        self.speed_cursor = self.ax_speed.axvline(0.0, color="tab:orange", ls=":", lw=1.0, alpha=0.8)
        speed_max = max(0.5, float(np.max(self.airspeed)))
        self.ax_speed.set_ylim(0.0, speed_max * 1.1)
        self.ax_speed.legend(loc="upper right")

        slider_ax = self.fig.add_axes([0.2, 0.05, 0.6, 0.03])
        self.time_slider = Slider(slider_ax, "Frame", 0, self.steps - 1, valinit=0, valstep=1)
        self.time_slider.on_changed(self._on_slider_change)

        speed_ax = self.fig.add_axes([0.2, 0.01, 0.6, 0.03])
        self.speed_slider = Slider(speed_ax, "Speed x", 0.1, 5.0, valinit=1.0, valstep=0.1)
        self.speed_slider.on_changed(self._on_speed_change)

        button_ax = self.fig.add_axes([0.83, 0.045, 0.12, 0.04])
        self.play_button = Button(button_ax, "Play")
        self.play_button.on_clicked(self._toggle_play)

        # Use a frequent UI timer and integrate by wall-clock time for stable real-time playback.
        self.timer = self.fig.canvas.new_timer(interval=16)
        self.timer.add_callback(self._on_timer)

    def _render_frame(self, frame_idx: int):
        frame_idx = int(np.clip(frame_idx, 0, self.steps - 1))
        p = self.result.pos[frame_idx]
        q = self.result.quat[frame_idx]
        rot = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()

        x_axis = p + rot[:, 0] * self.body_axis_len
        y_axis = p + rot[:, 1] * self.body_axis_len
        z_axis = p + rot[:, 2] * self.body_axis_len
        self.path_line.set_data(self.result.pos[: frame_idx + 1, 0], self.result.pos[: frame_idx + 1, 1])
        self.path_line.set_3d_properties(self.result.pos[: frame_idx + 1, 2])
        self.body_x.set_data([p[0], x_axis[0]], [p[1], x_axis[1]])
        self.body_x.set_3d_properties([p[2], x_axis[2]])
        self.body_y.set_data([p[0], y_axis[0]], [p[1], y_axis[1]])
        self.body_y.set_3d_properties([p[2], y_axis[2]])
        self.body_z.set_data([p[0], z_axis[0]], [p[1], z_axis[1]])
        self.body_z.set_3d_properties([p[2], z_axis[2]])
        self.point._offsets3d = ([p[0]], [p[1]], [p[2]])
        if self.result.desired_path is not None and len(self.result.desired_path) > 0 and self.active_waypoint_point is not None:
            active_idx = int(np.clip(self.result.target_waypoint_idx[frame_idx], 0, len(self.result.desired_path) - 1))
            active_wp = self.result.desired_path[active_idx]
            self.active_waypoint_point._offsets3d = ([active_wp[0]], [active_wp[1]], [active_wp[2]])

        arm_x_world = (rot @ self.arm_x_local.T).T
        arm_y_world = (rot @ self.arm_y_local.T).T
        rotors_world = (rot @ self.rotors_local.T).T
        nose_world = rot @ self.nose_offset
        thrust_dir_world = rot @ np.array([0.0, 0.0, 1.0])
        self.model_arm_x.set_data(arm_x_world[:, 0], arm_x_world[:, 1])
        self.model_arm_x.set_3d_properties(arm_x_world[:, 2])
        self.model_arm_y.set_data(arm_y_world[:, 0], arm_y_world[:, 1])
        self.model_arm_y.set_3d_properties(arm_y_world[:, 2])
        self.model_body._offsets3d = ([0.0], [0.0], [0.0])
        self.model_rotors._offsets3d = (rotors_world[:, 0], rotors_world[:, 1], rotors_world[:, 2])
        self.model_nose._offsets3d = ([nose_world[0]], [nose_world[1]], [nose_world[2]])

        motor_thrusts = self.result.motor_thrust[frame_idx]
        for idx, thrust_line in enumerate(self.thrust_lines):
            rotor = rotors_world[idx]
            thrust_mag = motor_thrusts[idx]
            thrust_len = 0.06 + 0.18 * np.clip(
                thrust_mag / self.result.hover_thrust_per_motor, 0.0, 2.0
            )
            thrust_tip = rotor + thrust_dir_world * thrust_len
            thrust_line.set_data([rotor[0], thrust_tip[0]], [rotor[1], thrust_tip[1]])
            thrust_line.set_3d_properties([rotor[2], thrust_tip[2]])

        self.pos_cursor.set_xdata([self.result.time[frame_idx], self.result.time[frame_idx]])
        self.speed_cursor.set_xdata([self.result.time[frame_idx], self.result.time[frame_idx]])
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, value):
        self.current_frame = int(value)
        self.playhead = float(self.current_frame)
        self._render_frame(self.current_frame)

    def _on_speed_change(self, value):
        self.playback_speed = float(value)

    def _on_timer(self):
        if not self.is_playing:
            return
        now = perf_counter()
        if self.last_tick_wall is None:
            self.last_tick_wall = now
            return
        elapsed = now - self.last_tick_wall
        self.last_tick_wall = now

        if self.playhead >= self.steps - 1:
            self.is_playing = False
            self.timer.stop()
            self.play_button.label.set_text("Play")
            self.last_tick_wall = None
            return

        # speed=1.0 => realtime: advance elapsed / dt frames.
        frame_advance = (elapsed / self.result.dt) * self.playback_speed
        self.playhead = min(self.steps - 1, self.playhead + frame_advance)
        next_frame = int(self.playhead)
        if next_frame != self.current_frame:
            self.current_frame = next_frame
            self.time_slider.set_val(self.current_frame)

    def _toggle_play(self, _event):
        self.is_playing = not self.is_playing
        self.play_button.label.set_text("Pause" if self.is_playing else "Play")
        if self.is_playing:
            if self.current_frame >= self.steps - 1:
                self.current_frame = 0
                self.playhead = 0.0
                self.time_slider.set_val(0)
            self.last_tick_wall = perf_counter()
            self.timer.stop()
            self.timer.start()
        else:
            self.timer.stop()
            self.last_tick_wall = None

    def show(self):
        self._build_figure()
        self._render_frame(0)
        plt.show()
