from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from drone_dynamics import DroneDynamicsConfig, SimulationState


def _wrap_angle_rad(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class SimpleFlightController:
    waypoints_xyz: np.ndarray
    waypoint_tolerance_m: float = 0.6
    max_speed_mps: float = 3.0
    max_accel_mps2: float = 4.0
    max_tilt_deg: float = 20.0
    max_motor_thrust_n: float = 20.0
    max_yaw_rate_rad_s: float = 0.8
    kp_pos: float = 0.7
    ki_pos: float = 0.03
    kd_vel: float = 1.2
    kp_att_rp: float = 4.0
    ki_att_rp: float = 0.05
    kd_rate_xy: float = 1.0
    kp_yaw: float = 1.6
    ki_yaw: float = 0.05
    kd_yaw_rate: float = 0.45
    pos_integral_limit: float = 6.0
    att_integral_limit: float = 0.4
    yaw_integral_limit: float = 1.2
    max_torque_xy_nm: float = 0.8
    max_torque_z_nm: float = 0.4

    def __post_init__(self):
        path = np.asarray(self.waypoints_xyz, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("waypoints_xyz must be shape [N, 3].")
        if len(path) == 0:
            raise ValueError("At least one waypoint is required.")
        self.waypoints_xyz = path
        self.current_waypoint_idx = 0
        self.desired_yaw_rad = 0.0
        self.pos_integral = np.zeros(3, dtype=float)
        self.att_integral_rp = np.zeros(2, dtype=float)
        self.yaw_integral = 0.0

    def _update_active_waypoint(self, position_world: np.ndarray) -> np.ndarray:
        while self.current_waypoint_idx < len(self.waypoints_xyz) - 1:
            waypoint = self.waypoints_xyz[self.current_waypoint_idx]
            if np.linalg.norm(waypoint - position_world) <= self.waypoint_tolerance_m:
                self.current_waypoint_idx += 1
            else:
                break
        return self.waypoints_xyz[self.current_waypoint_idx]

    def motor_thrust_command(
        self,
        _t: float,
        _step_idx: int,
        cfg: DroneDynamicsConfig,
        _hover_thrust_per_motor: float,
        state: SimulationState,
    ) -> tuple[np.ndarray, int]:
        target_wp = self._update_active_waypoint(state.pos)
        to_wp = target_wp - state.pos
        dist_to_wp = np.linalg.norm(to_wp)
        wp_dir = to_wp / max(dist_to_wp, 1e-9)
        dt = max(cfg.dt, 1e-4)

        self.pos_integral += to_wp * dt
        self.pos_integral = np.clip(self.pos_integral, -self.pos_integral_limit, self.pos_integral_limit)

        desired_vel = wp_dir * min(self.max_speed_mps, dist_to_wp)
        vel_err = desired_vel - state.vel
        accel_cmd_world = self.kp_pos * to_wp + self.ki_pos * self.pos_integral + self.kd_vel * vel_err
        accel_cmd_world[2] = np.clip(accel_cmd_world[2], -3.0, 3.0)
        accel_norm = np.linalg.norm(accel_cmd_world)
        if accel_norm > self.max_accel_mps2:
            accel_cmd_world *= self.max_accel_mps2 / accel_norm

        if dist_to_wp > 1e-6:
            yaw_target = np.arctan2(wp_dir[1], wp_dir[0])
            yaw_step_max = self.max_yaw_rate_rad_s * dt
            yaw_delta = np.clip(_wrap_angle_rad(yaw_target - self.desired_yaw_rad), -yaw_step_max, yaw_step_max)
            self.desired_yaw_rad = _wrap_angle_rad(self.desired_yaw_rad + yaw_delta)

        rot_bw = Rotation.from_quat(state.quat[[1, 2, 3, 0]]).as_matrix()
        z_body_world = rot_bw[:, 2]
        yaw_world = np.arctan2(rot_bw[1, 0], rot_bw[0, 0])
        yaw_err = _wrap_angle_rad(self.desired_yaw_rad - yaw_world)
        roll, pitch, _ = Rotation.from_quat(state.quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=False)

        yaw_des = self.desired_yaw_rad
        pitch_des = (accel_cmd_world[0] * np.cos(yaw_des) + accel_cmd_world[1] * np.sin(yaw_des)) / max(cfg.g, 1e-9)
        roll_des = (accel_cmd_world[0] * np.sin(yaw_des) - accel_cmd_world[1] * np.cos(yaw_des)) / max(cfg.g, 1e-9)
        max_tilt = np.deg2rad(self.max_tilt_deg)
        pitch_des = np.clip(pitch_des, -max_tilt, max_tilt)
        roll_des = np.clip(roll_des, -max_tilt, max_tilt)

        roll_err = roll_des - roll
        pitch_err = pitch_des - pitch
        self.att_integral_rp[0] = np.clip(
            self.att_integral_rp[0] + roll_err * dt, -self.att_integral_limit, self.att_integral_limit
        )
        self.att_integral_rp[1] = np.clip(
            self.att_integral_rp[1] + pitch_err * dt, -self.att_integral_limit, self.att_integral_limit
        )
        self.yaw_integral = np.clip(self.yaw_integral + yaw_err * dt, -self.yaw_integral_limit, self.yaw_integral_limit)

        tau_x = (
            self.kp_att_rp * roll_err
            + self.ki_att_rp * self.att_integral_rp[0]
            - self.kd_rate_xy * state.ang_vel[0]
        )
        tau_y = (
            self.kp_att_rp * pitch_err
            + self.ki_att_rp * self.att_integral_rp[1]
            - self.kd_rate_xy * state.ang_vel[1]
        )
        tau_z = self.kp_yaw * yaw_err + self.ki_yaw * self.yaw_integral - self.kd_yaw_rate * state.ang_vel[2]
        tau_x = float(np.clip(tau_x, -self.max_torque_xy_nm, self.max_torque_xy_nm))
        tau_y = float(np.clip(tau_y, -self.max_torque_xy_nm, self.max_torque_xy_nm))
        tau_z = float(np.clip(tau_z, -self.max_torque_z_nm, self.max_torque_z_nm))

        total_force_world = cfg.mass * (accel_cmd_world + np.array([0.0, 0.0, cfg.g]))
        total_thrust = float(np.dot(total_force_world, z_body_world))
        total_thrust = np.clip(total_thrust, 0.0, 4.0 * self.max_motor_thrust_n)

        l = cfg.arm_length
        k_yaw = cfg.yaw_drag_coeff
        s = cfg.motor_spin_dir.astype(float)
        mix = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [l, -l, -l, l],
                [-l, -l, l, l],
                [k_yaw * s[0], k_yaw * s[1], k_yaw * s[2], k_yaw * s[3]],
            ],
            dtype=float,
        )
        wrench = np.array([total_thrust, tau_x, tau_y, tau_z], dtype=float)
        motor_thrust = np.linalg.solve(mix, wrench)
        motor_thrust = np.clip(motor_thrust, 0.0, self.max_motor_thrust_n)

        return motor_thrust, self.current_waypoint_idx
