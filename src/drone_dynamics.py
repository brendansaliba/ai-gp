from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q1 * q2 for quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to XYZ euler angles in radians."""
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz", degrees=False)


@dataclass
class DroneDynamicsConfig:
    dt: float = 0.01
    steps: int = 200
    g: float = 9.81
    mass: float = 0.85
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.002, 0.002, 0.004]))
    arm_length: float = 0.17
    yaw_drag_coeff: float = 0.004
    motor_spin_dir: np.ndarray = field(default_factory=lambda: np.array([1.0, -1.0, 1.0, -1.0]))
    initial_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 10.0]))
    initial_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initial_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    initial_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Small perturbations to emulate atmospheric / aerodynamic effects.
    enable_perturbations: bool = True
    motor_thrust_noise_std: float = 0.005  # N
    motor_thrust_gust_amp: float = 0.01  # N
    attitude_torque_noise_std: float = 2e-5  # N*m
    attitude_torque_gust_amp: float = 5e-5  # N*m
    perturbation_seed: int | None = 42
    # Required motor thrust provider:
    # fn(t, step_idx, cfg, hover_thrust_per_motor) -> array-like(4) [N]
    motor_thrust_fn: Callable[[float, int, "DroneDynamicsConfig", float], np.ndarray] | None = None


@dataclass
class SimulationResult:
    dt: float
    time: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    motor_thrust: np.ndarray
    torque: np.ndarray
    hover_thrust_per_motor: float


def simulate_quadcopter(cfg: DroneDynamicsConfig) -> SimulationResult:
    if cfg.motor_thrust_fn is None:
        raise ValueError("DroneDynamicsConfig.motor_thrust_fn is required.")

    pos = cfg.initial_pos.astype(float).copy()
    vel = cfg.initial_vel.astype(float).copy()
    quat = cfg.initial_quat.astype(float).copy()
    ang_vel = cfg.initial_ang_vel.astype(float).copy()

    hover_thrust_per_motor = cfg.mass * cfg.g / 4.0
    rng = np.random.default_rng(cfg.perturbation_seed)
    motor_phase = np.array([0.0, 1.2, 2.1, 2.9])
    torque_phase = np.array([0.5, 1.7, 2.6])
    motor_positions = np.array(
        [
            [cfg.arm_length, cfg.arm_length, 0.0],
            [cfg.arm_length, -cfg.arm_length, 0.0],
            [-cfg.arm_length, -cfg.arm_length, 0.0],
            [-cfg.arm_length, cfg.arm_length, 0.0],
        ]
    )

    time = np.arange(cfg.steps) * cfg.dt
    pos_history = np.zeros((cfg.steps, 3))
    quat_history = np.zeros((cfg.steps, 4))
    motor_thrust_history = np.zeros((cfg.steps, 4))
    torque_history = np.zeros((cfg.steps, 3))

    for i in range(cfg.steps):
        t = i * cfg.dt
        motor_thrusts = np.asarray(cfg.motor_thrust_fn(t, i, cfg, hover_thrust_per_motor), dtype=float)
        if motor_thrusts.shape != (4,):
            raise ValueError("motor_thrust_fn must return a 4-element array [m1, m2, m3, m4] in Newtons.")

        if cfg.enable_perturbations:
            gust = cfg.motor_thrust_gust_amp * (
                np.sin(0.7 * t + motor_phase) + 0.5 * np.sin(1.6 * t + 0.7 * motor_phase)
            )
            noise = rng.normal(0.0, cfg.motor_thrust_noise_std, size=4)
            motor_thrusts = motor_thrusts + gust + noise
        motor_thrusts = np.clip(motor_thrusts, 0.0, None)

        total_thrust = np.sum(motor_thrusts)
        motor_forces = np.column_stack([np.zeros(4), np.zeros(4), motor_thrusts])
        torque_from_arms = np.sum(np.cross(motor_positions, motor_forces), axis=0)
        torque_yaw = np.array([0.0, 0.0, cfg.yaw_drag_coeff * np.dot(cfg.motor_spin_dir, motor_thrusts)])
        torque_body = torque_from_arms + torque_yaw
        if cfg.enable_perturbations:
            torque_gust = cfg.attitude_torque_gust_amp * np.array(
                [
                    np.sin(0.45 * t + torque_phase[0]),
                    np.cos(0.52 * t + torque_phase[1]),
                    np.sin(0.61 * t + torque_phase[2]),
                ]
            )
            torque_noise = rng.normal(0.0, cfg.attitude_torque_noise_std, size=3)
            torque_body = torque_body + torque_gust + torque_noise

        ang_acc = np.linalg.solve(cfg.inertia, torque_body - np.cross(ang_vel, cfg.inertia @ ang_vel))
        ang_vel += ang_acc * cfg.dt

        omega_quat = np.array([0.0, *ang_vel])
        quat += 0.5 * quat_multiply(quat, omega_quat) * cfg.dt
        quat /= np.linalg.norm(quat)

        rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        thrust_world = rot @ np.array([0.0, 0.0, total_thrust])
        acc_world = thrust_world / cfg.mass + np.array([0.0, 0.0, -cfg.g])
        vel += acc_world * cfg.dt
        pos += vel * cfg.dt

        pos_history[i] = pos
        quat_history[i] = quat
        motor_thrust_history[i] = motor_thrusts
        torque_history[i] = torque_body

    return SimulationResult(
        dt=cfg.dt,
        time=time,
        pos=pos_history,
        quat=quat_history,
        motor_thrust=motor_thrust_history,
        torque=torque_history,
        hover_thrust_per_motor=hover_thrust_per_motor,
    )
