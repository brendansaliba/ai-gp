import numpy as np
from pathlib import Path
import yaml
from scipy.spatial.transform import Rotation

from drone_dynamics import (
    DroneDynamicsConfig,
    quat_to_euler,
    simulate_quadcopter,
)
from control import SimpleFlightController
from matplotlib_viewer import MatplotlibQuadViewer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_motor_thrust_fn(thrust_cfg: dict):
    mode = thrust_cfg.get("mode", "hover")

    if mode == "hover":
        def hover_motor_thrust(_t, _step_idx, _cfg, hover_thrust_per_motor):
            return np.array([hover_thrust_per_motor] * 4, dtype=float)

        return hover_motor_thrust

    if mode == "manual":
        manual_values = np.asarray(
            thrust_cfg.get("manual_motor_thrust_newtons", []), dtype=float
        )
        if manual_values.shape != (4,):
            raise ValueError(
                "For thrust mode 'manual', manual_motor_thrust_newtons must be a 4-element list."
            )

        def manual_motor_thrust(_t, _step_idx, _cfg, _hover_thrust_per_motor):
            return manual_values

        return manual_motor_thrust

    if mode == "flight_controller":
        waypoints_xyz = np.asarray(thrust_cfg.get("waypoints_xyz", []), dtype=float)
        if waypoints_xyz.ndim != 2 or waypoints_xyz.shape[1] != 3 or len(waypoints_xyz) == 0:
            raise ValueError(
                "For thrust mode 'flight_controller', waypoints_xyz must be a non-empty [N, 3] list."
            )
        ctrl_cfg = thrust_cfg.get("controller", {})
        controller = SimpleFlightController(
            waypoints_xyz=waypoints_xyz,
            waypoint_tolerance_m=float(ctrl_cfg.get("waypoint_tolerance_m", 0.6)),
            max_speed_mps=float(ctrl_cfg.get("max_speed_mps", 3.0)),
            max_accel_mps2=float(ctrl_cfg.get("max_accel_mps2", 4.0)),
            max_tilt_deg=float(ctrl_cfg.get("max_tilt_deg", 30.0)),
            max_motor_thrust_n=float(ctrl_cfg.get("max_motor_thrust_n", 20.0)),
            max_yaw_rate_rad_s=float(ctrl_cfg.get("max_yaw_rate_rad_s", 0.8)),
            kp_pos=float(ctrl_cfg.get("kp_pos", 0.8)),
            ki_pos=float(ctrl_cfg.get("ki_pos", 0.08)),
            kd_vel=float(ctrl_cfg.get("kd_vel", 1.4)),
            kp_att_rp=float(ctrl_cfg.get("kp_att_rp", 10.0)),
            ki_att_rp=float(ctrl_cfg.get("ki_att_rp", 0.8)),
            kd_rate_xy=float(ctrl_cfg.get("kd_rate_xy", 0.8)),
            kp_yaw=float(ctrl_cfg.get("kp_yaw", 2.0)),
            ki_yaw=float(ctrl_cfg.get("ki_yaw", 0.2)),
            kd_yaw_rate=float(ctrl_cfg.get("kd_yaw_rate", 0.35)),
            pos_integral_limit=float(ctrl_cfg.get("pos_integral_limit", 6.0)),
            att_integral_limit=float(ctrl_cfg.get("att_integral_limit", 0.7)),
            yaw_integral_limit=float(ctrl_cfg.get("yaw_integral_limit", 1.2)),
        )

        def flight_controller_thrust(t, step_idx, cfg, hover_thrust_per_motor, state):
            return controller.motor_thrust_command(t, step_idx, cfg, hover_thrust_per_motor, state)

        return flight_controller_thrust

    raise ValueError(
        f"Unsupported thrust.mode '{mode}'. Use 'hover', 'manual', or 'flight_controller'."
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "drone_dynamics.yaml"
    raw = load_config(config_path)

    sim_cfg = raw["simulation"]
    dyn_cfg = raw["dynamics"]
    init_cfg = raw["initial_state"]
    thrust_cfg = raw["thrust"]
    pert_cfg = raw.get("perturbations", {})

    init_euler_deg = np.asarray(init_cfg["euler_deg"], dtype=float)
    init_quat_xyzw = Rotation.from_euler("xyz", init_euler_deg, degrees=True).as_quat()
    init_quat_wxyz = np.array(
        [init_quat_xyzw[3], init_quat_xyzw[0], init_quat_xyzw[1], init_quat_xyzw[2]],
        dtype=float,
    )

    thrust_fn = build_motor_thrust_fn(thrust_cfg)
    desired_path = None
    if thrust_cfg.get("mode", "hover") == "flight_controller":
        desired_path = np.asarray(thrust_cfg["waypoints_xyz"], dtype=float)

    cfg = DroneDynamicsConfig(
        dt=float(sim_cfg["dt"]),
        steps=int(sim_cfg["steps"]),
        g=float(sim_cfg.get("gravity", 9.81)),
        mass=float(dyn_cfg["mass"]),
        inertia=np.diag(np.asarray(dyn_cfg["inertia_diag"], dtype=float)),
        arm_length=float(dyn_cfg["arm_length"]),
        yaw_drag_coeff=float(dyn_cfg["yaw_drag_coeff"]),
        enable_attitude_drag=bool(dyn_cfg.get("enable_attitude_drag", True)),
        drag_coeff_min=float(dyn_cfg["drag_coeff_min"]),
        drag_coeff_max=float(dyn_cfg["drag_coeff_max"]),
        drag_reference_area=float(dyn_cfg.get("drag_reference_area", 0.03)),
        air_density=float(dyn_cfg.get("air_density", 1.225)),
        wind_velocity_world=np.asarray(dyn_cfg.get("wind_velocity_world", [0.0, 0.0, 0.0]), dtype=float),
        motor_spin_dir=np.asarray(dyn_cfg["motor_spin_dir"], dtype=float),
        initial_pos=np.asarray(init_cfg["position"], dtype=float),
        initial_vel=np.asarray(init_cfg["velocity"], dtype=float),
        initial_quat=init_quat_wxyz,
        initial_ang_vel=np.asarray(init_cfg["angular_velocity"], dtype=float),
        enable_perturbations=bool(pert_cfg.get("enable", True)),
        perturbation_seed=pert_cfg.get("perturbation_seed", 42),
        motor_thrust_noise_std=float(pert_cfg.get("motor_thrust_noise_std", 0.005)),
        motor_thrust_gust_amp=float(pert_cfg.get("motor_thrust_gust_amp", 0.01)),
        attitude_torque_noise_std=float(pert_cfg.get("attitude_torque_noise_std", 2e-5)),
        attitude_torque_gust_amp=float(pert_cfg.get("attitude_torque_gust_amp", 5e-5)),
        desired_path=desired_path,
        motor_thrust_fn=thrust_fn,
    )
    result = simulate_quadcopter(cfg)

    print("t    roll°   pitch°   yaw°")
    for frame in range(0, cfg.steps, 40):
        roll, pitch, yaw = np.degrees(quat_to_euler(result.quat[frame]))
        print(f"{frame * cfg.dt:4.1f}  {roll:6.2f}   {pitch:6.2f}   {yaw:6.2f} -- position {result.pos[frame]}")

    final_roll, final_pitch, final_yaw = np.degrees(quat_to_euler(result.quat[-1]))
    print("\nFinal attitude (deg):")
    print(
        f"roll = {final_roll:6.2f}°,  pitch = {final_pitch:6.2f}°,  yaw = {final_yaw:6.2f}°"
    )

    viewer = MatplotlibQuadViewer(result=result, body_axis_len=0.25)
    viewer.show()


if __name__ == "__main__":
    main()
