import numpy as np
from pathlib import Path
import yaml
from scipy.spatial.transform import Rotation

from drone_dynamics import (
    DroneDynamicsConfig,
    quat_to_euler,
    simulate_quadcopter,
)
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

        return manual_values

    raise ValueError(f"Unsupported thrust.mode '{mode}'. Use 'hover' or 'manual'.")


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
        motor_thrust_fn=build_motor_thrust_fn(thrust_cfg),
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
