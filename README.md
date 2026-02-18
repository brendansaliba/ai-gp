## Quadcopter Simulation (Matplotlib)

[![Built with Codex](https://img.shields.io/badge/Built%20with-Codex-1f2937?style=for-the-badge&labelColor=0f172a&color=22c55e)](https://openai.com/codex)

This project is a lightweight quadcopter dynamics simulator with an interactive Matplotlib viewer.

> [!NOTE]
> This project was built collaboratively with AI assistance from Codex. ALL code has been reviewed by Human engineers.

> [!IMPORTANT]
> Approximations have been made in the aerodynamic modeling and simulation of this project.
> Assumptions are clearly stated in this README.

Current implementation includes:

- 6-DoF rigid-body translational and rotational simulation (simplified)
- Quaternion-based attitude integration
- Motor-thrust-driven force and torque model
- Optional small perturbations to emulate atmosphere/aerodynamic disturbances
- Interactive playback UI (frame scrubber, play/pause, real-time speed scaling)

## Project Structure

- `src/main.py`: Entrypoint. Loads YAML config, builds motor-thrust input function, runs simulation, launches viewer.
- `src/drone_dynamics.py`: Physics and integration logic.
- `src/matplotlib_viewer.py`: Visualization and playback controls.
- `config/drone_dynamics.yaml`: User-editable simulation and dynamics configuration.

## Run

1. Install dependencies:
   `pip install -r requirements.txt`

2. Run:
   `python src/main.py`

## Configuration

All runtime parameters are in:
`config/drone_dynamics.yaml`

Key groups:

- `simulation`: `dt`, `steps`, `gravity`
- `dynamics`: mass, inertia, arm length, yaw drag coefficient, attitude-drag parameters (`drag_coeff_min`, `drag_coeff_max`, `drag_reference_area`, `air_density`, `wind_velocity_world`), motor spin directions
- `initial_state`: initial position/velocity/attitude/angular velocity
- `thrust`: `mode: hover` uses equal hover thrust on all motors; `mode: manual` uses constant per-motor values from `manual_motor_thrust_newtons`
- `perturbations`: small stochastic + sinusoidal disturbances

## Physics Model

### State

The simulated state is:

- Position $p$ and velocity $v$ in world frame
- Attitude quaternion $q = [w, x, y, z]$
- Body angular velocity $\omega = [p, q, r]$

### Forces and Torques

At each time step:

- Motor thrusts $T_i$ are provided by the configured motor-thrust function.
- Total body thrust is $T = \sum_i T_i$ along body $+z$.
- Thrust is rotated to world frame using the quaternion-derived rotation matrix $R(q)$.
- Translational acceleration:
  $$
  a = \frac{R(q)\begin{bmatrix}0\\0\\T\end{bmatrix}}{m} + \begin{bmatrix}0\\0\\-g\end{bmatrix}
  $$

Body torques include:

- Arm moment from each rotor:
  $$
  \tau_{\mathrm{arm}} = \sum_i r_i \times F_i
  $$
- Yaw drag torque:
  $$
  \tau_{\mathrm{yaw}} = \begin{bmatrix}0\\0\\k_{\mathrm{yaw}}\left(\mathrm{spin\_dir}\cdot T\right)\end{bmatrix}
  $$
- Optional perturbation torque (small gust/noise term)

Total torque:

$$
\tau = \tau_{\mathrm{arm}} + \tau_{\mathrm{yaw}} + \tau_{\mathrm{perturb}}
$$

### Aerodynamic Drag Model

The translational drag force is based on relative airflow and current vehicle attitude.

- Relative airflow velocity:
  $$
  v_{\mathrm{rel}} = v - v_{\mathrm{wind}}
  $$
- Airflow direction:
  $$
  \hat{a} = -\frac{v_{\mathrm{rel}}}{\|v_{\mathrm{rel}}\|}
  $$
- Body normal in world frame:
  $$
  n_b = R(q)\begin{bmatrix}0\\0\\1\end{bmatrix}
  $$
- Orientation alignment term:
  $$
  s = \left|n_b \cdot \hat{a}\right|
  $$
  where $s=0$ means airflow is mostly along the vehicle plane (minimum drag), and $s=1$ means airflow is normal to the plane (maximum drag).
- Drag coefficient interpolation:
  $$
  C_d = C_{d,\min} + \left(C_{d,\max} - C_{d,\min}\right)s
  $$
- Drag force magnitude:
  $$
  \|F_d\| = \frac{1}{2}\rho C_d A \|v_{\mathrm{rel}}\|^2
  $$
- Drag direction (opposes relative motion):
  $$
  F_d = -\|F_d\|\frac{v_{\mathrm{rel}}}{\|v_{\mathrm{rel}}\|}
  $$

The translational update uses thrust + drag + gravity:
$$
a = \frac{R(q)\begin{bmatrix}0\\0\\T\end{bmatrix} + F_d}{m} + \begin{bmatrix}0\\0\\-g\end{bmatrix}
$$

### Rotational Dynamics

Rigid-body rotational dynamics are integrated as:

$$
I\dot{\omega} = \tau - \omega \times (I\omega)
$$

Quaternion update uses:

$$
\dot{q} = \frac{1}{2} q \otimes [0,\omega]
$$

with renormalization every step.

### Integration Method

The simulator uses explicit Euler integration for both translational and rotational states.

## Approximations and Assumptions

This is intentionally simplified for clarity and interactivity:

- Rigid body with fixed diagonal inertia tensor (no inertia coupling changes)
- Drag uses a single scalar coefficient interpolated by attitude/airflow alignment (not a full 6-DoF aero model)
- No rotor dynamics (instantaneous thrust response)
- No motor saturation dynamics except non-negative thrust clipping
- No ground effect, blade flapping, induced-flow model, or spatially varying wind field model
- Fixed time step, first-order integration (can accumulate numerical error)
- Yaw drag represented by a linear coefficient (`yaw_drag_coeff`)

These assumptions make the model easy to understand and tune, but not high-fidelity for controller validation against real hardware.

## Viewer Behavior

The Matplotlib viewer includes:

- 3D trajectory and body-axis visualization
- Attitude model with rotor thrust vectors
- Position-vs-time plot with time cursor
- Frame slider for scrubbing
- Play/Pause and speed control

`Speed x = 1.0` is real-time playback relative to simulation time.
