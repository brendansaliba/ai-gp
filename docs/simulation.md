# Simulation

## Simulation pseudocode

```python
import math
import random

# seconds
start_time = 0
end_time = 10
dt = 0.005

# initial state
P = [0, 0, 10]      # inital position, described as x, y, z vector
P_dot = [0, 0, 0]       # initial velocity, described as x, y, x vector
theta =  [0, 0, 0]      # initial attitude, described as pitch (phi), roll (theta), yaw (psi)
theta_dot = [0, 0, 0] # initial attidue rates (rad/s)

# define an inital disturbance
deviation = 100
disturbance = [random.random() for _ in range(3)]
theta_dot = [math.radians((2 * deviation * d - deviation)) for d in disturbance]
```

After the initial state is defined, we can loop through the specified time interval to simulate flight.

```python
for t in range(start_time, end_time, dt):
    # calculate omega matrix from theta and theta dot
    # calculate linear from theta, xdot, m, g, k, and kd
    # calculate omega dot (angular acceleration) from omega, I, L, b, and k

    # omega = omega + dt * omega_dot
    # theta_dot = omega to theta_dot given omega and theta
    # theta = theta + dt * theta_dot
    # P_dot = P_dot + dt * a
    # P = P + dt * P_dot
```
