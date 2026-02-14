import math
import random

# timing
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
