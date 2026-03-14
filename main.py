#----------------------------------------------------------------------------------------

# Python code to simulate the 3D trajectory of a missile under gravity, thrust, and drag.
# The code calculates the position, velocity, and acceleration of the missile over time,
# and visualizes the trajectory in a 3D plot.
# Author: Abel Philippe

#----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp

from scipy import integrate

# ==============================================================================
#                              Physical parameters
# ==============================================================================

px = 0.0
py = 0.0
pz = 0.0

vx = 10.0
vy = 20.0
vz = 3.0

# -----------------------------------------------------------------------------
#                               Initial state
# -----------------------------------------------------------------------------

traj = []

pos = np.array([px, py, pz])            # position vector (px, py, pz)

vel = np.array([vx, vy, vz])            # velocity vector (vx, vy, vz)

# -----------------------------------------------------------------------------

g = 9.81                                # gravity (m/s²)

h = pos[2]                              # altitude (m)

rho = 1.225                             # air density (kg/m³)

Cd = 0.5                                # drag coefficient

A = 0.005                               # frontal area (m²)

velocity = np.linalg.norm(vel)          # speed magnitude

Q = 0.5 * rho * velocity**2             # dynamic pressure

mass = 130                              # mass (kg)

d = 3.6                                 # reference length (m)

burn_time = 6                           # burn time (s)

fuel_mass = 40                          # fuel mass (kg)

burn_rate = fuel_mass/burn_time         # fuel burn rate (kg/s)

def air_density(h):                     # function to calculate air density based on altitude
    rho = 1.225
    H = 8500
    return rho*np.exp(-h/H)

# -----------------------------------------------------------------------------
#                                  Forces
# -----------------------------------------------------------------------------

Fg = np.array([0,0,-mass*g])                                # gravity force vector

Fa = -0.5 * rho * velocity**2 * Cd * A * (vel/velocity)     # drag force vector

thrust = np.array([0,0,20000])                              # thrust vector

# -----------------------------------------------------------------------------
#                             Resultant force
# -----------------------------------------------------------------------------

F = thrust + Fa + Fg                           # total force vector

# -----------------------------------------------------------------------------
#                               Acceleration
# -----------------------------------------------------------------------------

a = F / mass                                   # acceleration vector

# -----------------------------------------------------------------------------
#                                 Inertia
# -----------------------------------------------------------------------------

I = (1/12) * mass * d**2                       # inertia approximation (kg·m²)

# -----------------------------------------------------------------------------
#                               Time parameters
# -----------------------------------------------------------------------------

dt = 0.02
t_max = 60

# ==============================================================================

# ------------------------------------
#           Simulation loop
# ------------------------------------

for t in np.arange(0,t_max,dt):

    h = pos[2]
    rho = air_density(h)

    velocity = np.linalg.norm(vel)
    Fg = np.array([0,0,-mass*g])


    if t < burn_time and fuel_mass > 0:
        thrust = np.array([0,0,20000])
        dm = burn_rate*dt
        fuel_mass -= dm
        mass -= dm
    else:
        thrust = np.array([0,0,0])

 
    if velocity > 1e-6:
        drag_mag = 0.5*rho*Cd*A*velocity**2
        Fd = -drag_mag*(vel/velocity)
    else:
        Fd = np.array([0,0,0])

    F = thrust + Fg + Fd
    acc = F/mass
    vel += acc*dt
    pos += vel*dt

    traj.append(pos.copy())

    if pos[2] < 0:
        break

traj = np.array(traj)

# -----------------------------
#           Plot 3D
# -----------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj[:,0],traj[:,1],traj[:,2],linewidth=2)
ax.set_box_aspect([1,1,1])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title("Simulation of 3D Trajectory")

plt.show()