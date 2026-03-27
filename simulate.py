import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import proj3d

# -------------------------
# CONFIG
# -------------------------

dt = 0.0015
t_max = 10.0
g = np.array([0.0, 0.0, -9.81])

rho0 = 1.225
H = 7200
A = 0.045

# -------------------------
# INITIAL STATE (3D)
# -------------------------

missile_pos = np.array([0.0, 0.0, 0.0])
missile_vel = np.array([0.0, 0.0, 0.0])

target_pos = np.array([18000.0, 5000.0, 35000.0])
target_vel = np.array([-4200.0, -500.0, -6500.0])

# -------------------------
# PARAMETERS
# -------------------------

NAV_CONST = 4.0
MAX_GUIDANCE_ACCEL = 90 * 9.81

# -------------------------
# SPRINT MODEL
# -------------------------

def missile_mass(t):
    if t < 1.2:
        return 3500 - 1500 * (t / 1.2)
    elif t < 5.0:
        return 2000 - 800 * ((t - 1.2) / 3.8)
    else:
        return 1200

def thrust(t):
    if t < 1.2:
        return 18e6
    elif t < 5.0:
        return 7.5e6
    else:
        return 0.0

# -------------------------
# AUX
# -------------------------

def norm(v):
    return np.linalg.norm(v)

def unit(v):
    n = norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)

def air_density(h):
    return rho0 * np.exp(-h / H)

def drag_coefficient(speed):
    mach = speed / 343
    if mach < 1: return 0.2
    elif mach < 5: return 0.3
    elif mach < 10: return 0.6
    else: return 0.9

# -------------------------
# PROPULSION
# -------------------------

def propulsion(t, vel, rel_pos):
    m = missile_mass(t)
    T = thrust(t)

    if t < 1.5:
        direction = unit(rel_pos)
    else:
        direction = unit(vel) if norm(vel) > 1 else unit(rel_pos)

    a = (T / m) * direction

    max_acc = 120 * 9.81
    if norm(a) > max_acc:
        a = unit(a) * max_acc

    return a, m

# -------------------------
# GUIDANCE 3D
# -------------------------

def guidance(t, rel_pos, rel_vel):

    if t < 0.5:
        return np.zeros(3)

    r = rel_pos
    v = rel_vel

    dist = norm(r)
    los = unit(r)

    omega = np.cross(r, v) / (dist**2 + 1e-6)
    closing = -np.dot(v, los)

    a = NAV_CONST * closing * np.cross(omega, los)

    if norm(a) > MAX_GUIDANCE_ACCEL:
        a = unit(a) * MAX_GUIDANCE_ACCEL

    return a

# -------------------------
# DRAG
# -------------------------

def drag(pos, vel, m):

    speed = norm(vel)
    if speed < 1e-3:
        return np.zeros(3)

    rho = air_density(pos[2])
    Cd = drag_coefficient(speed)

    D = 0.5 * rho * speed**2 * Cd * A
    return -(D / m) * unit(vel)

# -------------------------
# SIMULATION
# -------------------------

missile_traj = []
target_traj = []
missile_speed = []

t = 0.0
hit = False

while t < t_max:

    rel_pos = target_pos - missile_pos
    rel_vel = target_vel - missile_vel

    if norm(rel_pos) < 40:
        print(f"Intercept in t = {t:.2f}s")
        hit = True
        break

    a_prop, m = propulsion(t, missile_vel, rel_pos)
    a_guid = guidance(t, rel_pos, rel_vel)
    a_drag = drag(missile_pos, missile_vel, m)

    a_total = a_prop + a_guid + a_drag + g

    missile_vel += a_total * dt
    missile_pos += missile_vel * dt

    target_vel += g * dt
    target_pos += target_vel * dt

    missile_traj.append(missile_pos.copy())
    target_traj.append(target_pos.copy())
    missile_speed.append(norm(missile_vel))

    t += dt

# -------------------------
# PLOT 3D
# -------------------------

missile_traj = np.array(missile_traj)
target_traj = np.array(target_traj)

fig = plt.figure("Simulation")
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(0, 20000)
ax.set_ylim(-10000, 10000)
ax.set_zlim(0, 35000)

ax.set_title(
    "3D Interception - Sprint ABM vs Ogive",
    fontsize=14,
    pad=20
)

# trajectories
missile_line, = ax.plot([], [], [], label="Sprint ABM", color='blue')
target_line, = ax.plot([], [], [], label="Ogive", color='red')

# vertical lines
missile_alt_line, = ax.plot([], [], [], linestyle='dashed', color='blue')
target_alt_line, = ax.plot([], [], [], linestyle='dashed', color='red')

# labels (created empty)
missile_label = ax.text2D(0, 0, "Sprint ABM", color='blue')
target_label = ax.text2D(0, 0, "Ogive", color='red')

# HUD
text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

ax.legend()

# -------------------------
#       ANIMATION
# -------------------------

def update(frame):

    missile_line.set_data(missile_traj[:frame,0], missile_traj[:frame,1])
    missile_line.set_3d_properties(missile_traj[:frame,2])

    target_line.set_data(target_traj[:frame,0], target_traj[:frame,1])
    target_line.set_3d_properties(target_traj[:frame,2])

    mx, my, mz = missile_traj[frame]
    tx, ty, tz = target_traj[frame]

    # -------------------------
    # HORIZONTAL LABELS (REAL)
    # -------------------------
    mx2, my2, _ = proj3d.proj_transform(mx, my, mz, ax.get_proj())
    tx2, ty2, _ = proj3d.proj_transform(tx, ty, tz, ax.get_proj())

    missile_label.set_position((mx2, my2))
    target_label.set_position((tx2, ty2))

    # -------------------------
    #      VERTICAL LINES
    # -------------------------
    missile_alt_line.set_data([mx, mx], [my, my])
    missile_alt_line.set_3d_properties([0, mz])

    target_alt_line.set_data([tx, tx], [ty, ty])
    target_alt_line.set_3d_properties([0, tz])

    # HUD
    v = missile_speed[frame]
    text.set_text(f"Mach {v/343:.1f}")

    return (
        missile_line,
        target_line,
        missile_label,
        target_label,
        missile_alt_line,
        target_alt_line,
        text
    )

ani = FuncAnimation(fig, update, frames=len(missile_traj), interval=dt*1000)

plt.show()

if not hit:
    print("Fail")