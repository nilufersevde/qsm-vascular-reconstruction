
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# 1. Generate the cone coordinates
radius = np.linspace(0, 1, 100)
phi = np.linspace(0, 2 * np.pi, 200)

R, PHI = np.meshgrid(radius, phi)

kx = R * np.cos(PHI)
ky = R * np.sin(PHI)

kz_upper = R / np.sqrt(2)
kz_lower = -R / np.sqrt(2)


# 2. Create the 3D figure
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

# Both halves use the SAME color
cone_color = "steelblue"

ax.plot_surface(
    kx, ky, kz_upper,
    color=cone_color,
    alpha=0.50,
    linewidth=0,
    antialiased=True
)

ax.plot_surface(
    kx, ky, kz_lower,
    color=cone_color,
    alpha=0.50,
    linewidth=0,
    antialiased=True
)


# 3. Mark the B0 direction
# B0 is assumed to point along the kz axis

ax.quiver(
    0, 0, 0,      # start exactly at apex
    0, 0, 0.95,   # direction along +kz
    color="black",
    linewidth=2.5,
    arrow_length_ratio=0.10
)

ax.text(
    0, 0, 1.0,
    "$B_0$",
    fontsize=19,
    color="black"
)


# 4. Draw the magic-angle arc
theta = np.deg2rad(54.7356)

# Angles from the positive kz axis
arc_angles = np.linspace(0, theta, 60)
arc_radius = 0.38

arc_x = arc_radius * np.sin(arc_angles)
arc_y = np.zeros_like(arc_angles)
arc_z = arc_radius * np.cos(arc_angles)

ax.plot(
    arc_x, arc_y, arc_z,
    color="darkred",
    linewidth=3
)

# Label the angle
label_angle = theta / 2

ax.text2D(
    0.60, 0.63,
    "54.7°",
    transform=ax.transAxes,
    fontsize=17,
    color="darkred",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2)
)



# 5. Format axes for presentation
ax.set_xlabel("$k_x$", fontsize=18, labelpad=12)
ax.set_ylabel("$k_y$", fontsize=18, labelpad=12)
ax.set_zlabel("$k_z$", fontsize=18, labelpad=12)

ticks = [-1, -0.5, 0, 0.5, 1]

ax.set_xticks([-1, -0.5, 0, 0.5])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([-1, -0.5, 0, 0.5, 1])

ax.tick_params(labelsize=13, pad=5)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1.1)

ax.set_box_aspect((1, 1, 1.05))

# Reduce background clutter
ax.grid(False)

# Choose viewing angle
ax.view_init(elev=22, azim=45)

# No figure title — the slide has its own heading
plt.tight_layout()


# 6. Save the figure
ROOT = Path(__file__).resolve().parents[2]
output_dir = ROOT / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_dir / "magic_angle_cone_3d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
