
import numpy as np

x = 3.0
z = 4.0

angle_deg = 90.0
theta = np.deg2rad(angle_deg)

d_perp_squared = (
    x**2 + z**2
    - (x * np.sin(theta) + z * np.cos(theta))**2
)

d_perp = np.sqrt(d_perp_squared)

print("Perpendicular distance:", d_perp)


radius = 5.0

inside_vessel = d_perp_squared <= radius**2

print("Inside vessel:", inside_vessel)


N = 64
voxel_size = 1.0

coords = (np.arange(N) - N // 2) * voxel_size

X, Z = np.meshgrid(coords, coords, indexing="ij")


d_perp_squared = (
    X**2 + Z**2
    - (X * np.sin(theta) + Z * np.cos(theta))**2
)

vessel_mask = d_perp_squared <= radius**2



import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

plt.imshow(
    vessel_mask.T,
    origin="lower",
    extent=[
        coords[0], coords[-1],
        coords[0], coords[-1]
    ],
    cmap="gray"
)

plt.xlabel("x (mm)")
plt.ylabel("z (mm)")
plt.title(f"Vessel tilted by {angle_deg} degrees")

plt.axis("equal")
plt.show()
