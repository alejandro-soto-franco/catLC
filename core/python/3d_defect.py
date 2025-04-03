import numpy as np
from mayavi import mlab

# Grid setup
Nx, Ny, Nz = 50, 50, 50
x = np.linspace(-1.2, 1.2, Nx)
y = np.linspace(-1.2, 1.2, Ny)
z = np.linspace(-1.2, 1.2, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Ring radius
R0 = 0.6

# Distance from ring in xy-plane
rho = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Define a vector field resembling Fig. 33
# Outside ring: inward radial
# On-axis: vertical
# Interpolate using smooth tanh/sigmoid transition

# Smooth transition functions
def smoothstep(x, width=0.1):
    return 0.5 * (1 + np.tanh(x / width))

# Blend function: near ring vs far field
f_radial = smoothstep(R0 - rho)        # 1 inside, 0 far away
f_axial = 1 - f_radial                 # 0 inside, 1 far away

# Unit vectors
n_rho_x = -X / (rho + 1e-6)
n_rho_y = -Y / (rho + 1e-6)
n_z = Z / (np.abs(Z) + 1e-6)

# Combine radial inflow with vertical up/down
U = f_radial * 0 + f_axial * n_rho_x
V = f_radial * 0 + f_axial * n_rho_y
W = f_radial * (-np.sign(Z)) + f_axial * 0  # vertical outflow

# Normalize
norm = np.sqrt(U**2 + V**2 + W**2) + 1e-6
U /= norm
V /= norm
W /= norm

# Visualize the field
mlab.figure(bgcolor=(1,1,1), size=(1000,800))
mlab.quiver3d(X, Y, Z, U, V, W, line_width=1, mode = 'cone', scale_factor=0.02, color=(0.2, 0.2, 0.8))

# Optional: draw the loop explicitly
N = 100
loop_theta = np.linspace(0, 2*np.pi, N)
loop_x = R0 * np.cos(loop_theta)
loop_y = R0 * np.sin(loop_theta)
loop_z = np.zeros_like(loop_theta)
mlab.plot3d(loop_x, loop_y, loop_z, color=(0, 0, 0), tube_radius=0.02)

mlab.outline()
mlab.title("Disclination Ring Field", size=0.5)
mlab.show()
