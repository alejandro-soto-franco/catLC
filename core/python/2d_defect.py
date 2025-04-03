# Example guide on using Mayavi for 3D visualisations
# Example 1: Defect schematic
# Example 2: Defect in a polymer
# Example 3: Defect in a polymer with a field
# Example 4: Defect in a polymer with a field and a surface

# By Alejandro Soto Franco
# Original version by Louise Head
# First authored on 2025-04-02

################################################################################
# Import libraries
import numpy as np
import math
from mayavi import mlab
from pprint import pprint

################################################################################
# Routines

def exampledefect():
	# Example on how to make a nematic defect

	# Defect charge
	charge = -0.5

	# System size
	xyz = [5,5,1]

	# Make director field
	dir = np.zeros(shape=(3,xyz[0],xyz[1],xyz[2]))			# orientation
	pos = np.zeros(shape=(3,xyz[0],xyz[1],xyz[2]))			# position (don't need)
	for x in range(xyz[0]):
		for y in range(xyz[1]):
			phi = math.atan2((y-0.5*xyz[1]+0.5),(x-0.5*xyz[0]+0.5))		# so phi defined with defect core at origin

			# Remove director at defect core
			if x == int(0.5*xyz[0]) and y == int(0.5*xyz[1]):
				dir[0][x][y][0] = 0.
				dir[1][x][y][0] = 0.

			# Nematic orientation
			else:
				dir[0][x][y][0] = math.cos(charge*phi)
				dir[1][x][y][0] = math.sin(charge*phi)

			# Position field
			pos[0][x][y][0] = x
			pos[1][x][y][0] = y
			pos[2][x][y][0] = 0.

	# Plot director field
	directorcolour = (194./255., 211./255., 223./255.)
	DF = mlab.quiver3d(pos[0],pos[1],pos[2],dir[0],dir[1],dir[2],color=directorcolour,scale_factor = 0.7,mode='cylinder',opacity=0.8)
	# DF = mlab.quiver3d(dir[0],dir[1],dir[2],color=directorcolour,scale_factor = 0.7,mode='cylinder',opacity=0.8)
	DF.glyph.glyph_source.glyph_source.center = np.array([ 0. , 0.,  0. ])			# remove shift
	DF.actor.property.specular = 0.2
	DF.actor.property.specular_power = 10
	DF.actor.property.lighting = True
	DF.actor.property.edge_visibility = 0
	DF.actor.property.backface_culling = True

	# Plot defect core
	corecolour = (0./255.,50./255.,95./255)
	spheres = mlab.points3d([int(0.5*xyz[0])],[int(0.5*xyz[1])],[0.],[0.5],resolution=60,scale_factor=1,color=corecolour,opacity=0.8)
	spheres.glyph.glyph_source.glyph_source.center = np.array([ 0. , 0.,  0. ])		# remove shift

def disclination_loop(charge=0.5):
    """
    Create a disclination loop with specified charge.
    The directors rotate around small loops encircling the core.
    """
    # Parameters
    R = 5.0                    # loop radius
    thickness = 1.5            # visualization distance for directors around core
    N = 30                     # grid resolution
    xyz = [N, N, N]
    center = np.array([N // 2, N // 2, N // 2])

    dir = np.zeros((3, N, N, N))
    pos = np.zeros((3, N, N, N))

    # Loop parametrization - get points on the loop
    t = np.linspace(0, 2*np.pi, 100)
    x_loop = R * np.cos(t)
    y_loop = R * np.sin(t)
    z_loop = np.zeros_like(t)

    for x in range(N):
        for y in range(N):
            for z in range(N):
                X = np.array([x, y, z]) - center
                r_xy = np.sqrt(X[0]**2 + X[1]**2)
                theta = np.arctan2(X[1], X[0])
                z_offset = X[2]

                # Find the closest point on the loop
                # Convert to loop-centered cylindrical coordinates
                phi_point = np.arctan2(X[1], X[0])
                
                # Vector from loop center to current point in xy plane
                loop_center_to_point = np.array([X[0], X[1], 0])
                
                # Vector from loop center to closest point on the loop
                closest_loop_point = np.array([R * np.cos(phi_point), R * np.sin(phi_point), 0])
                
                # Vector from closest loop point to current point
                relative_pos = loop_center_to_point - closest_loop_point
                
                # Tangent vector to the loop at the closest point (counterclockwise)
                tangent = np.array([-np.sin(phi_point), np.cos(phi_point), 0])
                
                # Normal vector pointing outward from the loop
                normal = np.array([np.cos(phi_point), np.sin(phi_point), 0])
                
                # Binormal vector (perpendicular to both tangent and normal)
                binormal = np.array([0, 0, 1])
                
                # Distance from the point to the loop core
                loop_dist = np.sqrt((r_xy - R)**2 + z_offset**2)
                
                # Calculate angle around the loop core in the normal-binormal plane
                # The arctan2 accounts for the full 2π range
                local_angle = np.arctan2(np.dot(relative_pos, binormal), 
                                         np.dot(relative_pos, normal))
                
                if loop_dist < thickness:
                    # For points near the loop core
                    # Scale the influence based on distance (stronger near core)
                    influence = 1.0 - (loop_dist / thickness)**0.5
                    
                    # Director rotates around the loop as determined by charge
                    # Project the director into the normal-binormal plane
                    dir_normal = np.cos(charge * local_angle)
                    dir_binormal = np.sin(charge * local_angle)
                    
                    # Background contribution (weakens near core)
                    bg_contrib = 1.0 - influence
                    
                    # Combine the directional components in the global frame
                    dir[:, x, y, z] = (bg_contrib * np.array([1.0, 0, 0]) + 
                                      influence * (dir_normal * normal + 
                                                 dir_binormal * binormal))
                    
                    # Normalize director
                    norm = np.sqrt(np.sum(dir[:, x, y, z]**2))
                    if norm > 0:
                        dir[:, x, y, z] /= norm
                else:
                    # Background alignment
                    dir[:, x, y, z] = [1.0, 0.0, 0.0]

                # Position field
                pos[:, x, y, z] = [x, y, z]

    # Plot director field
    directorcolour = (194./255., 211./255., 223./255.)
    DF = mlab.quiver3d(pos[0], pos[1], pos[2], dir[0], dir[1], dir[2],
                       color=directorcolour, scale_factor=0.6,
                       mode='cylinder', opacity=0.7)
    DF.glyph.glyph_source.glyph_source.center = np.array([0., 0., 0.])
    DF.actor.property.specular = 0.2
    DF.actor.property.specular_power = 10
    DF.actor.property.lighting = True
    DF.actor.property.backface_culling = True

    # Plot the loop core as a circle in xy-plane
    corecolour = (0./255., 50./255., 95./255.)
    t = np.linspace(0, 2*np.pi, 100)
    x_loop = center[0] + R * np.cos(t)
    y_loop = center[1] + R * np.sin(t)
    z_loop = center[2] + np.zeros_like(t)
    mlab.plot3d(x_loop, y_loop, z_loop, color=corecolour, tube_radius=0.2, opacity=0.8)
    
    # Add text label for charge value
    mlab.text(0.02, 0.02, f"Charge = {charge}", width=0.2)

################################################################################
saveshow = 'show'
savename = 'disclination_loop.png'

# Setting up the Mayavi scene
if saveshow == 'save':
	# Needs to be before mlab.figure
	# Stops the pipeline from appearing.
	# Useful if you want to save multiple frames from a simulation
	mlab.options.offscreen = True

# Setting up the Mayavi scene
mlab.figure(size=(1024,768),bgcolor=(1.,1.,1.),fgcolor=(160./255.,160./255.,160./255.))

# Polymer example - now with charge as parameter
exampledefect()  # Try with charge=0.5 (positive) or charge=-0.5 (negative)

# Save
if saveshow == 'save':
	mlab.savefig(savename)

if saveshow == 'show':
	# Show the scene (can interact with pipeline)
	mlab.show()