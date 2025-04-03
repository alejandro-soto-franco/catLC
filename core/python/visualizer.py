import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation

class LCVisualizer:
    """Visualizer for liquid crystal configurations across different scales"""

    def __init__(self, output_dir="output"):
        """
        Initialize the visualizer
        
        Parameters
        ----------
        output_dir : str
            Directory containing the JSON output files from the Rust code
        """
        self.output_dir = output_dir
        self.fig = None
        self.ax = None

    def load_data(self, filename):
        """
        Load data from a JSON file
        
        Parameters
        ----------
        filename : str
            Path to the JSON file
            
        Returns
        -------
        dict
            The loaded data
        """
        with open(filename, 'r') as f:
            return json.load(f)

    def visualize_director_field(self, filename, subsample=1, scale=0.5, 
                                 color_by_order=True, show=True,
                                 save_path=None):
        """
        Visualize a director field from a JSON file
        
        Parameters
        ----------
        filename : str
            Path to the JSON file containing director field data
        subsample : int
            Subsample factor to reduce the number of directors shown
        scale : float
            Scale factor for the director lengths
        color_by_order : bool
            Whether to color directors by order parameter
        show : bool
            Whether to show the plot
        save_path : str
            Path to save the figure, if not None
            
        Returns
        -------
        fig, ax
            The figure and axis objects
        """
        data = self.load_data(filename)
        
        positions = np.array(data["positions"])
        directions = np.array(data["directions"])
        order_parameters = np.array(data["order_parameters"])
        dimensions = np.array(data["dimensions"])
        
        # Subsample to make visualization clearer
        positions = positions[::subsample]
        directions = directions[::subsample]
        order_parameters = order_parameters[::subsample]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        ax.set_xlim([0, dimensions[0]])
        ax.set_ylim([0, dimensions[1]])
        ax.set_zlim([0, dimensions[2]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Plot the directors as lines
        for i in range(len(positions)):
            start = positions[i]
            end = positions[i] + scale * directions[i]
            
            if color_by_order:
                # Color based on order parameter
                color = cm.plasma(order_parameters[i])
            else:
                color = 'b'
                
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    color=color, linewidth=1.5)
        
        # Add a colorbar if coloring by order parameter
        if color_by_order:
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, 
                                      norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Order Parameter')
            
        # Set title based on metadata
        if "temperature" in data["metadata"]:
            temp = data["metadata"]["temperature"]
            title = f"Director Field at T = {temp}"
            ax.set_title(title)
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        self.fig = fig
        self.ax = ax
        return fig, ax

    def visualize_defects(self, filename, show=True, save_path=None):
        """
        Visualize defects from a JSON file
        
        Parameters
        ----------
        filename : str
            Path to the JSON file containing defect data
        show : bool
            Whether to show the plot
        save_path : str
            Path to save the figure, if not None
            
        Returns
        -------
        fig, ax
            The figure and axis objects
        """
        data = self.load_data(filename)
        
        positions = np.array(data["positions"])
        charges = np.array(data["charges"])
        dimensions = np.array(data["dimensions"])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        ax.set_xlim([0, dimensions[0]])
        ax.set_ylim([0, dimensions[1]])
        ax.set_zlim([0, dimensions[2]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Plot positive defects in red, negative defects in blue
        pos_defects = positions[charges > 0]
        neg_defects = positions[charges < 0]
        
        if len(pos_defects) > 0:
            ax.scatter(pos_defects[:, 0], pos_defects[:, 1], pos_defects[:, 2], 
                       color='r', s=100, label='Positive Defects')
            
        if len(neg_defects) > 0:
            ax.scatter(neg_defects[:, 0], neg_defects[:, 1], neg_defects[:, 2], 
                       color='b', s=100, label='Negative Defects')
            
        ax.legend()
        
        # Set title based on metadata
        if "temperature" in data["metadata"]:
            temp = data["metadata"]["temperature"]
            title = f"Defect Configuration at T = {temp}"
            ax.set_title(title)
            
        if "defect_count" in data["metadata"]:
            count = data["metadata"]["defect_count"]
            plt.figtext(0.5, 0.01, f"Total defects: {count}", 
                       ha='center', fontsize=12)
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        self.fig = fig
        self.ax = ax
        return fig, ax

    def visualize_rg_flow(self, filename, dims=(0, 1), show=True, save_path=None):
        """
        Visualize RG flow data from a JSON file
        
        Parameters
        ----------
        filename : str
            Path to the JSON file containing RG flow data
        dims : tuple
            Which dimensions to plot (2D projection)
        show : bool
            Whether to show the plot
        save_path : str
            Path to save the figure, if not None
            
        Returns
        -------
        fig, ax
            The figure and axis objects
        """
        data = self.load_data(filename)
        
        param_names = data["parameter_names"]
        trajectory = np.array(data["trajectory"])
        fixed_points = np.array(data["fixed_points"])
        fixed_point_types = data["fixed_point_types"]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract 2D projection of the trajectory
        x = trajectory[:, dims[0]]
        y = trajectory[:, dims[1]]
        
        # Plot the trajectory with arrows showing flow direction
        ax.plot(x, y, 'k-', alpha=0.6, linewidth=1)
        
        # Add arrows to show flow direction
        for i in range(0, len(x)-1, max(1, len(x)//10)):
            ax.arrow(x[i], y[i], x[i+1]-x[i], y[i+1]-y[i], 
                    head_width=0.05, head_length=0.1, fc='k', ec='k')
        
        # Plot fixed points
        colors = {'stable': 'g', 'unstable': 'r', 'saddle': 'orange'}
        markers = {'stable': 'o', 'unstable': 's', 'saddle': 'd'}
        
        for i, fp_type in enumerate(fixed_point_types):
            color = colors.get(fp_type.lower(), 'b')
            marker = markers.get(fp_type.lower(), '*')
            ax.scatter(fixed_points[i, dims[0]], fixed_points[i, dims[1]], 
                      color=color, marker=marker, s=100, 
                      label=f"{fp_type} Fixed Point")
            
        ax.legend()
        ax.set_xlabel(param_names[dims[0]])
        ax.set_ylabel(param_names[dims[1]])
        ax.set_title("RG Flow in Parameter Space")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig, ax

    def visualize_curved_surface(self, filename, show=True, save_path=None):
        """
        Visualize liquid crystal on curved surfaces
        
        Parameters
        ----------
        filename : str
            Path to the JSON file containing curved surface data
        show : bool
            Whether to show the plot
        save_path : str
            Path to save the figure, if not None
            
        Returns
        -------
        fig, ax
            The figure and axis objects
        """
        data = self.load_data(filename)
        
        positions = np.array(data["positions"])
        directions = np.array(data["directions"])
        order_parameters = np.array(data["order_parameters"])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surface_type = data["metadata"].get("surface_type", "unknown")
        
        if surface_type == "sphere":
            # Create a sphere
            radius = float(data["metadata"].get("radius", 1.0))
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x = radius * np.outer(np.cos(u), np.sin(v))
            y = radius * np.outer(np.sin(u), np.sin(v))
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Plot the sphere with transparency
            ax.plot_surface(x, y, z, color='c', alpha=0.2)
            
        elif surface_type == "torus":
            # Create a torus
            major_radius = float(data["metadata"].get("major_radius", 2.0))
            minor_radius = float(data["metadata"].get("minor_radius", 0.5))
            
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, 2 * np.pi, 50)
            u, v = np.meshgrid(u, v)
            
            x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
            y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
            z = minor_radius * np.sin(v)
            
            # Plot the torus with transparency
            ax.plot_surface(x, y, z, color='c', alpha=0.2)
            
        elif surface_type == "hyperbolic":
            # For hyperbolic space, we'll just plot a disk
            radius = float(data["metadata"].get("radius", 1.0))
            
            u = np.linspace(-radius, radius, 50)
            v = np.linspace(-radius, radius, 50)
            u, v = np.meshgrid(u, v)
            
            # Only include points within the disk
            mask = u**2 + v**2 <= radius**2
            x = np.ma.masked_array(u, ~mask)
            y = np.ma.masked_array(v, ~mask)
            z = np.ma.masked_array(np.zeros_like(u), ~mask)
            
            # Plot the disk with transparency
            ax.plot_surface(x, y, z, color='c', alpha=0.2)
        
        # Plot the directors
        for i in range(len(positions)):
            start = positions[i]
            direction = directions[i]
            # Scale the direction vector to get the end point
            # Make it shorter to fit on the surface
            end = start + 0.2 * direction
            
            # Color based on order parameter
            color = cm.plasma(order_parameters[i])
                
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    color=color, linewidth=1)
        
        # Add a colorbar for the order parameter
        sm = plt.cm.ScalarMappable(cmap=cm.plasma, 
                                  norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Order Parameter')
        
        ax.set_title(f"Liquid Crystal on {surface_type.capitalize()} Surface")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        self.fig = fig
        self.ax = ax
        return fig, ax

    def create_animation(self, filename_template, frame_count, output_file,
                         visualize_func=None, **kwargs):
        """
        Create an animation from a sequence of visualization frames
        
        Parameters
        ----------
        filename_template : str
            Template for filenames with a {} placeholder for frame number
        frame_count : int
            Number of frames to animate
        output_file : str
            Output file path for the animation
        visualize_func : function
            Visualization function to use (default: visualize_director_field)
        **kwargs : dict
            Additional arguments to pass to the visualization function
            
        Returns
        -------
        anim
            The animation object
        """
        if visualize_func is None:
            visualize_func = self.visualize_director_field
            
        fig, ax = visualize_func(filename_template.format(0), show=False, **kwargs)
        
        def update(frame):
            ax.clear()
            visualize_func(filename_template.format(frame), show=False, **kwargs)
            return ax,
        
        anim = animation.FuncAnimation(fig, update, frames=frame_count, 
                                      interval=200, blit=True)
        
        anim.save(output_file, writer='pillow', fps=10)
        return anim


if __name__ == "__main__":
    # Basic usage examples
    visualizer = LCVisualizer()
    
    # Visualize a microscopic configuration
    visualizer.visualize_director_field("output/microscopic_twisted.json", 
                                      subsample=2,
                                      save_path="output/microscopic_twisted.png")
    
    # Visualize defects
    visualizer.visualize_defects("output/macroscopic_defects.json",
                               save_path="output/defects.png")
    
    # Visualize RG flow
    visualizer.visualize_rg_flow("output/meso_rg_flow.json",
                               save_path="output/rg_flow.png")
    
    # Visualize LC on curved surface
    visualizer.visualize_curved_surface("output/curved_sphere.json",
                                      save_path="output/curved_sphere.png")
