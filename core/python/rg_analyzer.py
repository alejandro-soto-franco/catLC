import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import fsolve
from scipy.interpolate import griddata, RegularGridInterpolator
import matplotlib.cm as cm

class RGFlowAnalyzer:
    """
    Class for analyzing renormalization group flows and finding fixed points
    """
    
    def __init__(self, parameter_names=None, beta_functions=None):
        """
        Initialize the analyzer
        
        Parameters
        ----------
        parameter_names : list of str
            Names of the parameters in the flow
        beta_functions : list of callable
            Beta functions for each parameter (if None, will be loaded from data)
        """
        self.parameter_names = parameter_names or []
        self.beta_functions = beta_functions or []
        self.trajectories = []
        self.fixed_points = []
        self.fixed_point_types = []
    
    def compute_beta_function(self, params):
        """
        Compute the beta function for the given parameters
        
        Parameters
        ----------
        params : array_like
            Parameter values
            
        Returns
        -------
        array_like
            Beta function values
        """
        return np.array([beta(params) for beta in self.beta_functions])
    
    def find_fixed_points(self, initial_guesses):
        """
        Find fixed points of the RG flow
        
        Parameters
        ----------
        initial_guesses : list of array_like
            Initial guesses for fixed points
            
        Returns
        -------
        list of array_like
            Fixed points
        """
        fixed_points = []
        for guess in initial_guesses:
            fp = fsolve(self.compute_beta_function, guess)
            fixed_points.append(fp)
        self.fixed_points = fixed_points
        return fixed_points
    
    def classify_fixed_points(self):
        """
        Classify fixed points as stable, unstable, or saddle
        
        Returns
        -------
        list of str
            Fixed point types
        """
        types = []
        for fp in self.fixed_points:
            stability_matrix = self.compute_stability_matrix(fp)
            eigenvalues = np.linalg.eigvals(stability_matrix)
            if all(e.real < 0 for e in eigenvalues):
                types.append("stable")
            elif all(e.real > 0 for e in eigenvalues):
                types.append("unstable")
            else:
                types.append("saddle")
        self.fixed_point_types = types
        return types
    
    def generate_phase_diagram(self, dims=(0, 1), grid_points=20, 
                             xlim=None, ylim=None, show=True, save_path=None):
        """
        Generate a phase diagram showing the RG flow in a 2D plane
        
        Parameters
        ----------
        dims : tuple
            Which two parameters to use for the 2D visualization
        grid_points : int
            Number of grid points in each dimension
        xlim, ylim : tuple
            Limits for the plot
        show : bool
            Whether to show the plot
        save_path : str
            Path to save the figure, if not None
            
        Returns
        -------
        fig, ax
            Figure and axis objects
        """
        if not self.beta_functions:
            raise ValueError("Beta functions not defined or fitted yet")

        # Determine the plot limits
        if xlim is None:
            if len(self.trajectories) > 0:
                xmin = np.min(self.trajectories[:, dims[0]])
                xmax = np.max(self.trajectories[:, dims[0]])
                xrange = xmax - xmin
                xlim = (xmin - 0.2 * xrange, xmax + 0.2 * xrange)
            else:
                xlim = (-1.0, 1.0)
                
        if ylim is None:
            if len(self.trajectories) > 0:
                ymin = np.min(self.trajectories[:, dims[1]])
                ymax = np.max(self.trajectories[:, dims[1]])
                yrange = ymax - ymin
                ylim = (ymin - 0.2 * yrange, ymax + 0.2 * yrange)
            else:
                ylim = (-1.0, 1.0)
                
        # Create a grid of points in parameter space
        x = np.linspace(xlim[0], xlim[1], grid_points)
        y = np.linspace(ylim[0], ylim[1], grid_points)
        X, Y = np.meshgrid(x, y)
        
        # Compute beta function on the grid
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(grid_points):
            for j in range(grid_points):
                # Create full parameter vector (use zeros for other dimensions)
                params = np.zeros(len(self.parameter_names))
                params[dims[0]] = X[i, j]
                params[dims[1]] = Y[i, j]
                
                # Compute beta function
                beta = self.compute_beta_function(params)
                U[i, j] = beta[dims[0]]
                V[i, j] = beta[dims[1]]
                
        # Normalize for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        U = U / (magnitude + 1e-10)
        V = V / (magnitude + 1e-10)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot streamlines
        ax.streamplot(X, Y, U, V, color='gray', density=1.5, linewidth=1, arrowsize=1.5)
        
        # Plot trajectories
        if len(self.trajectories) > 0:
            ax.plot(self.trajectories[:, dims[0]], self.trajectories[:, dims[1]], 
                   'b-', alpha=0.7, linewidth=2, label='RG Trajectory')
            
            # Add arrows to show flow direction
            for i in range(0, len(self.trajectories)-1, max(1, len(self.trajectories)//10)):
                ax.arrow(
                    self.trajectories[i, dims[0]], 
                    self.trajectories[i, dims[1]],
                    self.trajectories[i+1, dims[0]] - self.trajectories[i, dims[0]], 
                    self.trajectories[i+1, dims[1]] - self.trajectories[i, dims[1]],
                    head_width=0.03 * (xlim[1]-xlim[0]), 
                    head_length=0.05 * (xlim[1]-xlim[0]), 
                    fc='blue', ec='blue', alpha=0.7
                )
        
        # Plot fixed points
        colors = {'stable': 'g', 'unstable': 'r', 'saddle': 'orange'}
        markers = {'stable': 'o', 'unstable': 's', 'saddle': 'd'}
        
        for i, fp in enumerate(self.fixed_points):
            if i < len(self.fixed_point_types):
                fp_type = self.fixed_point_types[i]
                color = colors.get(fp_type.lower(), 'b')
                marker = markers.get(fp_type.lower(), '*')
            else:
                color = 'b'
                marker = '*'
                fp_type = "unknown"
                
            ax.scatter(fp[dims[0]], fp[dims[1]], color=color, marker=marker, s=100,
                      label=f"{fp_type} Fixed Point" if i == 0 else "")
            
            # Label the fixed point
            ax.annotate(f"FP{i+1}", 
                       (fp[dims[0]], fp[dims[1]]),
                       textcoords="offset points",
                       xytext=(10, 10),
                       fontsize=12)
        
        # Mark the Gaussian fixed point (origin) if within range
        if xlim[0] <= 0 <= xlim[1] and ylim[0] <= 0 <= ylim[1]:
            ax.scatter(0, 0, color='m', marker='o', s=80, label="Gaussian FP")
            ax.annotate("G", (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=12)
            
        # Set up the plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(self.parameter_names[dims[0]], fontsize=14)
        ax.set_ylabel(self.parameter_names[dims[1]], fontsize=14)
        ax.set_title("RG Flow Phase Diagram", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig, ax
    
    def compute_stability_matrix(self, fixed_point):
        """
        Compute the stability matrix (Jacobian of beta function) at a fixed point
        
        Parameters
        ----------
        fixed_point : array_like
            Fixed point coordinates
            
        Returns
        -------
        array_like
            Stability matrix
        """
        n = len(fixed_point)
        h = 1e-5  # Step size for numerical differentiation
        stability_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Compute partial derivatives using central difference
                point_plus = fixed_point.copy()
                point_minus = fixed_point.copy()
                
                point_plus[j] += h
                point_minus[j] -= h
                
                beta_plus = self.compute_beta_function(point_plus)[i]
                beta_minus = self.compute_beta_function(point_minus)[i]
                
                stability_matrix[i, j] = (beta_plus - beta_minus) / (2 * h)
                
        return stability_matrix
    
    def compute_critical_exponents(self, fixed_point):
        """
        Compute critical exponents from the stability matrix at a fixed point
        
        Parameters
        ----------
        fixed_point : array_like
            Fixed point coordinates
            
        Returns
        -------
        array_like
            Critical exponents (negated eigenvalues of stability matrix)
        """
        stability_matrix = self.compute_stability_matrix(fixed_point)
        eigenvalues = np.linalg.eigvals(stability_matrix)
        
        # Critical exponents are negatives of the eigenvalues
        critical_exponents = -eigenvalues.real
        
        return critical_exponents, eigenvalues
    
    def generate_3d_flow(self, dims=(0, 1, 2), grid_points=10, save_path=None):
        """
        Generate a 3D visualization of the RG flow
        
        Parameters
        ----------
        dims : tuple
            Which three parameters to use for the 3D visualization
        grid_points : int
            Number of grid points in each dimension
        save_path : str
            Path to save the figure, if not None
            
        Returns
        -------
        fig, ax
            Figure and axis objects
        """
        if not self.beta_functions:
            raise ValueError("Beta functions not defined or fitted yet")
        
        # Create a figure for 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories in 3D
        if len(self.trajectories) > 0:
            ax.plot(self.trajectories[:, dims[0]], 
                   self.trajectories[:, dims[1]], 
                   self.trajectories[:, dims[2]], 
                   'b-', alpha=0.7, linewidth=2, label='RG Trajectory')
            
        # Plot fixed points in 3D
        colors = {'stable': 'g', 'unstable': 'r', 'saddle': 'orange'}
        markers = {'stable': 'o', 'unstable': 's', 'saddle': 'd'}
        
        for i, fp in enumerate(self.fixed_points):
            if i < len(self.fixed_point_types):
                fp_type = self.fixed_point_types[i]
                color = colors.get(fp_type.lower(), 'b')
                marker = markers.get(fp_type.lower(), '*')
            else:
                color = 'b'
                marker = '*'
                fp_type = "unknown"
                
            ax.scatter(fp[dims[0]], fp[dims[1]], fp[dims[2]], 
                      color=color, marker=marker, s=100,
                      label=f"{fp_type} Fixed Point" if i == 0 else "")
            
            # Label the fixed point
            ax.text(fp[dims[0]], fp[dims[1]], fp[dims[2]], f"FP{i+1}", 
                   fontsize=12)
        
        # Gaussian fixed point
        ax.scatter(0, 0, 0, color='m', marker='o', s=80, label="Gaussian FP")
        ax.text(0, 0, 0, "G", fontsize=12)
            
        # Set up the plot
        ax.set_xlabel(self.parameter_names[dims[0]], fontsize=14)
        ax.set_ylabel(self.parameter_names[dims[1]], fontsize=14)
        ax.set_zlabel(self.parameter_names[dims[2]], fontsize=14)
        ax.set_title("3D RG Flow Diagram", fontsize=16)
        ax.legend(loc='best', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig, ax
    
    def save_analysis(self, filename):
        """
        Save the RG flow analysis results to a JSON file
        
        Parameters
        ----------
        filename : str
            File path to save to
        """
        data = {
            "parameter_names": self.parameter_names,
            "trajectories": [t.tolist() for t in self.trajectories] if isinstance(self.trajectories, list) else self.trajectories.tolist(),
            "fixed_points": [fp.tolist() for fp in self.fixed_points],
            "fixed_point_types": self.fixed_point_types,
        }
        
        # Add critical exponents if they exist
        if hasattr(self, 'critical_exponents'):
            data["critical_exponents"] = [ce.tolist() for ce in self.critical_exponents]
        
        # Save to file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Analysis saved to {filename}")
        
    @classmethod
    def from_file(cls, filename):
        """
        Load RG flow analysis from a JSON file
        
        Parameters
        ----------
        filename : str
            File path to load from
            
        Returns
        -------
        RGFlowAnalyzer
            Loaded analyzer
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            
        analyzer = cls(parameter_names=data["parameter_names"])
        analyzer.trajectories = np.array(data["trajectories"])
        analyzer.fixed_points = np.array(data["fixed_points"])
        analyzer.fixed_point_types = data["fixed_point_types"]
        
        if "critical_exponents" in data:
            analyzer.critical_exponents = [np.array(ce) for ce in data["critical_exponents"]]
            
        return analyzer


if __name__ == "__main__":
    # Example usage
    analyzer = RGFlowAnalyzer(parameter_names=["a", "b", "c"])
    
    # Define a simple beta function for the parameters
    # This corresponds to an Ising-like model with cubic coupling
    def beta_a(params):
        return -params[0] + 0.1 * params[1]**2
    
    def beta_b(params):
        return 0.5 * params[1] + 0.2 * params[0] * params[1]
    
    def beta_c(params):
        return params[2] - 0.3 * params[2]**3
    
    analyzer.beta_functions = [beta_a, beta_b, beta_c]
    
    # Find fixed points
    initial_guesses = [
        np.array([0.0, 0.0, 0.0]),  # Gaussian fixed point
        np.array([1.0, 1.0, 1.0]),  # Another potential fixed point
    ]
    
    fixed_points = analyzer.find_fixed_points(initial_guesses)
    print(f"Found {len(fixed_points)} fixed points:")
    for fp in fixed_points:
        print(f"  {fp}")
    
    # Classify fixed points
    types = analyzer.classify_fixed_points()
    print("Fixed point classifications:", types)
    
    # Generate a phase diagram
    analyzer.generate_phase_diagram(dims=(0, 1), save_path="output/rg_phase_diagram.png")
    
    # Generate a 3D flow diagram
    analyzer.generate_3d_flow(dims=(0, 1, 2), save_path="output/rg_3d_flow.png")