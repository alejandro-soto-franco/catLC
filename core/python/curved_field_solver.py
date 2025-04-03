import numpy as np
from scipy.optimize import minimize
import json
import os

class CurvedFieldSolver:
    """
    Solver for liquid crystal configurations on curved surfaces.
    Uses energy minimization to find equilibrium configurations.
    """
    
    def __init__(self, surface_type="sphere", resolution=20):
        """
        Initialize the solver
        
        Parameters
        ----------
        surface_type : str
            Type of surface ("sphere", "torus", "hyperbolic")
        resolution : int
            Resolution of the director field grid
        """
        self.surface_type = surface_type
        self.resolution = resolution
        
        # Initialize parameters specific to each surface
        if surface_type == "sphere":
            self.params = {
                "radius": 1.0,
                "center": [0.0, 0.0, 0.0],
                "K1": 1.0,  # Splay elastic constant
                "K3": 1.0,  # Bend elastic constant
                "temperature": 300.0
            }
        elif surface_type == "torus":
            self.params = {
                "major_radius": 2.0,
                "minor_radius": 0.5,
                "K1": 1.0,  # Splay elastic constant
                "K3": 1.0,  # Bend elastic constant
                "temperature": 300.0
            }
        elif surface_type == "hyperbolic":
            self.params = {
                "radius": 1.0,
                "K1": 1.0,  # Splay elastic constant
                "K3": 1.0,  # Bend elastic constant
                "temperature": 300.0
            }
        else:
            raise ValueError(f"Unsupported surface type: {surface_type}")
            
        # Initialize the points on the surface
        self.initialize_surface_points()
        
        # Initialize director field
        self.initialize_director_field()
        
    def initialize_surface_points(self):
        """Initialize points on the surface based on surface type"""
        if self.surface_type == "sphere":
            # Use spherical coordinates to generate points on the sphere
            radius = self.params["radius"]
            theta = np.linspace(0, np.pi, self.resolution)
            phi = np.linspace(0, 2 * np.pi, self.resolution)
            
            self.points = []
            for t in theta:
                for p in phi:
                    x = radius * np.sin(t) * np.cos(p)
                    y = radius * np.sin(t) * np.sin(p)
                    z = radius * np.cos(t)
                    self.points.append([x, y, z])
                    
            self.points = np.array(self.points)
            
        elif self.surface_type == "torus":
            # Use toroidal coordinates to generate points on the torus
            major_radius = self.params["major_radius"]
            minor_radius = self.params["minor_radius"]
            
            theta = np.linspace(0, 2 * np.pi, self.resolution)
            phi = np.linspace(0, 2 * np.pi, self.resolution)
            
            self.points = []
            for t in theta:
                for p in phi:
                    x = (major_radius + minor_radius * np.cos(p)) * np.cos(t)
                    y = (major_radius + minor_radius * np.cos(p)) * np.sin(t)
                    z = minor_radius * np.sin(p)
                    self.points.append([x, y, z])
                    
            self.points = np.array(self.points)
            
        elif self.surface_type == "hyperbolic":
            # Use the Poincaré disk model for the hyperbolic plane
            radius = self.params["radius"]
            
            # Generate points in a disk
            r = np.linspace(0, radius * 0.99, self.resolution // 2)  # Avoid boundary
            theta = np.linspace(0, 2 * np.pi, self.resolution)
            
            self.points = []
            for rad in r:
                for t in theta:
                    x = rad * np.cos(t)
                    y = rad * np.sin(t)
                    z = 0.0  # Flat disk
                    self.points.append([x, y, z])
                    
            self.points = np.array(self.points)
    
    def get_tangent_space(self, point):
        """
        Get two orthogonal tangent vectors at a given point on the surface
        
        Parameters
        ----------
        point : array_like
            A point on the surface
            
        Returns
        -------
        array_like
            Two orthogonal tangent vectors
        """
        if self.surface_type == "sphere":
            # For a sphere, tangent vectors are orthogonal to the radial direction
            x, y, z = point
            r = np.sqrt(x**2 + y**2 + z**2)
            
            # Radial direction (normal to the surface)
            normal = np.array([x/r, y/r, z/r])
            
            # First tangent vector: cross product of normal with [0,0,1] (or [0,1,0] if normal is [0,0,1])
            if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
                t1 = np.cross(normal, [0, 1, 0])
            else:
                t1 = np.cross(normal, [0, 0, 1])
                
            t1 = t1 / np.linalg.norm(t1)
            
            # Second tangent vector: cross product of normal and t1
            t2 = np.cross(normal, t1)
            t2 = t2 / np.linalg.norm(t2)
            
            return t1, t2
            
        elif self.surface_type == "torus":
            # For a torus, tangent vectors depend on position
            x, y, z = point
            major_radius = self.params["major_radius"]
            minor_radius = self.params["minor_radius"]
            
            # Calculate the center of the circular cross-section
            dist_from_origin = np.sqrt(x**2 + y**2)
            if dist_from_origin < 1e-10:
                # Avoid division by zero
                center_x, center_y = major_radius, 0.0
            else:
                center_x = major_radius * x / dist_from_origin
                center_y = major_radius * y / dist_from_origin
                
            # Vector from center of cross-section to the point
            radial = np.array([x - center_x, y - center_y, z])
            radial = radial / np.linalg.norm(radial)
            
            # Vector along the major circle
            tangent_major = np.array([-y, x, 0.0])
            tangent_major = tangent_major / np.linalg.norm(tangent_major)
            
            # Normalize the vectors
            t1 = tangent_major
            t2 = np.cross(radial, tangent_major)
            t2 = t2 / np.linalg.norm(t2)
            
            return t1, t2
            
        elif self.surface_type == "hyperbolic":
            # For the Poincaré disk, tangent vectors are the usual Euclidean ones
            # But the metric is scaled by the conformal factor
            x, y, z = point
            radius = self.params["radius"]
            
            # The tangent space is spanned by the unit vectors in the x and y directions
            t1 = np.array([1.0, 0.0, 0.0])
            t2 = np.array([0.0, 1.0, 0.0])
            
            return t1, t2
    
    def initialize_director_field(self):
        """Initialize the director field with a default configuration"""
        self.directors = []
        self.order_parameters = []
        
        for point in self.points:
            # Get tangent space at this point
            t1, t2 = self.get_tangent_space(point)
            
            # Initialize with a simple pattern
            if self.surface_type == "sphere":
                # Use a vortex pattern around the z-axis
                x, y, z = point
                angle = np.arctan2(y, x)
                director = np.cos(angle) * t1 + np.sin(angle) * t2
                
            elif self.surface_type == "torus":
                # Use a pattern along the major circle
                director = t1  # t1 points along the major circle
                
            elif self.surface_type == "hyperbolic":
                # Use a radial pattern
                x, y, z = point
                r = np.sqrt(x**2 + y**2)
                if r > 1e-10:
                    director = (x/r) * t1 + (y/r) * t2
                else:
                    director = t1
            
            # Normalize
            director = director / np.linalg.norm(director)
            self.directors.append(director)
            
            # Initially constant order parameter
            self.order_parameters.append(0.5)
            
        self.directors = np.array(self.directors)
        self.order_parameters = np.array(self.order_parameters)
    
    def calculate_energy(self, directors=None):
        """
        Calculate the Frank free energy for the current director configuration
        
        Parameters
        ----------
        directors : array_like, optional
            Director field to use (if None, use self.directors)
            
        Returns
        -------
        float
            The free energy
        """
        if directors is None:
            directors = self.directors
        else:
            directors = directors.reshape(-1, 3)
            
        # Simple model: calculate the splay and bend energies
        # This is a very simplified model
        K1 = self.params["K1"]  # splay
        K3 = self.params["K3"]  # bend
        
        energy = 0.0
        
        # Calculate energy by comparing neighboring directors
        # This is a crude approximation
        for i, point1 in enumerate(self.points):
            for j, point2 in enumerate(self.points):
                if i >= j:
                    continue
                    
                # Only consider neighbors
                dist = np.linalg.norm(point1 - point2)
                if dist > 0.5:  # Arbitrary threshold
                    continue
                    
                n1 = directors[i]
                n2 = directors[j]
                
                # Project n2 onto the tangent plane at point1
                t1, t2 = self.get_tangent_space(point1)
                n2_proj = (n2.dot(t1)) * t1 + (n2.dot(t2)) * t2
                n2_proj = n2_proj / np.linalg.norm(n2_proj)
                
                # Calculate the difference
                diff = n2_proj - n1
                
                # Simplified energy calculation
                energy += 0.5 * K1 * (np.linalg.norm(diff) / dist)**2
                
        return energy
    
    def optimize_director_field(self, iterations=10, method='BFGS'):
        """
        Optimize the director field to minimize the free energy
        
        Parameters
        ----------
        iterations : int
            Number of optimization iterations
        method : str
            Optimization method for scipy.optimize.minimize
            
        Returns
        -------
        tuple
            (optimized directors, optimized order parameters, energy)
        """
        def energy_function(flat_directors):
            directors = flat_directors.reshape(-1, 3)
            
            # Ensure directors are unit vectors
            norms = np.linalg.norm(directors, axis=1)
            directors = directors / norms[:, np.newaxis]
            
            return self.calculate_energy(directors)
        
        # Flatten the directors for optimization
        flat_directors = self.directors.flatten()
        
        # Optimize
        for _ in range(iterations):
            result = minimize(
                energy_function, 
                flat_directors,
                method=method,
                options={'maxiter': 5}  # Few iterations per outer loop
            )
            flat_directors = result.x
        
        # Reshape and normalize
        optimized_directors = flat_directors.reshape(-1, 3)
        norms = np.linalg.norm(optimized_directors, axis=1)
        optimized_directors = optimized_directors / norms[:, np.newaxis]
        
        # For simplicity, keep the order parameters unchanged
        # In a more sophisticated implementation, they would also be optimized
        
        self.directors = optimized_directors
        energy = self.calculate_energy()
        
        print(f"Optimization complete. Final energy: {energy}")
        
        return self.directors, self.order_parameters, energy
    
    def project_to_tangent_space(self):
        """Project directors onto tangent spaces to ensure they're tangent to the surface"""
        for i, point in enumerate(self.points):
            t1, t2 = self.get_tangent_space(point)
            
            # Project the director onto the tangent plane
            director = self.directors[i]
            projected = (director.dot(t1)) * t1 + (director.dot(t2)) * t2
            
            # Normalize
            norm = np.linalg.norm(projected)
            if norm > 1e-10:
                projected = projected / norm
                
            self.directors[i] = projected
    
    def export_to_json(self, filename):
        """
        Export the director field to JSON for visualization
        
        Parameters
        ----------
        filename : str
            Output file path
        """
        # Ensure the directors are tangent to the surface
        self.project_to_tangent_space()
        
        data = {
            "positions": self.points.tolist(),
            "directions": self.directors.tolist(),
            "order_parameters": self.order_parameters.tolist(),
            "dimensions": [
                2 * self.params.get("radius", 1.0) if self.surface_type in ["sphere", "hyperbolic"] else 
                2 * (self.params["major_radius"] + self.params["minor_radius"]),
                
                2 * self.params.get("radius", 1.0) if self.surface_type in ["sphere", "hyperbolic"] else 
                2 * (self.params["major_radius"] + self.params["minor_radius"]),
                
                2 * self.params.get("radius", 1.0) if self.surface_type == "sphere" else 
                2 * self.params["minor_radius"] if self.surface_type == "torus" else 
                0.1  # Thin for hyperbolic
            ],
            "metadata": {
                "surface_type": self.surface_type,
                "temperature": str(self.params["temperature"])
            }
        }
        
        # Add surface-specific metadata
        if self.surface_type == "sphere":
            data["metadata"]["radius"] = str(self.params["radius"])
        elif self.surface_type == "torus":
            data["metadata"]["major_radius"] = str(self.params["major_radius"])
            data["metadata"]["minor_radius"] = str(self.params["minor_radius"])
        elif self.surface_type == "hyperbolic":
            data["metadata"]["radius"] = str(self.params["radius"])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Exported director field to {filename}")


if __name__ == "__main__":
    # Example usage
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Solve and export director field on a sphere
    sphere_solver = CurvedFieldSolver(surface_type="sphere", resolution=20)
    sphere_solver.optimize_director_field(iterations=5)
    sphere_solver.export_to_json(f"{output_dir}/optimal_sphere.json")
    
    # Solve and export director field on a torus
    torus_solver = CurvedFieldSolver(surface_type="torus", resolution=20)
    torus_solver.optimize_director_field(iterations=5)
    torus_solver.export_to_json(f"{output_dir}/optimal_torus.json")
    
    # Solve and export director field on a hyperbolic space
    hyperbolic_solver = CurvedFieldSolver(surface_type="hyperbolic", resolution=20)
    hyperbolic_solver.optimize_director_field(iterations=5)
    hyperbolic_solver.export_to_json(f"{output_dir}/optimal_hyperbolic.json")
