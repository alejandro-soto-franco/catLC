use crate::microscopic::{MicroscopicConfiguration, QTensor};
use crate::mesoscopic::QTensorField;
use crate::macroscopic::{MacroscopicConfiguration, Defect};
use crate::manifold::{CurvedSpace, CurvedSpacePoint};
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Data format for visualizing director fields
#[derive(Serialize, Deserialize)]
pub struct DirectorFieldData {
    pub positions: Vec<[f64; 3]>,
    pub directions: Vec<[f64; 3]>,
    pub order_parameters: Vec<f64>,
    pub dimensions: [f64; 3],
    pub metadata: HashMap<String, String>,
}

/// Data format for visualizing defects
#[derive(Serialize, Deserialize)]
pub struct DefectData {
    pub positions: Vec<[f64; 3]>,
    pub charges: Vec<f64>,
    pub orientations: Vec<Option<[f64; 3]>>,
    pub dimensions: [f64; 3],
    pub metadata: HashMap<String, String>,
}

/// Data format for RG flow trajectories
#[derive(Serialize, Deserialize)]
pub struct RGFlowData {
    pub parameter_names: Vec<String>,
    pub trajectory: Vec<Vec<f64>>,
    pub fixed_points: Vec<Vec<f64>>,
    pub fixed_point_types: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Convert a microscopic configuration to director field data
pub fn microscopic_to_director_field(config: &MicroscopicConfiguration) -> DirectorFieldData {
    let (nx, ny, nz) = config.dimensions;
    let mut positions = Vec::with_capacity(config.q_tensors.len());
    let mut directions = Vec::with_capacity(config.q_tensors.len());
    let mut order_parameters = Vec::with_capacity(config.q_tensors.len());
    
    // Calculate the cell size based on total system dimensions and grid points
    let dx = 1.0;
    let dy = 1.0;
    let dz = 1.0;
    
    let mut index = 0;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if index < config.q_tensors.len() {
                    // Calculate position
                    let pos = [
                        i as f64 * dx, 
                        j as f64 * dy, 
                        k as f64 * dz
                    ];
                    positions.push(pos);
                    
                    // Extract director and order parameter
                    let (s, n) = config.q_tensors[index].to_director();
                    directions.push([n[0], n[1], n[2]]);
                    order_parameters.push(s);
                    
                    index += 1;
                }
            }
        }
    }
    
    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("temperature".to_string(), config.temperature.to_string());
    metadata.insert("system_type".to_string(), "microscopic".to_string());
    
    DirectorFieldData {
        positions,
        directions,
        order_parameters,
        dimensions: [nx as f64 * dx, ny as f64 * dy, nz as f64 * dz],
        metadata,
    }
}

/// Convert a mesoscopic Q-tensor field to director field data
pub fn mesoscopic_to_director_field(field: &QTensorField, temperature: f64) -> DirectorFieldData {
    let (nx, ny, nz) = field.resolution;
    let (dx, dy, dz) = field.spacing;
    let mut positions = Vec::with_capacity(field.values.len());
    let mut directions = Vec::with_capacity(field.values.len());
    let mut order_parameters = Vec::with_capacity(field.values.len());
    
    let mut index = 0;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if let Some(q) = field.get(i, j, k) {
                    // Calculate position
                    let pos = [
                        i as f64 * dx, 
                        j as f64 * dy, 
                        k as f64 * dz
                    ];
                    positions.push(pos);
                    
                    // Extract director and order parameter
                    let (s, n) = q.to_director();
                    directions.push([n[0], n[1], n[2]]);
                    order_parameters.push(s);
                    
                    index += 1;
                }
            }
        }
    }
    
    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("temperature".to_string(), temperature.to_string());
    metadata.insert("system_type".to_string(), "mesoscopic".to_string());
    
    DirectorFieldData {
        positions,
        directions,
        order_parameters,
        dimensions: [nx as f64 * dx, ny as f64 * dy, nz as f64 * dz],
        metadata,
    }
}

/// Convert a macroscopic configuration to defect data
pub fn macroscopic_to_defect_data(config: &MacroscopicConfiguration) -> DefectData {
    let mut positions = Vec::with_capacity(config.defects.len());
    let mut charges = Vec::with_capacity(config.defects.len());
    let mut orientations = Vec::with_capacity(config.defects.len());
    
    for defect in &config.defects {
        positions.push(defect.position);
        charges.push(defect.charge);
        orientations.push(defect.orientation);
    }
    
    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("temperature".to_string(), config.temperature.to_string());
    metadata.insert("system_type".to_string(), "macroscopic".to_string());
    metadata.insert("defect_count".to_string(), config.defects.len().to_string());
    
    DefectData {
        positions,
        charges,
        orientations,
        dimensions: config.dimensions,
        metadata,
    }
}

/// Generate a visualization of LC phase on a curved surface
pub fn generate_curved_surface_data(
    space: &CurvedSpace,
    resolution: usize,
) -> Result<DirectorFieldData, Box<dyn Error>> {
    match space {
        CurvedSpace::Sphere { radius, center } => {
            // Generate points on the sphere
            let mut positions = Vec::new();
            let mut directions = Vec::new();
            let mut order_parameters = Vec::new();
            
            for i in 0..resolution {
                for j in 0..resolution {
                    // Parametrize the sphere's surface
                    let theta = std::f64::consts::PI * (i as f64) / (resolution as f64 - 1.0);
                    let phi = 2.0 * std::f64::consts::PI * (j as f64) / (resolution as f64);
                    
                    let x = center[0] + radius * theta.sin() * phi.cos();
                    let y = center[1] + radius * theta.sin() * phi.sin();
                    let z = center[2] + radius * theta.cos();
                    
                    positions.push([x, y, z]);
                    
                    // Define a tangential director field (simplified)
                    // In a real implementation, this would solve for a specific LC configuration
                    let direction = [
                        -phi.sin(), 
                        phi.cos(), 
                        0.0
                    ];
                    directions.push(direction);
                    
                    // Sample order parameter
                    order_parameters.push(0.5); // Constant for now
                }
            }
            
            // Create metadata
            let mut metadata = HashMap::new();
            metadata.insert("surface_type".to_string(), "sphere".to_string());
            metadata.insert("radius".to_string(), radius.to_string());
            
            Ok(DirectorFieldData {
                positions,
                directions,
                order_parameters,
                dimensions: [2.0 * radius, 2.0 * radius, 2.0 * radius],
                metadata,
            })
        },
        CurvedSpace::Torus { major_radius, minor_radius } => {
            // Generate points on the torus
            let mut positions = Vec::new();
            let mut directions = Vec::new();
            let mut order_parameters = Vec::new();
            
            for i in 0..resolution {
                for j in 0..resolution {
                    // Parametrize the torus
                    let theta = 2.0 * std::f64::consts::PI * (i as f64) / (resolution as f64);
                    let phi = 2.0 * std::f64::consts::PI * (j as f64) / (resolution as f64);
                    
                    let x = (major_radius + minor_radius * phi.cos()) * theta.cos();
                    let y = (major_radius + minor_radius * phi.cos()) * theta.sin();
                    let z = minor_radius * phi.sin();
                    
                    positions.push([x, y, z]);
                    
                    // Define a tangential director field (simplified)
                    // For the torus, we'll use a field that goes around the major circle
                    let direction = [
                        -y, 
                        x, 
                        0.0
                    ];
                    // Normalize
                    let norm = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
                    if norm > 0.0 {
                        directions.push([
                            direction[0] / norm,
                            direction[1] / norm,
                            direction[2] / norm
                        ]);
                    } else {
                        directions.push([1.0, 0.0, 0.0]);
                    }
                    
                    // Sample order parameter
                    order_parameters.push(0.5); // Constant for now
                }
            }
            
            // Create metadata
            let mut metadata = HashMap::new();
            metadata.insert("surface_type".to_string(), "torus".to_string());
            metadata.insert("major_radius".to_string(), major_radius.to_string());
            metadata.insert("minor_radius".to_string(), minor_radius.to_string());
            
            Ok(DirectorFieldData {
                positions,
                directions,
                order_parameters,
                dimensions: [
                    2.0 * (major_radius + minor_radius),
                    2.0 * (major_radius + minor_radius),
                    2.0 * minor_radius
                ],
                metadata,
            })
        },
        CurvedSpace::HyperbolicSpace { radius } => {
            // Generate points in the Poincaré disk model
            let mut positions = Vec::new();
            let mut directions = Vec::new();
            let mut order_parameters = Vec::new();
            
            for i in 0..resolution {
                for j in 0..resolution {
                    // Map to the unit disk
                    let u = 2.0 * (i as f64) / (resolution as f64 - 1.0) - 1.0;
                    let v = 2.0 * (j as f64) / (resolution as f64 - 1.0) - 1.0;
                    
                    // Stay within the disk
                    if u*u + v*v < 1.0 {
                        // Scale by radius
                        let x = u * radius;
                        let y = v * radius;
                        let z = 0.0; // We're visualizing the 2D disk model
                        
                        positions.push([x, y, z]);
                        
                        // Define a tangential director field (simplified)
                        // For the hyperbolic space, we'll use a radial field
                        let r = (u*u + v*v).sqrt();
                        if r > 0.0 {
                            directions.push([u/r, v/r, 0.0]);
                        } else {
                            directions.push([1.0, 0.0, 0.0]);
                        }
                        
                        // Sample order parameter - higher near boundary
                        let order = 0.3 + 0.4 * r;
                        order_parameters.push(order);
                    }
                }
            }
            
            // Create metadata
            let mut metadata = HashMap::new();
            metadata.insert("surface_type".to_string(), "hyperbolic".to_string());
            metadata.insert("radius".to_string(), radius.to_string());
            
            Ok(DirectorFieldData {
                positions,
                directions,
                order_parameters,
                dimensions: [2.0 * radius, 2.0 * radius, 0.1], // Flat disk model
                metadata,
            })
        },
    }
}

/// Save visualization data to JSON file
pub fn save_to_json<T: Serialize>(
    data: &T, 
    filename: &str
) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_string_pretty(data)?;
    let mut file = File::create(filename)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Generate RG flow data for visualization
pub fn generate_rg_flow_data<P: serde::Serialize>(
    parameter_names: Vec<String>,
    trajectories: Vec<Vec<P>>,
    fixed_points: Vec<(Vec<f64>, String)>,
) -> RGFlowData {
    let mut trajectory_data = Vec::new();
    for traj in trajectories {
        let serialized = serde_json::to_value(traj).unwrap();
        let points: Vec<Vec<f64>> = serde_json::from_value(serialized).unwrap();
        trajectory_data.extend(points);
    }
    
    let mut fixed_point_data = Vec::new();
    let mut fixed_point_types = Vec::new();
    
    for (point, type_name) in fixed_points {
        fixed_point_data.push(point);
        fixed_point_types.push(type_name);
    }
    
    let mut metadata = HashMap::new();
    metadata.insert("dimensions".to_string(), parameter_names.len().to_string());
    metadata.insert("visualization_type".to_string(), "rg_flow".to_string());
    
    RGFlowData {
        parameter_names,
        trajectory: trajectory_data,
        fixed_points: fixed_point_data,
        fixed_point_types,
        metadata,
    }
}
