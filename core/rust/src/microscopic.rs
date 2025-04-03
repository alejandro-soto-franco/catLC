use crate::category::{Category, CategoryError, FinCategory, Morphism, Object};
use crate::functor::{Functor, ConcreteFunctor};
use crate::rg_flow::{ParameterSpace, RGFlowError};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use rand::Rng;
use std::f64::consts::PI;
use thiserror::Error;

/// Error types related to microscopic models
#[derive(Error, Debug)]
pub enum MicroscopicError {
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Q-tensor construction error: {0}")]
    QTensorConstructionError(String),
    
    #[error("Parameter out of range: {0}")]
    ParameterOutOfRange(String),
}

/// Q-tensor representation (symmetric traceless 3x3 matrix)
#[derive(Clone, Debug, PartialEq)]
pub struct QTensor {
    /// The components of this Q-tensor as a 3x3 matrix
    pub components: DMatrix<f64>,
}

impl QTensor {
    /// Create a new Q-tensor from a 3x3 matrix
    pub fn new(components: DMatrix<f64>) -> Self {
        // In a real implementation, we would ensure the matrix is
        // symmetric and traceless here
        Self { components }
    }
    
    /// Create a Q-tensor from a director and scalar order parameter
    pub fn from_director(director: &Vector3<f64>, scalar_order: f64) -> Result<Self, MicroscopicError> {
        if (director.norm() - 1.0).abs() > 1e-6 {
            return Err(MicroscopicError::QTensorConstructionError(
                "Director must be a unit vector".to_string()
            ));
        }
        
        let n = director;
        let identity = Matrix3::identity();
        
        // Q_ij = S (n_i n_j - δ_ij/3)
        let outer_product = n * n.transpose();
        let components = scalar_order * (outer_product - identity / 3.0);
        
        Ok(Self { components })
    }
    
    /// Convert this Q-tensor to a director and scalar order parameter
    pub fn to_director(&self) -> (f64, Vector3<f64>) {
        // Compute the eigendecomposition of the Q-tensor
        let eigen = self.components.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;
        
        // The director is the eigenvector corresponding to the largest eigenvalue
        let max_idx = eigenvalues.argmax();
        let director = Vector3::new(
            eigenvectors[(0, max_idx)],
            eigenvectors[(1, max_idx)],
            eigenvectors[(2, max_idx)]
        );
        
        // The scalar order parameter is related to the largest eigenvalue
        let scalar_order = eigenvalues[max_idx] * 3.0 / 2.0;
        
        (scalar_order, director)
    }
}

/// Microscopic configuration of a liquid crystal
#[derive(Clone, Debug, PartialEq)]
pub struct MicroscopicConfiguration {
    /// Dimensions of the lattice (nx, ny, nz)
    pub dimensions: (usize, usize, usize),
    
    /// Q-tensors at each lattice site
    pub q_tensors: Vec<QTensor>,
    
    /// Temperature of the system
    pub temperature: f64,
    
    /// External field (if any)
    pub external_field: Option<Vector3<f64>>,
}

impl Object for MicroscopicConfiguration {
    fn id(&self) -> String {
        let (nx, ny, nz) = self.dimensions;
        format!("MicroConfig_{}x{}x{}_T{:.2}", nx, ny, nz, self.temperature)
    }
    
    fn dimension(&self) -> Option<usize> {
        let (nx, ny, nz) = self.dimensions;
        Some(nx * ny * nz * 5) // 5 degrees of freedom per Q-tensor
    }
}

/// Parameters for the microscopic Maier-Saupe model
#[derive(Clone, Debug)]
pub struct MicroscopicParameters {
    /// Bulk free energy parameters
    pub a: f64,  // Temperature-dependent parameter
    pub b: f64,  // Cubic term coefficient
    pub c: f64,  // Quartic term coefficient
    
    /// Elastic constants
    pub l1: f64, // Splay
    pub l2: f64, // Twist
    pub l3: f64, // Bend
    
    /// External field coupling
    pub h: f64,
    
    /// Temperature
    pub temperature: f64,
}

impl ParameterSpace for MicroscopicParameters {
    fn dimension(&self) -> usize {
        7 // a, b, c, l1, l2, l3, h
    }
    
    fn as_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.a, self.b, self.c, 
            self.l1, self.l2, self.l3, self.h
        ])
    }
    
    fn from_vector(vec: DVector<f64>) -> Result<Self, RGFlowError> {
        if vec.len() != 7 {
            return Err(RGFlowError::ParameterOutOfRange(
                format!("Expected 7 parameters, got {}", vec.len())
            ));
        }
        
        Ok(Self {
            a: vec[0],
            b: vec[1],
            c: vec[2],
            l1: vec[3],
            l2: vec[4],
            l3: vec[5],
            h: vec[6],
            temperature: 300.0, // Default value, not stored in vector
        })
    }
    
    fn distance(&self, other: &Self) -> f64 {
        (self.as_vector() - other.as_vector()).norm()
    }
}

/// A morphism between microscopic configurations
#[derive(Clone, Debug)]
pub struct MicroscopicMorphism {
    /// Domain of this morphism
    pub domain: MicroscopicConfiguration,
    
    /// Codomain of this morphism
    pub codomain: MicroscopicConfiguration,
    
    /// Type of transformation
    pub transformation_type: String,
    
    /// Parameters of the transformation
    pub parameters: Option<DVector<f64>>,
}

impl Morphism for MicroscopicMorphism {
    type ObjectType = MicroscopicConfiguration;
    
    fn domain(&self) -> &Self::ObjectType {
        &self.domain
    }
    
    fn codomain(&self) -> &Self::ObjectType {
        &self.codomain
    }
    
    fn apply<T>(&self, data: &T) -> Result<T, CategoryError> 
    where T: Clone + Debug {
        // In a real implementation, this would transform the data
        // For now, just return a clone
        Ok(data.clone())
    }
}

/// Generate a microscopic configuration with a specified pattern
pub fn generate_microscopic_configuration(
    nx: usize, ny: usize, nz: usize, 
    pattern: &str, 
    temperature: f64
) -> MicroscopicConfiguration {
    let total_sites = nx * ny * nz;
    let mut q_tensors = Vec::with_capacity(total_sites);
    let mut rng = rand::thread_rng();
    
    // Generate Q-tensors based on the pattern
    match pattern {
        "uniform" => {
            // Uniform director along z-axis
            let director = Vector3::new(0.0, 0.0, 1.0);
            let q = QTensor::from_director(&director, 0.6).unwrap();
            for _ in 0..total_sites {
                q_tensors.push(q.clone());
            }
        },
        "twisted" => {
            // Twisted configuration with director rotating around z-axis
            for i in 0..nx {
                let angle = 2.0 * PI * (i as f64) / (nx as f64);
                let director = Vector3::new(angle.cos(), angle.sin(), 0.0);
                let q = QTensor::from_director(&director, 0.6).unwrap();
                
                for _ in 0..(ny * nz) {
                    q_tensors.push(q.clone());
                }
            }
        },
        "defect" => {
            // Configuration with a +1 defect at the center
            let center_x = nx / 2;
            let center_y = ny / 2;
            
            for i in 0..nx {
                for j in 0..ny {
                    // Calculate angle relative to the center
                    let dx = (i as f64) - (center_x as f64);
                    let dy = (j as f64) - (center_y as f64);
                    let angle = dy.atan2(dx);
                    
                    let director = Vector3::new(angle.cos(), angle.sin(), 0.0);
                    let q = QTensor::from_director(&director, 0.6).unwrap();
                    
                    for _ in 0..nz {
                        q_tensors.push(q.clone());
                    }
                }
            }
        },
        "random" => {
            // Random director orientations
            for _ in 0..total_sites {
                let phi = rng.gen_range(0.0..2.0 * PI);
                let theta = rng.gen_range(0.0..PI);
                
                let director = Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos()
                );
                
                let q = QTensor::from_director(&director, 0.6).unwrap();
                q_tensors.push(q);
            }
        },
        _ => {
            // Default to uniform
            let director = Vector3::new(0.0, 0.0, 1.0);
            let q = QTensor::from_director(&director, 0.6).unwrap();
            for _ in 0..total_sites {
                q_tensors.push(q.clone());
            }
        },
    };
    
    MicroscopicConfiguration {
        dimensions: (nx, ny, nz),
        q_tensors,
        temperature,
        external_field: None,
    }
}

/// Calculate the bulk free energy for a microscopic configuration
pub fn calculate_bulk_free_energy(config: &MicroscopicConfiguration, params: &MicroscopicParameters) -> f64 {
    let mut energy = 0.0;
    
    for q in &config.q_tensors {
        // Calculate trace quantities
        let tr_q2 = (q.components.clone() * q.components.clone()).trace();
        let tr_q3 = (q.components.clone() * q.components.clone() * q.components.clone()).trace();
        
        // Landau-de Gennes bulk free energy terms
        // F_bulk = a/2 tr(Q²) - b/3 tr(Q³) + c/4 [tr(Q²)]²
        energy += params.a / 2.0 * tr_q2;
        energy -= params.b / 3.0 * tr_q3;
        energy += params.c / 4.0 * tr_q2 * tr_q2;
    }
    
    energy
}

/// Calculate the elastic free energy for a microscopic configuration
pub fn calculate_elastic_free_energy(config: &MicroscopicConfiguration, params: &MicroscopicParameters) -> f64 {
    let (nx, ny, nz) = config.dimensions;
    let mut energy = 0.0;
    
    // Simplified calculation - in a real implementation we would use proper finite differences
    // for gradient calculations to handle the three elastic constants L1, L2, L3
    
    // Iterate through the interior of the lattice
    for i in 1..nx-1 {
        for j in 1..ny-1 {
            for k in 1..nz-1 {
                let idx = i * ny * nz + j * nz + k;
                let q = &config.q_tensors[idx];
                
                // Get neighboring Q-tensors for gradient calculation
                let q_x_plus = &config.q_tensors[(i+1) * ny * nz + j * nz + k];
                let q_y_plus = &config.q_tensors[i * ny * nz + (j+1) * nz + k];
                let q_z_plus = &config.q_tensors[i * ny * nz + j * nz + (k+1)];
                
                // Calculate gradients using finite differences
                let grad_x = &q_x_plus.components - &q.components;
                let grad_y = &q_y_plus.components - &q.components;
                let grad_z = &q_z_plus.components - &q.components;
                
                // Calculate gradient squared terms (simplified)
                let grad_sq_sum = grad_x.iter().map(|&x| x * x).sum::<f64>()
                                + grad_y.iter().map(|&y| y * y).sum::<f64>()
                                + grad_z.iter().map(|&z| z * z).sum::<f64>();
                
                // One-constant approximation for elastic energy
                energy += 0.5 * params.l1 * grad_sq_sum;
            }
        }
    }
    
    energy
}

/// Calculate the total free energy for a microscopic configuration
pub fn calculate_free_energy(config: &MicroscopicConfiguration, params: &MicroscopicParameters) -> f64 {
    let bulk_energy = calculate_bulk_free_energy(config, params);
    let elastic_energy = calculate_elastic_free_energy(config, params);
    
    // External field contribution (if present)
    let field_energy = if let Some(h_field) = &config.external_field {
        let mut energy = 0.0;
        for q in &config.q_tensors {
            // E_field = -h * Q_ij * H_i * H_j
            let h_vec = DVector::from_iterator(3, h_field.iter().cloned());
            energy -= params.h * (h_vec.transpose() * q.components.clone() * h_vec)[(0, 0)];
        }
        energy
    } else {
        0.0
    };
    
    bulk_energy + elastic_energy + field_energy
}

/// Create a category for microscopic configurations
pub fn create_microscopic_category() -> 
    FinCategory<MicroscopicConfiguration, MicroscopicMorphism> {
    // Create sample objects
    let config1 = generate_microscopic_configuration(5, 5, 5, "uniform", 300.0);
    let config2 = generate_microscopic_configuration(5, 5, 5, "twisted", 300.0);
    
    // Create a morphism between them
    let morphism = MicroscopicMorphism {
        domain: config1.clone(),
        codomain: config2.clone(),
        transformation_type: "Twist".to_string(),
        parameters: None,
    };
    
    // Create identity morphisms
    let id1 = MicroscopicMorphism {
        domain: config1.clone(),
        codomain: config1.clone(),
        transformation_type: "Identity".to_string(),
        parameters: None,
    };
    
    let id2 = MicroscopicMorphism {
        domain: config2.clone(),
        codomain: config2.clone(),
        transformation_type: "Identity".to_string(),
        parameters: None,
    };
    
    // Create the category
    FinCategory::new(
        "MicroscopicCategory".to_string(),
        vec![config1, config2],
        vec![morphism, id1, id2],
    )
}

/// Calculate the RG beta function for microscopic parameters
pub fn beta_function_microscopic(params: &MicroscopicParameters) -> Result<DVector<f64>, RGFlowError> {
    // Calculate beta functions for all parameters
    // β(g) = dg/dl where l is the log of the scale factor
    
    let beta_a = 2.0 * params.a; // a has scaling dimension 2
    let beta_b = 1.5 * params.b; // b has scaling dimension 3/2
    let beta_c = params.c; // c has scaling dimension 1
    let beta_l1 = 0.0; // l1 is marginal
    let beta_l2 = 0.0; // l2 is marginal
    let beta_l3 = 0.0; // l3 is marginal
    let beta_h = 1.5 * params.h; // h has scaling dimension 3/2
    
    Ok(DVector::from_vec(vec![
        beta_a, beta_b, beta_c, beta_l1, beta_l2, beta_l3, beta_h
    ]))
}

/// Implement an RG step for microscopic parameters
pub fn rg_step_microscopic(params: &MicroscopicParameters) -> Result<MicroscopicParameters, RGFlowError> {
    // Scale factor for this RG step
    let scale = 1.5;
    
    // Update parameters based on their scaling dimensions
    let a_new = params.a * scale.powf(2.0);
    let b_new = params.b * scale.powf(1.5);
    let c_new = params.c * scale.powf(1.0);
    // Elastic constants stay unchanged in this simplified model
    let l1_new = params.l1;
    let l2_new = params.l2;
    let l3_new = params.l3;
    let h_new = params.h * scale.powf(1.5);
    
    // Temperature flows toward the critical temperature
    let t_critical = 330.0;
    let temp_new = t_critical + (params.temperature - t_critical) * scale.powf(-1.0);
    
    Ok(MicroscopicParameters {
        a: a_new,
        b: b_new,
        c: c_new,
        l1: l1_new,
        l2: l2_new,
        l3: l3_new,
        h: h_new,
        temperature: temp_new,
    })
}
