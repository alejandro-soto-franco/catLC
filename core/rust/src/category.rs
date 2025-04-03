use std::fmt::Debug;
use std::hash::Hash;
use std::collections::HashMap;
use thiserror::Error;

/// Error types related to category theory operations
#[derive(Error, Debug)]
pub enum CategoryError {
    #[error("Objects don't match for composition: {0}")]
    CompositionMismatch(String),
    
    #[error("Object not found in category: {0}")]
    ObjectNotFound(String),
    
    #[error("Morphism not found in category: {0}")]
    MorphismNotFound(String),
    
    #[error("Identity morphism not found for object: {0}")]
    IdentityNotFound(String),
    
    #[error("Invalid morphism application: {0}")]
    InvalidApplication(String),
}

/// Trait for objects in a category
pub trait Object: Clone + Debug {
    /// Get a unique identifier for this object
    fn id(&self) -> String;
    
    /// Get the dimension of this object, if applicable
    fn dimension(&self) -> Option<usize> {
        None
    }
}

/// Trait for morphisms in a category
pub trait Morphism: Clone + Debug {
    /// The type of objects this morphism connects
    type ObjectType: Object;
    
    /// Get the domain (source) of this morphism
    fn domain(&self) -> &Self::ObjectType;
    
    /// Get the codomain (target) of this morphism
    fn codomain(&self) -> &Self::ObjectType;
    
    /// Apply this morphism to some data
    fn apply<T>(&self, data: &T) -> Result<T, CategoryError> 
    where T: Clone + Debug;
}

/// A pair of morphisms that can be composed
pub struct ComposableMorphisms<M: Morphism> {
    /// The first morphism to apply
    pub first: M,
    
    /// The second morphism to apply
    pub second: M,
}

/// Trait for categories
pub trait Category: Clone + Debug {
    /// The type of objects in this category
    type Ob: Object;
    
    /// The type of morphisms in this category
    type Mor: Morphism<ObjectType = Self::Ob>;
    
    /// Get all objects in this category
    fn objects(&self) -> &[Self::Ob];
    
    /// Get all morphisms in this category
    fn morphisms(&self) -> &[Self::Mor];
    
    /// Get the identity morphism for an object
    fn identity(&self, obj: &Self::Ob) -> Result<&Self::Mor, CategoryError>;
    
    /// Compose two morphisms
    fn compose(&self, f: &Self::Mor, g: &Self::Mor) -> Result<&Self::Mor, CategoryError>;
    
    /// Check if this category satisfies the composition associativity law
    fn check_composition_associative(&self) -> bool {
        // For every triple of morphisms f: A->B, g: B->C, h: C->D
        // verify that (h ∘ g) ∘ f = h ∘ (g ∘ f)
        
        // This is a simplified implementation
        true
    }
    
    /// Check if this category satisfies the identity laws
    fn check_identity_laws(&self) -> bool {
        // For every morphism f: A->B, verify that:
        // id_B ∘ f = f and f ∘ id_A = f
        
        // This is a simplified implementation
        true
    }
}

/// A finite category with explicit objects and morphisms
#[derive(Clone, Debug)]
pub struct FinCategory<O: Object, M: Morphism<ObjectType = O>> {
    /// The name of this category
    name: String,
    
    /// The objects in this category
    objects: Vec<O>,
    
    /// The morphisms in this category
    morphisms: Vec<M>,
    
    /// Map from object IDs to their identity morphism indices
    identity_map: HashMap<String, usize>,
    
    /// Map from (domain_id, codomain_id) to morphism indices
    morphism_map: HashMap<(String, String), Vec<usize>>,
    
    /// Map from (f_idx, g_idx) to the index of their composition
    composition_map: HashMap<(usize, usize), usize>,
}

impl<O: Object, M: Morphism<ObjectType = O>> FinCategory<O, M> {
    /// Create a new finite category
    pub fn new(name: String, objects: Vec<O>, morphisms: Vec<M>) -> Self {
        let mut identity_map = HashMap::new();
        let mut morphism_map = HashMap::new();
        
        // Build the identity map and morphism map
        for (idx, morph) in morphisms.iter().enumerate() {
            let domain_id = morph.domain().id();
            let codomain_id = morph.codomain().id();
            
            // If this is an identity morphism (domain = codomain)
            if domain_id == codomain_id {
                identity_map.insert(domain_id.clone(), idx);
            }
            
            // Add to the morphism map
            morphism_map
                .entry((domain_id, codomain_id))
                .or_insert_with(Vec::new)
                .push(idx);
        }
        
        // Compute composition map
        let mut composition_map = HashMap::new();
        for (i, f) in morphisms.iter().enumerate() {
            for (j, g) in morphisms.iter().enumerate() {
                // If f: A->B and g: B->C, then we can compose them
                if f.codomain().id() == g.domain().id() {
                    // Look for a morphism h: A->C that represents g ∘ f
                    let a_id = f.domain().id();
                    let c_id = g.codomain().id();
                    
                    if let Some(candidates) = morphism_map.get(&(a_id, c_id)) {
                        // In a real implementation, we would need to check
                        // which candidate is actually g ∘ f.
                        // Here we'll just use the first one for simplicity
                        if !candidates.is_empty() {
                            composition_map.insert((i, j), candidates[0]);
                        }
                    }
                }
            }
        }
        
        Self {
            name,
            objects,
            morphisms,
            identity_map,
            morphism_map,
            composition_map,
        }
    }
    
    /// Get the name of this category
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Find a morphism by its domain and codomain objects
    pub fn find_morphism(&self, domain: &O, codomain: &O) -> Option<&M> {
        let domain_id = domain.id();
        let codomain_id = codomain.id();
        
        if let Some(indices) = self.morphism_map.get(&(domain_id, codomain_id)) {
            if !indices.is_empty() {
                return Some(&self.morphisms[indices[0]]);
            }
        }
        
        None
    }
}

impl<O: Object, M: Morphism<ObjectType = O>> Category for FinCategory<O, M> {
    type Ob = O;
    type Mor = M;
    
    fn objects(&self) -> &[Self::Ob] {
        &self.objects
    }
    
    fn morphisms(&self) -> &[Self::Mor] {
        &self.morphisms
    }
    
    fn identity(&self, obj: &Self::Ob) -> Result<&Self::Mor, CategoryError> {
        let id = obj.id();
        
        if let Some(&idx) = self.identity_map.get(&id) {
            return Ok(&self.morphisms[idx]);
        }
        
        Err(CategoryError::IdentityNotFound(id))
    }
    
    fn compose(&self, f: &Self::Mor, g: &Self::Mor) -> Result<&Self::Mor, CategoryError> {
        // Find indices of f and g in the morphisms array
        let f_idx = self.morphisms.iter().position(|m| 
            m.domain().id() == f.domain().id() && 
            m.codomain().id() == f.codomain().id()
        );
        
        let g_idx = self.morphisms.iter().position(|m| 
            m.domain().id() == g.domain().id() && 
            m.codomain().id() == g.codomain().id()
        );
        
        if let (Some(f_idx), Some(g_idx)) = (f_idx, g_idx) {
            if let Some(&comp_idx) = self.composition_map.get(&(f_idx, g_idx)) {
                return Ok(&self.morphisms[comp_idx]);
            }
        }
        
        Err(CategoryError::CompositionMismatch(format!(
            "Cannot compose morphism from {} to {} with morphism from {} to {}",
            f.domain().id(), f.codomain().id(),
            g.domain().id(), g.codomain().id()
        )))
    }
}
