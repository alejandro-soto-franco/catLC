//! catLC: Category Theory-based Liquid Crystal Analysis
//!
//! This library provides tools for analyzing liquid crystal systems
//! at different scales using category theory and renormalization group flow.

pub mod category;
pub mod functor;
pub mod rg_flow;
pub mod microscopic;
pub mod mesoscopic;
pub mod macroscopic;
pub mod manifold;
pub mod visualization_data;

// Re-export key types for convenience
pub use category::{Category, Object, Morphism, FinCategory};
pub use functor::{Functor, ConcreteFunctor};
pub use rg_flow::{RGFlow, ParameterSpace};
pub use manifold::{Manifold, CurvedSpace, ManifoldPoint, TangentSpace};
