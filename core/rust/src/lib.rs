//! # catLC: Categorical Liquid Crystals
//!
//! A category theoretic framework for renormalization group flows in liquid crystals.

pub mod category;
pub mod functor;
pub mod rg_flow;
pub mod manifold;
pub mod microscopic;
pub mod mesoscopic;
pub mod macroscopic;
pub mod visualization_data;

/// Re-export commonly used types
pub use category::{Category, Object, Morphism, ComposableMorphisms};
pub use functor::{Functor, NaturalTransformation};
pub use rg_flow::{RGFlow, RGFixedPoint};
pub use manifold::{Manifold, TangentSpace, CurvedSpace};
