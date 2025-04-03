use catlc::{
    microscopic::{self, MicroscopicConfiguration, MicroscopicParameters},
    mesoscopic::{self, MesoscopicParameters},
    macroscopic::{self, MacroscopicParameters},
    rg_flow::{RGFlow, ConcreteRGFlow},
    category::{Category, FinCategory},
    functor::{Functor, ConcreteFunctor},
    visualization_data::{
        microscopic_to_director_field, 
        mesoscopic_to_director_field,
        macroscopic_to_defect_data,
        generate_curved_surface_data,
        save_to_json,
        generate_rg_flow_data
    },
    manifold::CurvedSpace,
};
use nalgebra::{Vector3, DVector};
use log::{info, error};
use std::error::Error;
use std::path::Path;
use std::fs;
use std::io::Write;

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    info!("Starting catLC CLI");
    
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }
    
    match args[1].as_str() {
        "micro" => {
            info!("Running microscopic simulation");
            run_microscopic_simulation()?;
        },
        "meso" => {
            info!("Running mesoscopic simulation");
            run_mesoscopic_simulation()?;
        },
        "macro" => {
            info!("Running macroscopic simulation");
            run_macroscopic_simulation()?;
        },
        "rg" => {
            info!("Running RG flow analysis");
            run_rg_flow_analysis()?;
        },
        "curved" => {
            info!("Generating LC configuration on curved surface");
            let surface_type = if args.len() > 2 { &args[2] } else { "sphere" };
            run_curved_surface_simulation(surface_type)?;
        },
        "help" | "--help" | "-h" => {
            print_usage();
        },
        _ => {
            error!("Unknown command: {}", args[1]);
            print_usage();
            return Err("Invalid command".into());
        }
    }
    
    Ok(())
}

fn print_usage() {
    println!(r#"
catLC: Category Theory-based Liquid Crystal Analysis

USAGE:
    catlc_cli [COMMAND] [OPTIONS]

COMMANDS:
    micro       Run microscopic simulation and generate visualization data
    meso        Run mesoscopic simulation and generate visualization data
    macro       Run macroscopic simulation and generate visualization data
    rg          Perform renormalization group flow analysis
    curved      Generate LC configurations on curved surfaces
    help        Show this help message

OPTIONS:
    For 'curved' command:
        sphere      Generate on a sphere (default)
        torus       Generate on a torus
        hyperbolic  Generate on a hyperbolic space
    "#);
}

fn run_microscopic_simulation() -> Result<(), Box<dyn Error>> {
    // Create output directory if it doesn't exist
    fs::create_dir_all("output")?;
    
    // Generate configurations with different patterns
    let patterns = vec!["uniform", "twisted", "defect", "random"];
    
    for pattern in patterns {
        info!("Generating {} configuration", pattern);
        let config = microscopic::generate_microscopic_configuration(
            20, 20, 20, pattern, 300.0
        );
        
        // Convert to visualization data
        let viz_data = microscopic_to_director_field(&config);
        
        // Save to JSON
        let filename = format!("output/microscopic_{}.json", pattern);
        info!("Saving to {}", filename);
        save_to_json(&viz_data, &filename)?;
    }
    
    // Create sample parameters
    let params = MicroscopicParameters {
        a: 0.1 * (300.0 - 330.0), // A(T-T*)
        b: 2.0,
        c: 1.0,
        l1: 1.0,
        l2: 1.0,
        l3: 1.0,
        h: 0.0,
        temperature: 300.0,
    };
    
    // Calculate free energy for a configuration
    let config = microscopic::generate_microscopic_configuration(
        10, 10, 10, "uniform", 300.0
    );
    let energy = microscopic::calculate_free_energy(&config, &params);
    println!("Free energy of uniform configuration: {}", energy);
    
    Ok(())
}

fn run_mesoscopic_simulation() -> Result<(), Box<dyn Error>> {
    // Create output directory
    fs::create_dir_all("output")?;
    
    // Create mesoscopic category
    let meso_cat = mesoscopic::create_mesoscopic_category();
    
    // Extract a sample Q-tensor field
    let field = &meso_cat.objects()[0].field;
    
    // Convert to visualization data
    let viz_data = mesoscopic_to_director_field(field, 300.0);
    
    // Save to JSON
    info!("Saving mesoscopic visualization data");
    save_to_json(&viz_data, "output/mesoscopic_field.json")?;
    
    // Create sample mesoscopic parameters
    let params = MesoscopicParameters {
        a: 0.1 * (300.0 - 330.0), // A(T-T*)
        b: 2.0,
        c: 1.0,
        l1: 1.0,
        l2: 1.0,
        h: 0.0,
        temperature: 300.0,
        xi: 1.0,
    };
    
    // Execute RG steps
    let mut current_params = params.clone();
    let mut parameter_trajectory = Vec::new();
    parameter_trajectory.push(current_params.clone());
    
    for i in 0..5 {
        info!("RG step {}", i+1);
        current_params = mesoscopic::rg_step_mesoscopic(&current_params)?;
        parameter_trajectory.push(current_params.clone());
    }
    
    // Save parameter trajectory
    let param_names = vec![
        "a".to_string(), "b".to_string(), "c".to_string(),
        "l1".to_string(), "l2".to_string(), "h".to_string(),
        "temperature".to_string(), "xi".to_string()
    ];
    
    let rg_data = generate_rg_flow_data(
        param_names,
        vec![parameter_trajectory],
        vec![
            (vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 300.0, 0.0], "stable".to_string()),
            (vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 330.0, 0.0], "unstable".to_string()),
        ]
    );
    
    info!("Saving RG flow data");
    save_to_json(&rg_data, "output/mesoscopic_rg_flow.json")?;
    
    Ok(())
}

fn run_macroscopic_simulation() -> Result<(), Box<dyn Error>> {
    // Create output directory
    fs::create_dir_all("output")?;
    
    // Create macroscopic category
    let macro_cat = macroscopic::create_macroscopic_category();
    
    // Extract a sample configuration with defects
    let config = macro_cat.objects()[0].clone();
    
    // Convert to visualization data
    let viz_data = macroscopic_to_defect_data(&config);
    
    // Save to JSON
    info!("Saving macroscopic visualization data");
    save_to_json(&viz_data, "output/macroscopic_defects.json")?;
    
    // Create sample macroscopic parameters
    let params = MacroscopicParameters {
        k1: 1.0, // Splay
        k2: 1.0, // Twist
        k3: 1.0, // Bend
        chi_a: 1.0,
        temperature: 300.0,
        core_energy: 5.0,
    };
    
    // Calculate interaction energy
    let energy = macroscopic::defect_interaction_energy(&config, &params);
    println!("Defect interaction energy: {}", energy);
    
    Ok(())
}

fn run_rg_flow_analysis() -> Result<(), Box<dyn Error>> {
    // Create output directory
    fs::create_dir_all("output")?;
    
    // Create categories for different scales
    let micro_cat = microscopic::create_microscopic_category();
    let meso_cat = mesoscopic::create_mesoscopic_category();
    let macro_cat = macroscopic::create_macroscopic_category();
    
    // Create functors between categories
    info!("Creating functors between categories");
    let micro_to_meso = mesoscopic::create_micro_to_meso_functor(
        micro_cat.clone(), 
        meso_cat.clone()
    );
    
    let meso_to_macro = macroscopic::create_meso_to_macro_functor(
        meso_cat.clone(),
        macro_cat.clone()
    );
    
    // Define RG flows
    let meso_rg = ConcreteRGFlow::new(
        "MesoscopicRG".to_string(),
        meso_cat.clone(),
        micro_to_meso.clone(),
        mesoscopic::rg_step_mesoscopic,
        mesoscopic::beta_function_mesoscopic,
    );
    
    let macro_rg = ConcreteRGFlow::new(
        "MacroscopicRG".to_string(),
        macro_cat.clone(),
        meso_to_macro.clone(),
        macroscopic::rg_step_macroscopic,
        macroscopic::beta_function_macroscopic,
    );
    
    // Initial mesoscopic parameters
    let meso_params = MesoscopicParameters {
        a: 0.1,
        b: 2.0,
        c: 1.0,
        l1: 1.0,
        l2: 1.0,
        h: 0.0,
        temperature: 290.0,
        xi: 1.0,
    };
    
    // Run RG flow
    info!("Running mesoscopic RG flow");
    let mut meso_trajectory = Vec::new();
    let mut current = meso_params.clone();
    meso_trajectory.push(current.clone());
    
    for _ in 0..10 {
        current = meso_rg.step(&current)?;
        meso_trajectory.push(current.clone());
    }
    
    // Initial macroscopic parameters
    let macro_params = MacroscopicParameters {
        k1: 1.0,
        k2: 1.0,
        k3: 1.0,
        chi_a: 1.0,
        temperature: 290.0,
        core_energy: 5.0,
    };
    
    // Run RG flow
    info!("Running macroscopic RG flow");
    let mut macro_trajectory = Vec::new();
    let mut current = macro_params.clone();
    macro_trajectory.push(current.clone());
    
    for _ in 0..10 {
        current = macro_rg.step(&current)?;
        macro_trajectory.push(current.clone());
    }
    
    // Generate visualization data
    let meso_param_names = vec![
        "a".to_string(), "b".to_string(), "c".to_string(),
        "l1".to_string(), "l2".to_string(), "h".to_string(),
        "temperature".to_string(), "xi".to_string()
    ];
    
    let macro_param_names = vec![
        "k1".to_string(), "k2".to_string(), "k3".to_string(),
        "chi_a".to_string(), "temperature".to_string(), "core_energy".to_string()
    ];
    
    let meso_rg_data = generate_rg_flow_data(
        meso_param_names,
        vec![meso_trajectory],
        vec![
            (vec![0.0, 2.0, 1.0, 2.0, 2.0, 0.0, 300.0, 0.0], "stable".to_string()),
            (vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 330.0, 0.0], "unstable".to_string()),
        ]
    );
    
    let macro_rg_data = generate_rg_flow_data(
        macro_param_names,
        vec![macro_trajectory],
        vec![
            (vec![2.0, 2.0, 2.0, 0.0, 300.0, 10.0], "stable".to_string())
        ]
    );
    
    // Save visualizations
    info!("Saving RG flow data");
    save_to_json(&meso_rg_data, "output/meso_rg_flow.json")?;
    save_to_json(&macro_rg_data, "output/macro_rg_flow.json")?;
    
    // Try to find fixed points
    info!("Searching for fixed points");
    let meso_fixed = meso_rg.find_fixed_point(&meso_params, 50, 1e-3)?;
    println!("Mesoscopic fixed point: {:?}", meso_fixed);
    
    Ok(())
}

fn run_curved_surface_simulation(surface_type: &str) -> Result<(), Box<dyn Error>> {
    // Create output directory
    fs::create_dir_all("output")?;
    
    // Create curved space based on type
    let curved_space = match surface_type {
        "sphere" => {
            CurvedSpace::Sphere {
                radius: 1.0,
                center: Vector3::new(0.0, 0.0, 0.0),
            }
        },
        "torus" => {
            CurvedSpace::Torus {
                major_radius: 2.0,
                minor_radius: 0.5,
            }
        },
        "hyperbolic" => {
            CurvedSpace::HyperbolicSpace {
                radius: 1.0,
            }
        },
        _ => {
            return Err(format!("Unknown surface type: {}", surface_type).into());
        }
    };
    
    info!("Generating LC configuration on {} surface", surface_type);
    let resolution = 50;
    let viz_data = generate_curved_surface_data(&curved_space, resolution)?;
    
    // Save visualization data
    let filename = format!("output/curved_{}.json", surface_type);
    info!("Saving to {}", filename);
    save_to_json(&viz_data, &filename)?;
    
    Ok(())
}
