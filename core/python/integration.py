#!/usr/bin/env python3
"""
Integration script for catLC

This script demonstrates how the various components of catLC work together
to analyze liquid crystal configurations at different scales and perform
renormalization group flow analysis.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess
from visualizer import LCVisualizer
from curved_field_solver import CurvedFieldSolver
from rg_analyzer import RGFlowAnalyzer

# Make sure the output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_rust_simulation(command):
    """Run a Rust simulation using the catLC CLI"""
    rust_bin_path = os.path.join(os.path.dirname(__file__), "..", "rust", "target", "release", "catlc_cli")
    
    # Check if the binary exists, if not try debug build
    if not os.path.exists(rust_bin_path):
        rust_bin_path = os.path.join(os.path.dirname(__file__), "..", "rust", "target", "debug", "catlc_cli")
    
    # If still not found, try building it
    if not os.path.exists(rust_bin_path):
        print("Rust binary not found. Building the project...")
        rust_dir = os.path.join(os.path.dirname(__file__), "..", "rust")
        try:
            subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, check=True)
            rust_bin_path = os.path.join(rust_dir, "target", "release", "catlc_cli")
        except subprocess.CalledProcessError:
            print("Failed to build the Rust project. Make sure Rust is installed.")
            return False
    
    # Run the command
    try:
        cmd = [rust_bin_path] + command.split()
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Rust simulation: {e}")
        return False

def microscopic_analysis():
    """Perform a microscopic analysis"""
    print("\n=== Microscopic Analysis ===")
    
    # Run the Rust simulation for microscopic model
    if not run_rust_simulation("micro"):
        print("Failed to run microscopic simulation")
        return
    
    # Visualize the results
    visualizer = LCVisualizer(output_dir=OUTPUT_DIR)
    
    # Visualize different microscopic configurations
    patterns = ["uniform", "twisted", "defect"]
    for pattern in patterns:
        filename = f"{OUTPUT_DIR}/microscopic_{pattern}.json"
        if os.path.exists(filename):
            print(f"Visualizing {pattern} configuration...")
            fig, ax = visualizer.visualize_director_field(
                filename, 
                subsample=2, 
                save_path=f"{OUTPUT_DIR}/microscopic_{pattern}_viz.png",
                show=False
            )
            plt.close(fig)
    
    print("Microscopic analysis completed.")

def mesoscopic_analysis():
    """Perform a mesoscopic analysis"""
    print("\n=== Mesoscopic Analysis ===")
    
    # Run the Rust simulation for mesoscopic model
    if not run_rust_simulation("meso"):
        print("Failed to run mesoscopic simulation")
        return
    
    # Visualize the results
    visualizer = LCVisualizer(output_dir=OUTPUT_DIR)
    
    # Visualize mesoscopic field
    filename = f"{OUTPUT_DIR}/mesoscopic_field.json"
    if os.path.exists(filename):
        print("Visualizing mesoscopic field...")
        fig, ax = visualizer.visualize_director_field(
            filename, 
            subsample=1, 
            save_path=f"{OUTPUT_DIR}/mesoscopic_field_viz.png",
            show=False
        )
        plt.close(fig)
    
    # Visualize RG flow
    filename = f"{OUTPUT_DIR}/mesoscopic_rg_flow.json"
    if os.path.exists(filename):
        print("Visualizing mesoscopic RG flow...")
        fig, ax = visualizer.visualize_rg_flow(
            filename,
            save_path=f"{OUTPUT_DIR}/mesoscopic_rg_flow_viz.png",
            show=False
        )
        plt.close(fig)
    
    print("Mesoscopic analysis completed.")

def macroscopic_analysis():
    """Perform a macroscopic analysis"""
    print("\n=== Macroscopic Analysis ===")
    
    # Run the Rust simulation for macroscopic model
    if not run_rust_simulation("macro"):
        print("Failed to run macroscopic simulation")
        return
    
    # Visualize the results
    visualizer = LCVisualizer(output_dir=OUTPUT_DIR)
    
    # Visualize defects
    filename = f"{OUTPUT_DIR}/macroscopic_defects.json"
    if os.path.exists(filename):
        print("Visualizing macroscopic defects...")
        fig, ax = visualizer.visualize_defects(
            filename,
            save_path=f"{OUTPUT_DIR}/macroscopic_defects_viz.png",
            show=False
        )
        plt.close(fig)
    
    print("Macroscopic analysis completed.")

def curved_surface_analysis():
    """Analyze liquid crystals on curved surfaces"""
    print("\n=== Curved Surface Analysis ===")
    
    # Create solvers for different surface types
    surface_types = ["sphere", "torus", "hyperbolic"]
    
    for surface_type in surface_types:
        print(f"Analyzing liquid crystal on {surface_type}...")
        
        # Create solver
        solver = CurvedFieldSolver(surface_type=surface_type, resolution=30)
        
        # Optimize the director field
        print(f"Optimizing director field on {surface_type}...")
        solver.optimize_director_field(iterations=10)
        
        # Export to JSON
        output_file = f"{OUTPUT_DIR}/optimal_{surface_type}.json"
        solver.export_to_json(output_file)
        
        # Visualize the results
        visualizer = LCVisualizer(output_dir=OUTPUT_DIR)
        print(f"Visualizing liquid crystal on {surface_type}...")
        fig, ax = visualizer.visualize_curved_surface(
            output_file,
            save_path=f"{OUTPUT_DIR}/curved_{surface_type}_viz.png",
            show=False
        )
        plt.close(fig)
    
    print("Curved surface analysis completed.")

def rg_flow_analysis():
    """Perform RG flow analysis"""
    print("\n=== RG Flow Analysis ===")
    
    # Run the Rust simulation for RG flow analysis
    if not run_rust_simulation("rg"):
        print("Failed to run RG flow analysis")
        return
    
    # Load the mesoscopic RG flow data
    filename = f"{OUTPUT_DIR}/meso_rg_flow.json"
    if os.path.exists(filename):
        analyzer = RGFlowAnalyzer()
        analyzer.load_from_json(filename)
        
        # Generate phase diagrams for different parameter pairs
        parameter_pairs = [(0, 1), (1, 2), (0, 2)]
        for i, j in parameter_pairs:
            if i < len(analyzer.parameter_names) and j < len(analyzer.parameter_names):
                print(f"Generating phase diagram for {analyzer.parameter_names[i]} vs {analyzer.parameter_names[j]}...")
                fig, ax = analyzer.generate_phase_diagram(
                    dims=(i, j),
                    save_path=f"{OUTPUT_DIR}/phase_diagram_{i}_{j}.png",
                    show=False
                )
                plt.close(fig)
    
    # Load the macroscopic RG flow data
    filename = f"{OUTPUT_DIR}/macro_rg_flow.json"
    if os.path.exists(filename):
        analyzer = RGFlowAnalyzer()
        analyzer.load_from_json(filename)
        
        # Generate 3D flow visualization if we have at least 3 parameters
        if len(analyzer.parameter_names) >= 3:
            print("Generating 3D flow visualization...")
            fig, ax = analyzer.generate_3d_flow(
                dims=(0, 1, 2),
                save_path=f"{OUTPUT_DIR}/3d_flow.png",
                show=False
            )
            plt.close(fig)
    
    print("RG flow analysis completed.")

def full_analysis():
    """Run a full analysis pipeline"""
    print("=== Running Full Analysis ===")
    microscopic_analysis()
    mesoscopic_analysis()
    macroscopic_analysis()
    curved_surface_analysis()
    rg_flow_analysis()
    print("\n=== Full Analysis Completed ===")

def main():
    parser = argparse.ArgumentParser(description='catLC Integration Script')
    parser.add_argument('--microscopic', action='store_true', help='Run microscopic analysis')
    parser.add_argument('--mesoscopic', action='store_true', help='Run mesoscopic analysis')
    parser.add_argument('--macroscopic', action='store_true', help='Run macroscopic analysis')
    parser.add_argument('--curved', action='store_true', help='Run curved surface analysis')
    parser.add_argument('--rg', action='store_true', help='Run RG flow analysis')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    
    args = parser.parse_args()
    
    # Default to all if no specific analysis is requested
    if not (args.microscopic or args.mesoscopic or args.macroscopic or args.curved or args.rg):
        args.all = True
    
    if args.all:
        full_analysis()
    else:
        if args.microscopic:
            microscopic_analysis()
        if args.mesoscopic:
            mesoscopic_analysis()
        if args.macroscopic:
            macroscopic_analysis()
        if args.curved:
            curved_surface_analysis()
        if args.rg:
            rg_flow_analysis()

if __name__ == "__main__":
    main()
