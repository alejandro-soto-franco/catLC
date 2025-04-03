# catLC: Multiscale Categorical Field Theory for Liquid Crystals

A framework for analyzing liquid crystal systems across scales using category theory and renormalization group flow.

## Overview

catLC provides a comprehensive computational framework for analyzing liquid crystal systems at different scales (microscopic, mesoscopic, and macroscopic) through the lens of category theory. It implements renormalization group flows as functors between categories, allowing rigorous analysis of how information propagates across scales.

The framework consists of three main components:
1. **Formal Layer (Lean)**: Mathematical formalization of the category-theoretic framework
2. **Implementation Layer (Rust)**: Efficient computational implementation of the models
3. **Visualization Layer (Python)**: Tools for visualizing and analyzing results

## Architecture

- `core/lean/`: Formal mathematical framework in the Lean theorem prover
- `core/rust/`: Implementation of the computational models in Rust
- `core/python/`: Visualization and analysis tools in Python

## Features

- **Multiscale Modeling**: Analysis of liquid crystals at microscopic, mesoscopic, and macroscopic scales
- **Category Theory Foundation**: Mathematical rigor through category-theoretic formulations
- **RG Flow Analysis**: Implementation of renormalization group flow as functors
- **Curved Manifolds**: Support for liquid crystal configurations on curved surfaces
- **3D Visualization**: Advanced visualization of director fields, defects, and RG flows

## Installation

### Rust Implementation

```bash
cd core/rust
cargo build --release
```

### Python Visualization

```bash
cd core/python
pip install -r requirements.txt
```

### Lean Formalization (optional)

Requires Lean 4 and mathlib.

```bash
cd core/lean
lake build
```

## Usage

### Running the Full Pipeline

```bash
cd core/python
python integration.py --all
```

### Running Specific Components

```bash
# Microscopic analysis
python integration.py --microscopic

# Mesoscopic analysis
python integration.py --mesoscopic

# Macroscopic analysis
python integration.py --macroscopic

# Curved surface analysis
python integration.py --curved

# RG flow analysis
python integration.py --rg
```

### Direct Rust CLI Usage

```bash
cd core/rust
cargo run --release -- micro  # Run microscopic simulation
cargo run --release -- meso   # Run mesoscopic simulation
cargo run --release -- macro  # Run macroscopic simulation
cargo run --release -- curved sphere  # Run simulation on a sphere
```

## Examples

### Microscopic Configuration
![Microscopic Configuration](docs/imgs/microscopic_example.png)

### Defect Structure
![Defect Structure](docs/imgs/defects_example.png)

### RG Flow
![RG Flow](docs/imgs/rg_flow_example.png)

### LC on Curved Surface
![LC on Curved Surface](docs/imgs/curved_example.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
