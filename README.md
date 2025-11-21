# Numerical Algorithms for Orbital Calculations

This repository contains a collection of scripts designed to implement, compare, and evaluate the performance of numerical algorithms commonly used in orbital mechanics.

Specifically, it includes:

- **Izzo’s Lambert solver** — a modern and efficient method for solving Lambert’s problem, used to compute transfer trajectories between two position vectors over a specified time of flight.
- **Two orbital propagators** based on numerical integration:
  - **Runge–Kutta 4th order (RK4)**
  - **Runge–Kutta 8th order (RK8)**

These tools allow users to test the accuracy, stability, and computational efficiency of different numerical approaches to solving two-body orbital motion and trajectory design problems.

## Features
- Modular and easy-to-extend Python scripts.
- Numerical experimentation for educational or research purposes.
- Direct comparison between RK4 and RK8 step accuracy.
- Analysis-ready output for plotting and performance evaluation.

## Requirements

### Core Software
- Python 3.12.4 (recommended)

### Mandatory Python Packages
- numpy
- scipy
- matplotlib

### Internal Modules
- `auxiliar.utils.py`  
- `auxiliar.izzo.py`  
- `auxiliar.rk4_solver.py`

### Outputs 
The numerical results will not be shown on screen, however they will be saved locally in the same folder under the names:
- `Lambert_solution_results.txt`  
- `RK4_results.txt`
- `RK8_results.txt`
- `RK8_results_J2.txt`  
