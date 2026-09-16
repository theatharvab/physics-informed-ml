# Computational Fluid Dynamics   (WIP)

This folder contains the OpenFOAM CFD work for my regenerative cooling-channel optimization project.

## Overview

I used **OpenFOAM 14** to simulate regenerative cooling channels using liquid methane as the coolant. The CFD simulations were used to generate the data needed for my ML model and to validate that the simulations were producing physically reasonable results.

### What I looked at

* Wall temperature
* Coolant pressure loss
* Different cooling-channel geometries
* CFD results across the design space

## Software

* **OpenFOAM 14** — CFD simulations
* **ParaView** — Visualization and post-processing

## Files

* `methaneChannel/` — OpenFOAM case files
* `validation/` — CFD validation
* `figures/` — CFD visualizations and results

## Validation

The CFD results were validated before being used for the ML portion of the project.

These simulations serve as the physics-based reference for my machine learning model.

## Next Step

The validated CFD data is used to train a **PyTorch neural network** that can predict cooling-channel performance much faster than running a new CFD simulation for every design.

[← Back to main project](../)
