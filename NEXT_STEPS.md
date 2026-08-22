# Next Steps-

> **Current stage:** v5.7 is complete. The next major step is moving from the simplified 0-D model into CFD validation using OpenFOAM and ParaView.

---

## 1. Build the OpenFOAM Cooling Channel

- **Current:** 0-D physics + PINN model
- **Next:** Build the optimized rectangular cooling-channel geometry in OpenFOAM
- **Goal:** Create a CFD case using the geometry and flow conditions from v5.7
- **Status:** Starting v6

## 2. Validate Coolant Flow

- **Current:** Velocity and Reynolds number calculated using simplified equations
- **Next:** Simulate the coolant flow directly in OpenFOAM
- **Goal:** Compare CFD velocity, pressure drop, and flow behavior against the v5.7 predictions
- **Status:** Not yet simulated

## 3. Validate Heat Transfer

- **Current:** Heat transfer is represented using correlations and a simplified resistance network
- **Next:** Simulate conjugate heat transfer between the hot wall and methane coolant
- **Goal:** Compare CFD wall temperature and heat flux against the v5.7 physics model and PINN prediction
- **Status:** Not yet simulated

## 4. Replace the Fixed Heat-Flux Assumption

- **Current:** Uses an average-to-peak heat-flux factor to approximate the thermal load
- **Next:** Allow the CFD model to calculate the local heat-transfer behavior along the channel
- **Goal:** Obtain a real axial heat-flux and wall-temperature profile instead of using a single average value
- **Status:** Planned for v6

## 5. Improve Coolant Properties

- **Current:** Methane density, specific heat, conductivity, and viscosity are treated as constant
- **Next:** Introduce temperature- and pressure-dependent methane properties
- **Goal:** Make the CFD model more realistic, especially as the coolant approaches the methane critical region
- **Status:** Planned for v6

## 6. Validate the Gas-Side Heating

- **Current:** Uses fixed chamber and throat gas-side heat-transfer coefficients based on Bartz estimates
- **Next:** Replace the fixed values with a more realistic thermal boundary condition in CFD
- **Goal:** Determine whether the assumed gas-side heat transfer is reasonable and develop a more realistic chamber/throat heat-load distribution
- **Status:** Planned for v6

## 7. Compare PINN vs CFD

- **Current:** PINN has only been compared against the simplified physics model
- **Next:** Compare the PINN prediction directly against OpenFOAM results
- **Goal:** Determine how accurately the neural network predicts a higher-fidelity simulation
- **Status:** Major v6 objective

## 8. Investigate Model Error

- **Current:** No high-fidelity validation has been completed
- **Next:** Identify where the PINN and simplified model disagree with CFD
- **Goal:** Determine whether the errors come from the neural network, training data, simplified physics, or assumptions about the cooling system
- **Status:** Planned

## 9. Improve the Physics Model

- **Current:** Simplified 0-D thermal-fluid model
- **Next:** Add higher-fidelity physics based on what is learned from CFD
- **Goal:** Improve the training data and potentially retrain the PINN using more realistic thermal-fluid behavior
- **Status:** Future work

## 10. Explore Better Optimization

- **Current:** v5.2 optimized the cooling-channel design using the PINN surrogate
- **Next:** Include more realistic engineering constraints and potentially CFD-informed objectives
- **Goal:** Optimize wall temperature, pressure loss, coolant temperature rise, and manufacturability together instead of optimizing only a single temperature prediction
- **Status:** Future work

---

# v6 Main Goal

The main goal of v6 is to answer the question that the v5 series could not fully answer:

> **Does the optimized cooling-channel design predicted by my PINN actually work when simulated using higher-fidelity CFD?**

The planned workflow is:

```text
v5.7 Optimized Design
        │
        ▼
OpenFOAM Geometry
        │
        ▼
Coolant Flow Simulation
        │
        ▼
Conjugate Heat Transfer
        │
        ▼
ParaView Analysis
        │
        ▼
Compare CFD vs PINN
        │
        ▼
Improve Model
        │
        ▼
Future Optimization
