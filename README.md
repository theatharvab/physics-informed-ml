# Physics-Informed Machine Learning

**Physics-informed machine learning experiments for engineering and scientific modeling.**

---

## Research Question

> **Can a physics-informed machine learning framework predict and optimize regenerative cooling channel geometries for liquid rocket engines by minimizing combustion chamber wall temperatures and coolant pressure losses?**

---

## About My Project

I’m exploring how **physics-informed machine learning** can be used to predict and optimize regenerative cooling channel designs for liquid rocket engines.

My goal is to build a framework that combines **physics-based thermal-fluid calculations, neural networks, optimization, and CFD validation**.

Right now, I’m focusing on rectangular cooling channels and using a neural network to predict combustion-chamber wall temperature from channel geometry and coolant mass flow rate.

My long-term goal is to investigate whether machine learning can make the process of exploring large numbers of aerospace engineering designs faster while still staying connected to the underlying physics.

---

## My Current Project

My current model focuses on **rectangular regenerative cooling channels** surrounding a liquid rocket engine combustion chamber.

### My Design Variables

| Variable | Description            | Units |
| -------- | ---------------------- | ----- |
| `L`      | Channel length         | m     |
| `w`      | Channel width          | m     |
| `H`      | Channel height         | m     |
| `mdot`   | Coolant mass flow rate | kg/s  |

These variables define my cooling-channel geometry and coolant flow conditions.

---

## My Physics-Based Model

I generate my training data using simplified **heat-transfer and fluid-flow relationships**.

### Hydraulic Diameter

For my rectangular cooling channels, I calculate hydraulic diameter using:

$$
D_h = \frac{2wH}{w+H}
$$

### Flow Area

I calculate the channel cross-sectional flow area as:

$$
A = wH
$$

### Coolant Velocity

I calculate coolant velocity from mass flow rate and channel area:

$$
v = \frac{\dot{m}}{\rho A}
$$

### Reynolds Number

I use Reynolds number to characterize the coolant flow:

$$
Re = \frac{\rho vD_h}{\mu}
$$

### Prandtl Number

I calculate the Prandtl number from my coolant properties:

$$
Pr = \frac{C_p\mu}{k}
$$

### Nusselt Number

I currently use the **Dittus–Boelter correlation** to estimate convective heat transfer:

$$
Nu = 0.023Re^{0.8}Pr^{0.4}
$$

### Heat Transfer Coefficient

I calculate the heat-transfer coefficient using:

$$
h = \frac{Nu,k}{D_h}
$$

### Wall Temperature

I currently estimate my wall-temperature target using the coolant temperature, heat flux, and heat-transfer coefficient:

$$
T_{wall}=T_{coolant}+\frac{q''}{h}
$$

These equations give me a simplified physics-based model that I can use to generate training data for my neural network.

---

## My Machine Learning Model

I built my current neural network using **PyTorch**.

My model takes four input variables:

```text
Channel Length
Channel Width
Channel Height
Coolant Mass Flow Rate
```

These inputs are standardized using the mean and standard deviation from my training dataset.

My wall-temperature output is also standardized during training and converted back into Kelvin when I make predictions.

### My Current Network

```text
4 Input Features
      │
      ▼
Linear(4 → 32)
      │
     ReLU
      │
      ▼
Linear(32 → 32)
      │
     ReLU
      │
      ▼
Linear(32 → 1)
      │
      ▼
Predicted Wall Temperature
```

My current network is intentionally simple because I’m using it as a starting point for exploring physics-informed machine learning and engineering optimization.

---

## My Optimization

After training my neural network, I use it as a **surrogate model** to rapidly evaluate thousands of possible cooling-channel designs.

My optimization searches through the allowed design space and looks for geometries that produce lower predicted combustion-chamber wall temperatures.

My current workflow is:

```text
Physics-Based Model
        │
        ▼
Training Dataset
        │
        ▼
Neural Network
        │
        ▼
Fast Wall-Temperature Predictions
        │
        ▼
Large Design-Space Search
        │
        ▼
Optimized Cooling Channel
```

Using a neural-network surrogate allows me to explore many candidate designs much faster than running a full CFD simulation for every individual geometry.

---

## My Thermal & Flow Analysis

After finding an optimized design, I calculate additional engineering performance metrics.

### Heat Removed

I estimate heat removed from the combustion chamber using:

$$
Q=hA_s(T_{hot}-T_{wall})
$$

My current approximation for cooling-channel surface area is:

$$
A_s=2L(w+H)
$$

### Cooling Effectiveness

I calculate cooling effectiveness using:

$$
\epsilon=
\frac{T_{hot}-T_{wall}}
{T_{hot}-T_{in}}
$$

### Pressure Loss

I estimate coolant pressure loss using the **Darcy–Weisbach equation**:

$$
\Delta P =
f\frac{L}{D_h}
\frac{\rho v^2}{2}
$$

For my current model, I use a Blasius-style approximation for the friction factor:

$$
f=0.3164Re^{-0.25}
$$

These calculations allow me to look at more than just wall temperature and begin evaluating the overall performance of my cooling-channel designs.

---

## My CFD Validation

My next major step is connecting my machine-learning framework to **OpenFOAM** for CFD validation.

I plan to use:

* **OpenFOAM** for CFD simulation
* **ParaView** for visualization and post-processing

My intended workflow is:

```text
Physics-Based Model
        │
        ▼
PIML / Neural-Network Model
        │
        ▼
Optimized Geometry
        │
        ▼
OpenFOAM CFD
        │
        ▼
Compare Results
        │
        ▼
Evaluate Model Accuracy
```

My goal is to compare my simplified physics and machine-learning predictions against higher-fidelity CFD results.

This should help me identify where my current model performs well and where my simplified assumptions begin to break down.

---

## My Project Progress

### v5.1 — Building My Physics-Based Dataset

In v5.1, I built the foundation for my machine-learning model.

I:

* Defined my **liquid-methane coolant properties**
* Generated randomized cooling-channel geometries
* Generated coolant mass-flow rates
* Calculated hydraulic diameter
* Calculated coolant velocity
* Calculated Reynolds number
* Calculated Nusselt number
* Calculated the heat-transfer coefficient
* Generated wall-temperature targets
* Standardized my input data
* Standardized my output temperature data
* Trained my neural network
* Saved my trained model and normalization parameters

My v5.1 model became the foundation for the optimization work in v5.2.

---

### v5.2 — Optimizing My Cooling Channel

In v5.2, I loaded my trained v5.1 model and used it to explore a much larger design space.

I:

* Generated thousands of possible cooling-channel designs
* Normalized the designs using my v5.1 training statistics
* Used my trained neural network to predict wall temperatures
* Searched for the design with the lowest predicted wall temperature
* Saved the optimized design for further analysis

This allowed me to use my trained neural network as a fast surrogate for design exploration.

---

### v5.3 — Analyzing My Optimized Design

In v5.3, I loaded my optimized design from v5.2 and performed additional engineering calculations.

I calculated:

* Hydraulic diameter
* Cross-sectional flow area
* Coolant velocity
* Reynolds number
* Prandtl number
* Nusselt number
* Heat-transfer coefficient
* Cooling-channel surface area
* Predicted wall temperature
* Heat removed
* Cooling effectiveness
* Friction factor
* Coolant pressure loss

I’m also using this stage to prepare my optimized geometry for **CFD validation with OpenFOAM**.

---

## My Coolant Model

I currently use **liquid methane** as my coolant.

### My Simplified Coolant Properties

```text
Density                  ρ  = 420 kg/m³
Specific Heat            Cp = 3500 J/(kg·K)
Thermal Conductivity     k  = 0.1 W/(m·K)
Dynamic Viscosity        μ  = 1.1 × 10⁻⁵ Pa·s
Coolant Temperature           150 K
Heat Flux                     5 × 10⁶ W/m²
```

These are simplified properties that I’m currently using for my research model. They are not intended to represent a complete real-fluid property model across the full range of conditions inside an actual liquid rocket engine.

---

## Tools & Technologies

I’m currently using:

* **Python** — primary programming language
* **PyTorch** — neural-network development
* **NumPy** — numerical calculations
* **Matplotlib** — visualization
* **OpenFOAM** — CFD validation
* **ParaView** — CFD visualization and analysis

---

## What I Want to Improve

There are several areas I want to improve as I continue developing my framework:

* [ ] Improve my coolant property modeling
* [ ] Add higher-fidelity regenerative cooling physics
* [ ] Generate training and validation data using CFD
* [ ] Connect my ML predictions directly with CFD results
* [ ] Optimize both wall temperature and pressure loss
* [ ] Add additional physical constraints
* [ ] Perform model error and uncertainty analysis
* [ ] Investigate more realistic combustion-chamber and cooling-channel geometries
* [ ] Compare my ML predictions against CFD results
* [ ] Investigate physics-informed loss functions
* [ ] Improve the physical realism of my thermal-fluid model

---

## My Project Status

🚧 **Active Research Project**

I’m continuing to develop my physics-informed machine learning framework and work toward validating my optimized cooling-channel designs using CFD.

My current focus is moving from a simplified physics-based surrogate model toward a more realistic **aerospace thermal-fluid simulation and optimization workflow**.

---

## Copyright & Usage

© 2026 Atharva B. All rights reserved.

This project and its original content are my own work. If you would like to use, reproduce, modify, or build upon any part of my work, please contact me first.

