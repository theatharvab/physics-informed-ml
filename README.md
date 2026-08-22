# Physics-Informed Machine Learning

**Physics-informed machine learning experiments for regenerative cooling and liquid rocket engine thermal-fluid modeling.**

---

## Research Question

> **Can a physics-informed machine learning framework predict and optimize regenerative cooling channel geometries for liquid rocket engines while satisfying thermal and hydraulic constraints?**

---

## About My Project

I’m exploring how **physics-informed machine learning** can be used to predict and optimize regenerative cooling channel designs for liquid rocket engines.

My goal is to build a framework that combines **physics-based thermal-fluid calculations, neural networks, optimization, engineering constraints, and CFD validation**.

My current model focuses on rectangular cooling channels around a liquid rocket engine combustion chamber. I use a neural network as a fast surrogate model to predict combustion-chamber wall temperature and then use that model to search for an optimized cooling-channel design.

The long-term goal is to see whether machine learning can make large engineering design searches faster while still staying connected to the underlying physics.

---

## My Current Project

My current project focuses on **rectangular regenerative cooling channels** surrounding a liquid rocket engine combustion chamber.

The coolant I am modeling is **liquid methane**.

The current design variables are:

| Variable | Description | Units |
| -------- | ----------- | ----- |
| `L` | Channel length | m |
| `w` | Channel width | m |
| `H` | Channel height | m |
| `nch` | Number of cooling channels | - |

The total coolant mass flow is currently set separately and is divided between the cooling channels.

This changed from the original model, where coolant mass flow was treated as an independent design variable. I changed this because the channel count and total flow need to be handled consistently when calculating the flow inside each individual channel.

---

## My Physics-Based Model

I generate and evaluate my designs using simplified **heat-transfer and fluid-flow relationships**.

The model is not a full CFD simulation. It is a lower-order model that lets me explore a large number of designs much faster.

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

### Per-Channel Mass Flow

The total coolant mass flow is divided between the cooling channels:

$$
\dot{m}_{channel}=\frac{\dot{m}_{total}}{n_{ch}}
$$

### Coolant Velocity

I calculate coolant velocity from the per-channel mass flow and channel area:

$$
v=\frac{\dot{m}_{channel}}{\rho A}
$$

### Reynolds Number

I use Reynolds number to determine the flow regime:

$$
Re=\frac{\rho vD_h}{\mu}
$$

### Prandtl Number

I calculate the Prandtl number from the coolant properties:

$$
Pr=\frac{C_p\mu}{k}
$$

### Nusselt Number

My original model used the Dittus–Boelter correlation.

As the project developed, I changed the main heat-transfer correlation to **Gnielinski**, with Dittus–Boelter kept as a fallback when Gnielinski is outside its valid range.

$$
Nu=
\frac{(f/8)(Re-1000)Pr}
{1+12.7\sqrt{f/8}(Pr^{2/3}-1)}
$$

I also check whether the current `Re` and `Pr` values are inside the applicable correlation ranges before using the result.

### Heat Transfer Coefficient

I calculate the coolant-side heat-transfer coefficient using:

$$
h=\frac{Nu\,k}{D_h}
$$

### Pressure Loss

I calculate coolant pressure loss using the Darcy–Weisbach equation:

$$
\Delta P =
f\frac{L}{D_h}
\frac{\rho v^2}{2}
$$

The friction factor used with the Gnielinski correlation is calculated using the Petukhov relationship.

---

## My Thermal Model

One of the biggest changes between my early models and the final v5.7 model was how I calculate wall temperature and heat flux.

Instead of simply imposing a fixed heat flux, I use a thermal resistance network:

```text
Hot Gas
   │
   ▼
Gas-Side Convection
   │
   ▼
Chamber Wall
   │
   ▼
Wall Conduction
   │
   ▼
Coolant-Side Convection
   │
   ▼
Liquid Methane
