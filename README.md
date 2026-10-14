# Physics-Informed Machine Learning for Regenerative Cooling-Channel Optimization

**Ongoing independent research project | Initiated for the Texas Junior Academy of Science (TJAS) 2026 | Atharva Banchhod**

This project explores how machine learning can be used to optimize regenerative cooling channels in liquid-methane rocket engines. I developed a physics-based thermal-fluid model, trained a neural network to approximate its predictions, and used it to search for cooling-channel designs that meet thermal and hydraulic constraints. I then used OpenFOAM CFD to evaluate the selected design.

I presented an earlier version of this research at the 2026 Texas Junior Academy of Science. I am continuing the project by working on improving the CFD validation, refining the physical assumptions, and making the model more representative of actual rocket-engine operating conditions.

**Research question:** Can a physics-informed machine learning framework predict and optimize regenerative cooling-channel geometries for liquid rocket engines while meeting thermal and hydraulic constraints?

## Scope and assumptions

There are a few important things to understand about what the current model does and does not cover.

- **The optimization only covers chamber conditions.** The neural network and optimizer use a gas-side heat-transfer coefficient of 5,500 W/m²·K. Throat conditions, where the coefficient ranges from 18,000 to 35,000 W/m²·K, are tested separately after optimization. They are not included in training or the design search.
- **The model is physics-informed, but it is not a PINN in the usual sense.** The training data come from a physics-based model, and the optimizer applies physics-based constraints. However, the neural network uses a standard mean squared error loss and does not include governing-equation residuals in its loss function. The scripts still use `PINN` in their names for historical reasons.
- **The neural network approximates a physics model, not CFD directly.** Its training targets come from a thermal resistance-network calculation. The speedup reported below compares neural-network inference with CFD, not with the original physics model, which is already fast.
- **The selected design is a candidate, not a proven global optimum.** It was selected through a random search. The project does not establish a Pareto front or prove that no better design exists.
- **Some inputs are assumed rather than simulated.** These include the gas-side heat-transfer coefficient, constant coolant properties, and a fixed total coolant mass flow rate of 20 kg/s.

## Final design under chamber conditions

The optimization selected the following design from the feasible candidates.

| Parameter | Result |
|---|---|
| Channel length | 0.51290 m |
| Channel width | 1.9757 mm |
| Channel height | 7.6601 mm |
| Aspect ratio | 3.877 |
| Hydraulic diameter | 3.1412 mm |
| Number of channels | 155 |
| Mass flow per channel | 0.1290 kg/s |
| Coolant velocity | Approximately 20.2 m/s |
| Reynolds number | Approximately \(2.2 \times 10^5\) |
| Pressure drop | 0.215 MPa, against a limit of 1.8 MPa |
| Coolant outlet temperature from the energy balance | 203.86 K, against a limit of 380 K |
| Feasible designs | 1,498 out of 12,699 candidates |

The outlet-temperature result needs to be interpreted carefully because the energy-balance calculation and the CFD setup use different heat loads. I explain this in the limitations section below.

## Results

### Chamber-condition wall temperatures

I compared the physics model and neural-network predictions with CFD results at three mesh resolutions.

| Method | Hot-gas-side wall temperature | Notes |
|---|---:|---|
| Physics-based resistance network | 564.19 K | Used to generate the neural network's training targets |
| Neural network | 553.26 K | 10.92 K, or 1.94%, below the physics model |
| Coarse CFD mesh, 38,400 cells | 541.75 K | |
| Medium CFD mesh, 307,200 cells | 544.07 K | 0.43% higher than the coarse result |
| Fine CFD mesh, 2,457,600 cells | 552.95 K | 1.63% higher than the medium result |

Compared with the fine-mesh CFD result, the physics model differs by 1.99%, while the neural network differs by 0.06%.

However, the 0.06% agreement does not necessarily mean the neural network independently predicts the actual physical behavior that accurately. The network predicts a temperature about 11 K below its training target, while the physics model predicts a temperature about 11 K above the CFD result. These errors partly cancel each other out.

The CFD temperature also changes as the mesh is refined, so the final result has not yet been shown to be grid-independent.

### Throat-condition stress test

After selecting the chamber design, I tested the same geometry at higher gas-side heat-transfer coefficients to see how it performed under throat-like heating conditions. These tests did not affect the optimization.

| Gas-side heat-transfer coefficient | Physics-model temperature | CFD temperature |
|---|---:|---:|
| 18,000 W/m²·K | 1,244.4 K | Not run |
| 26,500 W/m²·K | 1,551.5 K | 1,475.23 K |
| 35,000 W/m²·K | 1,784.9 K | Not run |

The wall-temperature limit used in this project is 825 K. All three physics-model predictions exceed this limit, and the CFD result at 26,500 W/m²·K exceeds it as well.

Under the current assumptions, the selected geometry does not meet the temperature constraint at throat-level heating. This means additional cooling methods, such as film cooling, would need to be investigated for those conditions.

### Computational time

One neural-network inference took 0.000382 seconds, excluding training. The reported CFD runtimes were:

| CFD mesh | Wall-clock time |
|---|---:|
| Coarse, 38,400 cells | 11.1 seconds |
| Medium, 307,200 cells | 263.0 seconds |
| Fine, 2,457,600 cells | 8,206.8 seconds |

Using these timings, the neural network was approximately 21.5 million times faster than the fine-mesh CFD run. This comparison is between neural-network inference and CFD. It does not mean the network is 21.5 million times faster than the original physics model.

The hardware and batch size used for the inference timing still need to be documented.

## Limitations of the comparison

There are several things I still need to check before treating these results as a complete validation of the model.

**1. The close neural-network and CFD agreement may be misleading.**

The neural network predicts 553.26 K, while the fine-mesh CFD result is 552.95 K. However, the network is trained on a physics model that predicts 564.19 K. The network's error relative to its training target and the difference between the physics model and CFD partly cancel each other out. The 0.06% difference should not be treated as proof of independent physical accuracy.

**2. Mesh independence has not been established.**

The wall temperature increases from 541.75 K on the coarse mesh to 544.07 K on the medium mesh and 552.95 K on the fine mesh. The change becomes larger with refinement rather than smaller. Wall-function \(y^+\) values also change between meshes. More refinement and a formal grid-convergence analysis are needed.

More details are available in [CFD/README.md](CFD/README.md).

**3. The physics model and CFD do not currently report the same wall surface.**

The CFD domain contains only the coolant, so the temperature reported at `heatedWall` is the coolant-side wall temperature. The physics model predicts the hot-gas-side wall temperature.

These surfaces are separated by the wall's thermal resistance, with an estimated temperature difference of about 50 K in the chamber case and 161 K in the mid-throat case.

To make the comparison more meaningful, I need to evaluate the same wall surface in both models or account for the temperature difference across the wall.

**4. The CFD and energy balance use different heat loads.**

In the CFD setup, heat flux is applied to one channel face, corresponding to approximately 16.4 kW. The physics-model outlet-temperature calculation applies 0.25 times the heat flux over the full wetted perimeter, corresponding to approximately 39.9 kW.

Because the heat loads differ, the predicted coolant outlet temperature of 203.86 K should not be interpreted as the outlet temperature from the current CFD setup. Under the CFD heat load, the estimated outlet temperature is approximately 151 K.

**5. The energy-balance calculation is not independent validation.**

The outlet temperature is calculated using the same assumed heat load used in the energy balance. Checking the result against that calculation shows internal consistency, but it does not independently validate the model against experimental or CFD data.

## Project workflow

The project has gone through several versions of the Python model, followed by CFD evaluation.

| Stage | Script | Purpose |
|---|---|---|
| v5.1 | `ML/v5_1_training.py` | Generate training data and train the neural network |
| v5.2 | `ML/v5_2_optimization.py` | Search channel designs and apply constraints |
| v5.3 | `ML/v5_3_performance.py` | Evaluate performance |
| v5.4 | `ML/v5_4_physics_audit.py` | Audit the physics model |
| v5.5 | `ML/v5_5_thermal_validation.py` | Check thermal calculations |
| v5.6 | `ML/v5_6_physical_realism_audit.py` | Check physical realism |
| v5.7 | `ML/v5_7_final.py` | Final model and constraint audit |
| CFD | `CFD/` | OpenFOAM case documentation and CFD evaluation |

The CFD setup and results are documented in [CFD/README.md](CFD/README.md).

## Model assumptions

The following inputs are fixed in the current model.

| Parameter | Value |
|---|---|
| Coolant | Liquid methane |
| Coolant density | 422 kg/m³ |
| Specific heat capacity | 3,480 J/kg·K |
| Thermal conductivity | 0.187 W/m·K |
| Dynamic viscosity | \(1.2 \times 10^{-4}\) Pa·s |
| Prandtl number | 2.2332 |
| Coolant inlet temperature | 115 K |
| Hot-gas driving temperature | 3,500 K |
| Wall thermal conductivity | 320 W/m·K |
| Wall thickness | 1.0 mm |
| Chamber gas-side heat-transfer coefficient | 5,500 W/m²·K |
| Throat stress-test coefficients | 18,000, 26,500, and 35,000 W/m²·K |
| Total coolant mass flow | 20 kg/s, divided equally between channels |
| Average-to-peak heat-flux factor | 0.25, used only for the outlet-temperature energy balance |

The wall conductivity represents a copper-alloy-class material. The heat-transfer calculation uses the Gnielinski correlation with the Petukhov friction factor, a Dittus–Boelter fallback, and the Darcy–Weisbach equation for pressure loss.

## Design constraints

The optimizer rejects designs that fail the constraints below. However, not all of these limits have been independently verified against a source that applies directly to this engine configuration.

| Parameter | Constraint | Basis and limitations |
|---|---|---|
| Coolant velocity | ≤ 100 m/s | Engineering judgment. NASA SP-8087 supports the general rationale for limiting velocity but cites approximately 61 m/s, not 100 m/s. |
| Reynolds number | ≤ \(3.0 \times 10^6\) | Based on correlation-validity considerations. The specific limit has not been located in Huzel and Huang. |
| Pressure drop | ≤ 1.8 MPa | An engineering margin. Absolute channel pressure must also remain above methane's critical pressure of 4.599 MPa, which is not established by this per-channel model. |
| Hot-gas-side wall temperature | ≤ 825 K | Based on the copper-alloy service range, including GRCop-84 information in NASA NTRS 20050196556. |
| Coolant outlet temperature | ≤ 380 K | The supporting coking literature, NASA NTRS 19810021741, concerns RP-1 or propane wall temperatures. Applying it to methane bulk outlet temperature is an extrapolation. |
| Channel width | 0.5–2.0 mm | Placeholder manufacturing limits that have not been independently verified. |
| Channel height | 1.0–8.0 mm | Placeholder manufacturing limits that have not been independently verified. |
| Aspect ratio | 1–10 | Derived from the selected width and height limits. |
| Channel count | 50–300 | No verified source has been identified for this range. |
| Mass flux and wall-temperature plausibility | Within the 1st–99th percentile of Waxenegger-Wilfing et al. (2020) CFD data | The paper's lower mass-flux bound is listed as 3,207 kg/s·m², while the code uses 3,027. The source table needs to be checked and reconciled. |

These constraints help narrow the design search, but they should not be interpreted as a complete set of validated rocket-engine design requirements.

## Current limitations and future work

The model still has several limitations that I want to address as I continue the project.

- **Constant methane properties:** Methane is transcritical under relevant conditions, with a critical point of 190.56 K and 4.599 MPa. Its properties, especially specific heat capacity near the pseudocritical region, can change significantly with temperature and pressure.
- **Simplified hot-gas conditions:** The model uses a prescribed gas-side heat-transfer coefficient instead of simulating combustion. It does not include conjugate heat transfer or an axial heat-flux profile.
- **Limited optimization:** The total coolant flow is fixed, and the search returns one selected candidate rather than a Pareto front.
- **Sensitivity to random seeds:** The 10-seed analysis and official optimization run do not fully agree. The differences need to be documented and reconciled.
- **No held-out test set:** The neural network has not yet been evaluated on a separate test set covering the feasible design region.

My next steps are:

1. Compare the physics model and CFD using the same wall surface, either by extracting the coolant-side wall temperature or by accounting for the wall-conduction temperature difference.
2. Run CFD with a heat load consistent with the physics model, or explicitly compare the heat loads used in both approaches.
3. Report \(y^+\) for each mesh, add a finer mesh, estimate grid-convergence error, and consider a wall-resolved turbulence model.
4. Compare the CFD pressure drop with the Darcy–Weisbach prediction of 0.215 MPa and compare the CFD Nusselt number with the Gnielinski correlation.
5. Create a held-out test set and report neural-network error across the feasible region, rather than relying on one design.
6. Train and evaluate a surrogate using CFD-generated data and introduce temperature-dependent methane properties.

## Repository structure

```text
├── README.md          Project overview and main results
├── NEXT_STEPS.md      Development notes and future work
├── ML/                Dataset generation, training, optimization, and audits
├── CFD/               OpenFOAM case documentation and CFD results
└── paper/             Planned location for the TJAS paper PDF
```

## Reproducibility

The machine-learning work was developed in Google Colab using Python, PyTorch, NumPy, pandas, and Matplotlib. The CFD simulations used OpenFOAM v14 on Ubuntu through WSL, with ParaView for visualization.

A `requirements.txt` file and the trained model checkpoint still need to be added to the repository. The training script does not set a random seed, so preserving the trained checkpoint is necessary to reproduce the exact network. The official optimization run used random seed 39 for candidate sampling.

As I continue the project, I plan to improve the documentation and make the full workflow easier to reproduce.

## Citation and license

Banchhod, A. (2026). *Physics-Informed Machine Learning for Regenerative Cooling Channel Optimization in Liquid Rocket Engines, Validated by Computational Fluid Dynamics.* Texas Junior Academy of Science.

This citation refers to the version of the research presented at TJAS 2026. The project is still ongoing, so later results and revisions may differ from that version.

A license has not yet been selected for the repository. Advisor and mentor acknowledgements are included in the research paper.
