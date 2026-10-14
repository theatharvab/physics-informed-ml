# CFD: OpenFOAM Validation of the Optimized Cooling Channel

This folder documents the computational fluid dynamics (CFD) work for my regenerative cooling-channel optimization project. I used OpenFOAM Foundation v14 to model coolant flow through a cooling channel and compare the resulting thermal predictions with my physics-based model and machine-learning surrogate.

The CFD work is still a work in progress. The results below describe the simulations and assumptions currently documented for this project. They should not be treated as a completed, grid-independent validation of a full rocket-engine cooling system.

[← Back to the main project](../)

## What I modeled

The simulations focus on coolant flow through a single cooling channel. The coolant is modeled as liquid methane with constant thermophysical properties. The chamber case was run at multiple mesh resolutions, and a separate mid-throat case was used to explore the effect of a higher imposed heat flux.

The current model does **not** include a complete combustion-gas domain or conjugate heat transfer through a solid engine wall. Heat is applied as a boundary condition on the channel wall. This means the results are useful for investigating the modeled setup, but they do not yet represent a complete engine-level regenerative cooling simulation.

## Software

- **OpenFOAM Foundation v14** for CFD
- **ParaView** for visualization and post-processing
- **Ubuntu/WSL** as the OpenFOAM environment

The solver used for the documented runs was `foamRun -solver fluid`, with a (k\text{-}\varepsilon) turbulence model. A previous attempt to use (k\text{-}\omega) SST did not complete because of a thermophysical-transport configuration issue. A wall-resolved SST study is still planned.

## Geometry and operating assumptions

The optimized geometry currently used for the project is:

| Parameter | Value |
|---|---:|
| Channel length | 0.51290 m |
| Channel width | 1.9757 mm |
| Channel height | 7.6601 mm |
| Number of channels around the engine | 155 |
| Hydraulic diameter | 3.1412 mm |
| Total coolant mass flow used in the design model | 20 kg/s |
| Approximate mass flow per channel | 0.1290 kg/s |
| Coolant inlet temperature | 115 K |

The channel dimensions and total flow are design inputs. The approximate per-channel mass flow is the total flow divided by 155 channels.

### Constant-property coolant model

The current CFD setup uses these constant liquid-methane properties:

| Property | Value |
|---|---:|
| Density | 422 kg/m³ |
| Specific heat capacity | 3480 J/(kg·K) |
| Thermal conductivity | 0.187 W/(m·K) |
| Dynamic viscosity | 1.2 × 10⁻⁴ Pa·s |
| Prandtl number | 2.2332 |

These values are assumptions in the current model, not a temperature- and pressure-dependent methane equation of state. That limitation matters because the estimated mid-throat outlet temperature is around 232 K, above methane's critical temperature of approximately 190.6 K. The physical state of methane depends on pressure as well as temperature, so the outlet state and constant-property approximation need to be investigated before treating the result as representative of a real engine.

## Boundary conditions and numerical setup

The documented setup uses a coolant inlet temperature of 115 K, a prescribed pressure reference at the outlet, no-slip channel walls, and an imposed heat flux on the heated wall. The outlet pressure value of 100 kPa is a numerical reference in the current setup, not a claim about the engine's actual coolant operating pressure.

The turbulence setup uses (k\text{-}\varepsilon) with wall functions and a turbulent Prandtl number of 0.85. The current runs used a single-block structured mesh. The exact case dictionaries and final boundary-condition files should be checked against the original simulation folders before attempting to reproduce these runs. I do not want the README to imply that an unverified case configuration is already reproducible.

## Chamber mesh study

The documented chamber-wall temperature results are:

| Mesh | Cell count | Reported wall temperature | Change from previous mesh |
|---|---:|---:|---:|
| Coarse | 38,400 | 541.75 K | — |
| Medium | 307,200 | 544.07 K | +0.43% |
| Fine | 2,457,600 | 552.95 K | +1.63% |

Reported runtimes were approximately 11.1 s, 263 s, and 8206.8 s, respectively. Runtime depends on the machine and execution setup, so these are observations from the runs, not universal performance estimates.

The fine-mesh temperature is about 1.63% higher than the medium-mesh temperature. This means the current results have **not demonstrated mesh independence**. A more complete study should report convergence histories and wall y-plus, use a suitable near-wall treatment, and estimate discretization uncertainty, for example with a Grid Convergence Index when its assumptions are satisfied.

## Current thermal comparison

The current chamber results include:

| Method | Reported temperature |
|---|---:|
| Fine-mesh CFD | 552.95 K |
| Neural-network prediction | 553.26 K |
| Physics-based thermal model | 564.19 K |

The neural network prediction is close to the fine-mesh CFD value for this case. However, that agreement alone does not prove that either model is physically accurate. The CFD result and physics-model result may represent different wall surfaces or use different heat-transfer assumptions. The comparison needs to be made on a like-for-like basis before drawing a strong validation conclusion.

For the mid-throat stress case, the reported CFD temperature is 1475.23 K and the physics-model temperature is 1551.5 K, a difference of approximately 4.92%. Both exceed the project's 825 K wall-temperature constraint. This stress case therefore indicates that the current setup does not satisfy that constraint under the imposed mid-throat heat load.

## Heat-flux assumptions

The chamber case used a reported heat flux of approximately 16.147 MW/m² on the heated face, corresponding to roughly 16.4 kW over the modeled heated surface. The mid-throat case used a reported heat flux of approximately 51.6 MW/m², corresponding to roughly 52 kW over that surface. These values and heated-face area should be verified against the original case files before they are treated as final.

The CFD heat load and the heat load used in the physics model are not currently identical. That difference needs to be reconciled because it affects the resulting temperature and makes direct comparisons harder to interpret.

## Limitations

The main limitations I am working to address are:

- **Mesh independence has not been established.** The wall temperature still changes between the medium and fine chamber meshes.
- **Near-wall resolution needs checking.** Earlier estimates of y-plus varied substantially across meshes, so the wall treatment and mesh need to be considered together.
- **The models may not predict the same wall temperature.** The CFD temperature is associated with the modeled coolant-side wall, while the physics model may report a hot-gas-side wall temperature. The wall-temperature difference across the solid should be accounted for before comparing them.
- **The heat loads are not yet fully reconciled.** The current chamber CFD heat load is approximately 16.4 kW, while the physics-model energy balance was reported at approximately 39.9 kW.
- **Methane properties are constant.** The predicted mid-throat outlet temperature makes a temperature- and pressure-dependent property model an important next step.
- **The CFD is not yet independent validation of the full cooling design.** It uses prescribed heat fluxes and simplified coolant-only modeling, rather than a coupled combustion-gas, solid-wall, and coolant simulation.
- **The turbulence model needs further verification.** A wall-resolved SST setup and more careful near-wall treatment are planned.

These limitations do not make the current work useless. They define what the current simulations can support and what needs to be tested next.

## Next steps

1. Recover and publish the exact OpenFOAM case dictionaries and mesh-generation setup used for each reported run.
2. Record residual histories, convergence of monitored quantities, and final y-plus values.
3. Extend the mesh study and estimate discretization uncertainty.
4. Compare CFD and the physics model using the same temperature definition, geometry, and heat load.
5. Check coolant pressure drop against an independent estimate, including a Darcy–Weisbach calculation.
6. Compare heat-transfer predictions against an appropriate correlation such as Gnielinski within its range of applicability.
7. Investigate temperature- and pressure-dependent methane properties and the predicted coolant outlet state.
8. Consider conjugate heat transfer or an equivalent wall-conduction treatment so coolant-side and hot-gas-side wall temperatures can be compared consistently.

## Reproducing the simulations

The general OpenFOAM workflow for a case is:

```bash
blockMesh
foamRun -solver fluid | tee run.log
```

These commands are examples of the workflow, not a guarantee that a complete runnable case is currently included in this repository. The case dictionaries, mesh settings, and boundary conditions must be present and checked before running them. Post-processing commands should likewise be matched to the actual patch names and OpenFOAM function-object syntax in the case.

When the original case files are added, this section should include the exact case directory names, complete commands, software version, expected outputs, and a small results summary. Large generated meshes and time directories can be regenerated rather than committed, provided the instructions and source dictionaries are complete.

## Related work

The CFD results are compared with the machine-learning and physics-based methods documented in the [main project repository](../). The reported values are preliminary and may change as the simulation setup, physical assumptions, and validation procedure are improved.
