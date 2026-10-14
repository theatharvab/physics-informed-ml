# ML: Dataset Generation, Neural Network, and Design Optimization

This folder contains the Python code I used for the machine-learning part of my regenerative cooling-channel project. I built the workflow in Google Colab using PyTorch and NumPy. It generates a dataset from a physics-based model, trains a neural network to predict hot-gas-side wall temperature, and searches for cooling-channel designs that meet a set of engineering constraints.

One important distinction: the network is trained on outputs from a closed-form thermal model. Its loss is standard mean squared error, with no governing-equation residual in the loss function. So this is a neural-network surrogate trained on physics-generated data, not a physics-informed neural network in the strict sense used in the original PINN literature. The optimizer also uses the physics model to evaluate constraints and score designs. The network prediction is recorded for comparison, but does not choose the winning design in the v5.2 optimization script.

## Scripts and run order

The scripts are meant to be run in order. Later stages use outputs saved by the earlier ones.

| Order | Script | Purpose |
|---|---|---|
| 1 | [`v5_1_training.py`](v5_1_training.py) | Generates the training data and trains the network. |
| 2 | [`v5_2_optimization.py`](v5_2_optimization.py) | Searches the design space and saves the optimization results. |
| 3 | [`v5_3_performance.py`](v5_3_performance.py) | Checks the selected design's chamber performance and runs a throat stress test. |
| 4 | [`v5_4_physics_audit.py`](v5_4_physics_audit.py) | Compares the network's predictions with the physics model. |
| 5 | [`v5_5_thermal_validation.py`](v5_5_thermal_validation.py) | Checks the energy balance and thermal margins. |
| 6 | [`v5_6_physical_realism_audit.py`](v5_6_physical_realism_audit.py) | Reviews physical assumptions and plausibility checks. |
| 7 | [`v5_7_final.py`](v5_7_final.py) | Runs the final constraint audit and reports values for the CFD hand-off. |

The repository currently contains the scripts, but not all of the saved model checkpoints and output files needed to reproduce the reported results directly. I still need to add a pinned `requirements.txt` and decide which generated artifacts should be committed.

## Physics model

For a rectangular channel with width \(w\) and height \(H\), the model calculates the flow area and hydraulic diameter as

\[
A = wH, \qquad D_h = \frac{2wH}{w+H}.
\]

The total coolant flow is 20 kg/s and the approximate flow per channel is \(\dot m = \dot m_{total}/N\). From that, the model calculates velocity, Reynolds number, and Prandtl number:

\[
v = \frac{\dot m}{\rho A}, \qquad
Re = \frac{\rho vD_h}{\mu}, \qquad
Pr = \frac{c_p\mu}{k}.
\]

The friction factor uses the Petukhov correlation,

\[
f = (0.79\ln Re - 1.64)^{-2},
\]

and the model uses the Gnielinski correlation for Nusselt number where its range of applicability is met. Dittus-Boelter is used as a fallback when Gnielinski does not apply. The coolant-side heat-transfer coefficient is \(h=Nu\,k/D_h\).

The overall heat-transfer coefficient includes the gas-side coefficient, wall conduction, and coolant-side coefficient:

\[
U = \frac{1}{1/h_{gas}+t_{wall}/k_{wall}+1/h}.
\]

The model then calculates heat flux, hot-gas-side wall temperature, pressure loss using Darcy-Weisbach, and an estimated coolant outlet temperature. The outlet-temperature calculation uses a factor of 0.25 to convert the local heat flux into an assumed average for the energy balance. That factor is an assumption, not a calculated result. Also, the heat-flux area in this model does not match the single heated face used in the current CFD setup, so the outlet-temperature results should not be compared directly without reconciling the heat-load assumptions.

### Fixed inputs

| Parameter | Value |
|---|---:|
| Methane density | 422 kg/m³ |
| Specific heat capacity | 3480 J/(kg·K) |
| Thermal conductivity | 0.187 W/(m·K) |
| Dynamic viscosity | 1.2 × 10⁻⁴ Pa·s |
| Prandtl number | 2.2332 |
| Coolant inlet temperature | 115 K |
| Hot-gas driving temperature | 3500 K |
| Wall thermal conductivity | 320 W/(m·K) |
| Wall thickness | 1 mm |
| Chamber gas-side heat-transfer coefficient | 5500 W/(m²·K) |
| Throat stress-test coefficients | 18000, 26500, and 35000 W/(m²·K) |

These are the assumptions used in the current model, not temperature- and pressure-dependent methane properties. That is an important limitation for the throat case, where the predicted coolant temperature can enter a regime where the constant-property liquid-methane approximation is questionable.

## Dataset

The script samples candidate channel geometries using these ranges:

| Variable | Sampling |
|---|---|
| Length \(L\) | Uniform from 0.5 to 2.0 m |
| Width \(w\) | Uniform from 0.5 to 2.0 mm |
| Aspect ratio \(H/w\) | Uniform from 1 to 10 |
| Height filter | Keep only designs with \(1.0\text{ mm} \le H \le 8.0\text{ mm}\) |
| Channel count \(N\) | Uniform from 50 to 300 |

The initial sample contains 20,000 candidates. After applying the height filter, 12,699 remain. The training script also removes designs for which neither heat-transfer correlation applies or the velocity exceeds 300 m/s. I still need to record the exact number of samples that make it into the final training set.

The inputs and output are standardized, and the normalization values are saved with the model checkpoint.

One detail to keep in mind is that the uploaded training and optimization scripts treat channel count as a continuous value. The multi-seed analysis version rounded it to an integer, while the reported final geometry uses 155 channels. I need to make sure the final script and saved checkpoint correspond to the same integer-channel design.

## Network and training

| Item | Value |
|---|---|
| Inputs | Length, width, height, and channel count |
| Output | Hot-gas-side wall temperature |
| Architecture | 4 → 32 → 32 → 1, with ReLU activations |
| Loss | Mean squared error |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training | 1,500 epochs, full batch |
| Reported final loss | Approximately 5 × 10⁻⁴ in standardized units |

The current training script does not use a separate validation or test set, and it does not set a random seed. Because of that, the reported final-design error is not enough to show how well the network generalizes across the full feasible design space. I need to add a held-out test set and report metrics such as MAE and RMSE in kelvin, along with an error map across the feasible region.

## Optimization

The optimization script samples 20,000 candidate designs. The reported official run used random seed 39. After the height filter, it evaluates each candidate with the physics model and also records the network prediction.

The hard constraints are:

- Velocity no greater than 100 m/s
- Reynolds number no greater than 3.0 × 10⁶
- Pressure drop no greater than 1.8 MPa
- Hot-gas-side wall temperature no greater than 825 K
- Estimated outlet temperature no greater than 380 K
- A valid heat-transfer correlation must be available

There are also plausibility filters. The maximum absolute standardized input distance must be no greater than 3, and mass flux and wall temperature must fall within the 1st to 99th percentile ranges reported by Waxenegger-Wilfing et al. (2020).

For designs that pass the constraints, the score is

\[
Score = 0.6\left(\frac{\Delta P}{1.8\text{ MPa}}\right)
+0.4\left(\frac{T_{wall}}{825\text{ K}}\right).
\]

The 60/40 weighting is a design choice. It is not derived from the governing equations. The lowest-scoring feasible candidate is selected. In the reported run, 1,498 of the 12,699 candidates passed the feasibility checks. Since this is a random search, the result can change with the random seed.

In the current v5.2 script, the physics-model values determine feasibility and the score. The neural-network prediction is saved for comparison but does not affect the selected design. The final audit script has a wall-temperature check that uses the network prediction instead, so those checks need to be made consistent.

### Random-seed sensitivity

The multi-seed analysis repeats the search for seeds 1, 15, 16, 17, 21, 28, 39, 44, 52, and 58. It reports the winning design and summary statistics for wall temperature, pressure loss, Reynolds number, channel count, and velocity.

The reported multi-seed results do not perfectly match the official result. One possible reason is the difference in how channel count is rounded, but I still need to compare the runs and document the exact cause.

## Current reported result

| Quantity | Value |
|---|---:|
| Channel length | 0.51290 m |
| Channel width | 1.9757 mm |
| Channel height | 7.6601 mm |
| Channel count | 155 |
| Hydraulic diameter | 3.1412 mm |
| Aspect ratio | 3.877 |
| Physics-model hot-gas-side wall temperature | 564.19 K |
| Neural-network prediction | 553.26 K |
| Difference between network and physics model | 10.92 K, or 1.94% below the physics result |
| Pressure loss | 0.215 MPa |
| Estimated outlet temperature in the optimization result | 203.86 K |
| Throat stress-test wall temperatures | 1244.4 K, 1551.5 K, and 1784.9 K for the low, middle, and high gas-side coefficients |

The outlet-temperature number above comes from the optimization result. Other checks have produced different outlet estimates because the heat-transfer area and heat-load assumptions are not yet fully consistent across the models. I need to reconcile those before presenting one value as definitive.

The throat stress-test temperatures are all above the 825 K wall-temperature constraint. That means the selected design does not satisfy that constraint under those throat stress-test conditions.

The CFD results and the remaining assumptions are documented in the [CFD README](../CFD/README.md).

## Speed comparison

A single neural-network call was reported to take 0.000382 seconds, excluding training. The hardware and whether the timing used a batch still need to be documented. Compared with the reported coarse, medium, and fine CFD runtimes of 11.1 s, 263 s, and 8206.8 s, this is approximately 29,000×, 688,000×, and 21.5 million× faster, respectively.

That comparison is against the CFD runs, not against the closed-form physics model. In a quick vectorized NumPy test, evaluating the physics model for 12,699 designs took well under a millisecond on the test CPU. The neural network's larger speed advantage would matter most if its training data came from an expensive source, such as CFD, rather than a fast analytical model.

## Current limitations

These are the main issues I still need to address:

- There is no held-out test set, and the training script does not set a random seed.
- The scripts currently treat channel count inconsistently, including using a continuous value in the training and optimization scripts.
- The v5.2 optimization uses physics values for selection, while the final audit also checks the network prediction.
- The energy-balance check in v5.5 is a self-consistency check because the outlet temperature is calculated from the same heat quantity being checked. It is not independent validation.
- The mass-flux plausibility lower bound is 3027 kg/(m²·s) in the code and 3207 kg/(m²·s) in the paper. I need to check the source table and correct the discrepancy.
- The coolant properties are constant even though the predicted operating conditions may take methane outside the range where a constant-property liquid model is appropriate.
- The 100 m/s velocity cap is an engineering choice. It is above the roughly 61 m/s value cited in NASA SP-8087 and below the mean velocity of 126 m/s reported in the Waxenegger-Wilfing et al. (2020) dataset.
- The sources for the width, height, and channel-count limits still need to be verified. The main project README tracks the status of the constraint sources.

## What I plan to do next

1. Add a held-out test set and report MAE, RMSE, and prediction error across the feasible region.
2. Decide whether the network is only used for screening or should influence design selection. If it is used for screening, rank candidates with the network and verify the strongest candidates with the physics model.
3. Train and evaluate a surrogate on CFD-generated data and compare it against the current closed-form-model baseline.
4. Replace constant methane properties with a temperature- and pressure-dependent model.
5. Report a Pareto front showing the trade-off between pressure drop and wall temperature instead of relying only on one weighted score.
6. Reconcile the channel-count rounding, mass-flux limit, heat-load assumptions, and outlet-temperature calculations.
7. Add pinned package versions and the model checkpoint or other artifacts needed to reproduce the reported results.

This is still ongoing work. The numbers here describe the current implementation and should be updated as the model and validation process improve.
