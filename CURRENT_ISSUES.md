# Current Model Issues-

> **Coolant basis:** Liquid/supercritical methane (LCH₄). The coolant
> properties currently used in the model are based on methane.

## 1. Wall Temperature

- Current: ~160–167 K
- Preferred/target: ~600–800 K
- Hard constraint: ≤800–1000 K, depending on material and thermal limits
- Issue: The predicted wall temperature is much lower than expected for the
  intended liquid-methane regenerative cooling environment.
- Status: Investigating the thermal formulation, boundary conditions, and
  heat-transfer implementation.

## 2. Coolant Velocity

- Current: 145.9 m/s
- Preferred/target: Lower velocity while still providing enough cooling
- Hard constraint: Limited by acceptable pressure drop and liquid/supercritical
  methane engine conditions
- Issue: The optimized design produces a very high methane coolant velocity,
  which also contributes to the high pressure loss.
- Status: Investigating channel geometry, methane mass flow, and optimization
  constraints.

## 3. Reynolds Number

- Current: 2.10 × 10⁷
- Preferred/target: Within the validated range of the selected methane
  heat-transfer correlation
- Hard constraint: Must stay within the range where the selected correlation
  is considered reliable
- Issue: The current Reynolds number is far outside the range being used for
  the current methane heat-transfer correlation.
- Status: Reviewing channel geometry, methane flow conditions, and correlation
  validity.

## 4. Pressure Drop

- Current: 2.82 MPa
- Preferred/target: As low as possible while still providing effective methane
  cooling and sufficient enthalpy rise
- Hard constraint: Must stay below the allowable pressure drop for the selected
  engine conditions
- Issue: The predicted pressure loss is too high for the current design.
- Status: Investigating channel geometry, methane coolant velocity, and
  friction-factor assumptions.

## 5. Channel Width

- Current: 0.72 mm
- Preferred/target: ~1.0 mm or greater
- Hard constraint: Depends on the selected engine and manufacturing limits
- Issue: The current channel width is below the ~1.0 mm value used as a
  manufacturing constraint in the cooling-channel literature.
- Status: Reviewing channel geometry and manufacturing constraints.

## 6. Channel Height

- Current: 2.5 mm
- Preferred/target: Based on the desired channel aspect ratio and hydraulic
  diameter
- Hard constraint: Depends on the selected engine and channel geometry
- Issue: The current height itself is not necessarily a problem, but it
  contributes to the current aspect-ratio issue.
- Status: Reviewing channel geometry and aspect-ratio constraints.

## 7. Aspect Ratio

- Current: 0.35
- Preferred/target: Within the validated range of the selected methane
  cooling model
- Hard constraint: Must remain within the range supported by the model and
  heat-transfer correlations
- Issue: The current aspect ratio is outside the range used by the relevant
  methane cooling-channel studies.
- Status: Investigating channel geometry and optimization constraints.

## 8. Coolant Outlet Temperature

- Current: 162.6 K
- Preferred/target: Enough temperature/enthalpy rise for the intended
  liquid-methane engine cycle while remaining thermally stable
- Hard constraint: Must remain below the applicable thermal degradation/
  coking limit and satisfy engine-cycle requirements
- Issue: The predicted methane outlet temperature is unusually low and may
  be related to the same issue causing the unrealistically low wall
  temperature.
- Status: Investigating the energy balance, methane enthalpy rise, and
  temperature-dependent coolant properties.
