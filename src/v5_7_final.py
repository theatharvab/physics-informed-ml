"""
v5.7 — Constraint-Aware Regenerative Cooling Optimization (FINAL)

Loads the v5.1 cooling surrogate and v5.2 optimized design, re-evaluates
the design using the v5.6-style physical realism checks, applies the
verified v5.7 engineering constraints against the CHAMBER regime (the
regime the optimizer actually searched), and runs the THROAT stress test
as a separate, clearly labeled evaluation that never feeds back into the
feasibility classification of the optimized design itself.

This is the last stage before v6 (OpenFOAM + ParaView CFD validation),
so the end of this file also prints a handoff block: every geometry,
flow, and boundary-condition value v6 will need, plus an explicit list
of what this 0-D model could NOT resolve and which v6 is meant to
replace (axial heat-flux profile, temperature-dependent coolant
properties, a real Bartz calculation instead of a fixed assumed h_gas).
"""

import torch
import torch.nn as nn

#recreates the same neural-network architecture used during v5.1 training
class CoolingNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.network=nn.Sequential(
      nn.Linear(4,32),
      nn.ReLU(),
      nn.Linear(32,32),
      nn.ReLU(),
      nn.Linear(32,1)
)
  def forward(self,x):
#passes normalized design variables through the trained neural network
    return self.network(x)

model=CoolingNN()

#loads the trained v5.1 surrogate model, which was trained only on the chamber regime
checkpoint=torch.load(
"PINN_Modelv5_1_complete.pth",
weights_only=False
)

model.load_state_dict(checkpoint["model"])
xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]
ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]
model.eval()

print("Complete Model v5.1 loaded")

#loads the optimized design generated during v5.2 chamber-regime optimization
designcheckpoint=torch.load(
"PINN_Modelv5_2_optimized_design.pth",
weights_only=False
)

bestdesign=designcheckpoint["bestdesign"]

print("Optimized design loaded")
print(bestdesign)

#extracts the optimized design variables
#the 4th design variable is nch, not mdot
#mdot was removed in v5.1 because it was independently sampled and caused unrealistic flow conditions
L=bestdesign[0]
w=bestdesign[1]
H=bestdesign[2]
nch=bestdesign[3]

print("L =",L)
print("w =",w)
print("H =",H)
print("nch =",nch)

#corrected liquid methane properties used consistently throughout v5.1-v5.7
#the old viscosity was about 11x too low and was the main source of the original unrealistic Re and h values
rho=422
Cp=3480
k=0.187
mu=1.2e-4
Pr=(Cp*mu)/k

#wall properties representing the NARloy-Z / GRCop-84 / C18150 range
kwall=320
tw=0.001

#corrected subcooled liquid methane inlet temperature
Tin=115

#combustion-side hot-gas temperature
T_hot=3500

#total coolant flow is fixed at the system level and split across nch channels
#this replaces the old independently sampled mdot design variable
totalmdot=20

#used for the coolant energy balance because the peak heat flux does not act uniformly over the full channel
avgfluxfactor=0.25

#chamber gas-side film coefficient used by v5.1 training and v5.2 optimization
#5500 W/m2K is the midpoint of the estimated 4000-7000 W/m2K chamber range from the Bartz-based operating class
hgaschamber=5500

#throat gas-side film coefficients are only used for the post-optimization stress test
#they are never included in training, optimization, or chamber feasibility checks
hgasthroat_low=18000
hgasthroat_mid=26500
hgasthroat_high=35000

print("Properties Defined")

#v5.7 engineering constraints
#these are the limits used to classify the optimized chamber-regime design
#the source notes below describe whether each value is directly supported or an engineering judgment

#maximum coolant velocity
#100 m/s is an engineering limit used to bound pressure drop and flow-related effects
#NASA SP-8087 supports the general velocity/pressure-drop basis, although its cited guidance is closer to 61 m/s
MAX_VELOCITY=100.0

#maximum reynolds number used for the accepted correlation/design range
#3e6 is retained as the v5.7 engineering constraint
#NASA SP-125 is a general reference for liquid rocket engine design and correlation practice, but the exact 3e6 value was not verified directly
MAX_REYNOLDS=3.0e6

#maximum allowable pressure drop
#1.8 MPa is the engineering pressure-drop limit used throughout the corrected v5.1-v5.7 chain
#NIST provides methane's critical pressure of about 4.599 MPa, while the 1.8 MPa drop is the engineering margin used here
MAX_PRESSURE_DROP=1.8e6

#maximum allowable hot-gas-side chamber wall temperature
#825 K is used as the thermal limit for the copper-alloy liner
#NASA NTRS 20050196556 discusses GRCop-84, NARloy-Z, and copper-alloy thermal fatigue behavior
MAX_WALL_TEMPERATURE=825.0

#maximum allowable coolant outlet temperature
#380 K is used as the modeled thermal-stability/coking limit for the coolant
#NASA NTRS 19810021741 is the verified hydrocarbon-deposit reference used for this constraint
MAX_OUTLET_TEMPERATURE=380.0

#channel width limits used by the corrected design space
#these remain engineering placeholders because a specific manufacturing source was not independently verified
MIN_WIDTH=0.0005
MAX_WIDTH=0.0020

#channel height limits used by the corrected design space
#these remain engineering placeholders because a specific manufacturing source was not independently verified
MIN_HEIGHT=0.0010
MAX_HEIGHT=0.0080

#aspect ratio is derived from the channel width and height limits
MIN_ASPECT_RATIO=1.0
MAX_ASPECT_RATIO=10.0

#nch is a design variable, not a fixed value
#50-300 is kept only as a reference range from the corrected v5.1/v5.2 design space
#the previously suggested NASA citation for this range was incorrect and should not be used
N_CHANNELS_MIN_REF=50
N_CHANNELS_MAX_REF=300

#calculates hydraulic diameter for the rectangular cooling channel
Dh=(2*w*H)/(w+H)

#calculates coolant flow area for one channel
A=w*H

#calculates per-channel flow and velocity from total flow and channel count
mdot_channel=totalmdot/nch
v=mdot_channel/(rho*A)

#calculates reynolds number for the corrected liquid methane properties
Re=(rho*v*Dh)/mu

#gnielinski is the primary heat-transfer correlation with petukhov friction factor
#dittus-boelter is kept only as a fallback when gnielinski is outside its valid range
f=(0.79*torch.log(torch.clamp(Re,min=1.0))-1.64)**-2
Nu_gnielinski=(f/8)*(Re-1000)*Pr/(1+12.7*torch.sqrt(f/8)*(Pr**(2/3)-1))
Nu_db=0.023*(Re**0.8)*(Pr**0.4)
gnielinski_valid=(Re>=3000)&(Re<=5e6)&(Pr>=0.5)&(Pr<=2000)
db_valid=(Re>=10000)&(Pr>=0.6)&(Pr<=160)
correlation_valid=gnielinski_valid|db_valid
Nu_standard=torch.where(gnielinski_valid,Nu_gnielinski,torch.where(db_valid,Nu_db,torch.tensor(float("nan"))))

#converts the selected nusselt correlation into the coolant-side heat-transfer coefficient
h_standard=(Nu_standard*k)/Dh

#calculates approximate wetted channel surface area
As=2*L*(w+H)

#uses the same petukhov friction factor as the gnielinski heat-transfer calculation
#the old model used a separate blasius-style factor that did not match the heat-transfer correlation
dP=f*(L/Dh)*(rho*v**2/2)

#uses the corrected three-resistance thermal network
#hot gas -> gas-side film -> wall conduction -> coolant-side convection
#qflux is calculated from the temperature difference instead of being imposed as a fixed value
U_chamber=1.0/(1.0/hgaschamber+tw/kwall+1.0/h_standard)
qflux_chamber=(T_hot-Tin)*U_chamber

#uses average heat flux for the coolant energy balance because the peak chamber flux is not uniform over the full channel
qflux_avg=qflux_chamber*avgfluxfactor
Q=qflux_avg*As

#calculates coolant temperature rise from the heat load and per-channel mass flow
dTcoolant=Q/(mdot_channel*Cp)

#calculates coolant outlet temperature from the energy balance
Tout=Tin+dTcoolant

#normalizes the optimized design using the same training normalization from v5.1
designnorm=(bestdesign-xmean)/xstd

with torch.no_grad():
  Twallnorm=model(designnorm)

#converts the normalized neural-network prediction back into kelvin
Twall=Twallnorm*ystd+ymean

#calculates the physics-based hot-gas-side wall temperature for the chamber regime
Tphysics=T_hot-qflux_chamber/hgaschamber

#compares the PINN prediction against the corrected physics model
difference=torch.abs(Twall-Tphysics)

#calculates the percentage difference between the PINN and physics result
errorpercent=(difference/Tphysics)*100

#calculates the channel aspect ratio from the optimized geometry
aspect_ratio=H/w

#checks which heat-transfer correlation is actually valid for this design
if correlation_valid:
  if gnielinski_valid:
    correlation_status="Gnielinski (primary)"
  else:
    correlation_status="Dittus-Boelter (fallback -- outside Gnielinski range)"
else:
  correlation_status="INVALID -- neither correlation applies at this Re/Pr"

#checks the corrected v5.7 reynolds constraint
if Re>MAX_REYNOLDS:
  reynolds_status="VIOLATED"
else:
  reynolds_status="SATISFIED"

#checks every v5.7 hard constraint against the chamber-regime result
#the throat stress test is intentionally not included here
velocity_constraint=v<=MAX_VELOCITY
reynolds_constraint=Re<=MAX_REYNOLDS
pressure_constraint=dP<=MAX_PRESSURE_DROP
wall_constraint=float(Twall)<=MAX_WALL_TEMPERATURE
outlet_constraint=Tout<=MAX_OUTLET_TEMPERATURE
width_constraint=(w>=MIN_WIDTH and w<=MAX_WIDTH)
height_constraint=(H>=MIN_HEIGHT and H<=MAX_HEIGHT)
aspect_constraint=(aspect_ratio>=MIN_ASPECT_RATIO and aspect_ratio<=MAX_ASPECT_RATIO)
correlation_constraint=bool(correlation_valid)

#the design is feasible only if every chamber-regime hard constraint passes
#correlation validity is included as an additional hard gate
design_feasible=all([
velocity_constraint,
reynolds_constraint,
pressure_constraint,
wall_constraint,
outlet_constraint,
width_constraint,
height_constraint,
aspect_constraint,
correlation_constraint
])

#throat stress test uses the same finalized geometry and coolant-side h
#only hgas changes between cases
#this is post-optimization only and never feeds back into design_feasible
def throat_eval(hgasthroat):
  U_t=1.0/(1.0/hgasthroat+tw/kwall+1.0/h_standard)
  qflux_t=(T_hot-Tin)*U_t
  Twall_t=T_hot-qflux_t/hgasthroat
  return qflux_t,Twall_t

qflux_throat_low,Twall_throat_low=throat_eval(hgasthroat_low)
qflux_throat_mid,Twall_throat_mid=throat_eval(hgasthroat_mid)
qflux_throat_high,Twall_throat_high=throat_eval(hgasthroat_high)

throat_ok_low=bool(Twall_throat_low<=MAX_WALL_TEMPERATURE)
throat_ok_mid=bool(Twall_throat_mid<=MAX_WALL_TEMPERATURE)
throat_ok_high=bool(Twall_throat_high<=MAX_WALL_TEMPERATURE)

print("\n========== V5.7 CONSTRAINT-AWARE DESIGN AUDIT (FINAL) ==========\n")

print("----- OPTIMIZED DESIGN -----")
print("Best Design =",bestdesign)

print("\n----- GEOMETRY -----")
print("Length L =",L)
print("Width w =",w)
print("Height H =",H)
print("Flow Area A =",A)
print("Hydraulic Diameter Dh =",Dh)
print("Surface Area As =",As)
print("Aspect Ratio =",aspect_ratio)

print("\n----- COOLANT PROPERTIES -----")
print("Density rho =",rho)
print("Specific Heat Cp =",Cp)
print("Thermal Conductivity k =",k)
print("Dynamic Viscosity mu =",mu)
print("Prandtl Number Pr =",Pr)

print("\n----- FLOW -----")
print("Total Mass Flow totalmdot =",totalmdot)
print("Number of Channels nch =",nch)
print("Per-Channel Mass Flow =",mdot_channel)
print("Coolant Velocity =",v)
print("Reynolds Number Re =",Re)

if Re<2300:
  print("Flow Regime = Laminar")
elif Re<4000:
  print("Flow Regime = Transitional")
else:
  print("Flow Regime = Turbulent")

print("Correlation Status =",correlation_status)

print("\n----- HEAT TRANSFER (coolant side) -----")
print("Nusselt Number =",Nu_standard)
print("Heat Transfer Coefficient =",h_standard)

print("\n----- PRESSURE LOSS -----")
print("Friction Factor f =",f)
print("Pressure Loss dP =",dP)

print("\n----- CHAMBER-REGIME THERMAL RESULT (what v5.2 optimized for) -----")
print("hgaschamber =",hgaschamber)
print("Peak Heat Flux qflux_chamber =",qflux_chamber)
print("Average Heat Flux qflux_avg =",qflux_avg)
print("Imposed Heat Load Q =",Q)
print("Coolant Temperature Rise =",dTcoolant)
print("Coolant Outlet Temperature =",Tout)
print("PINN Wall Temperature =",Twall)
print("Physics Wall Temperature =",Tphysics)
print("PINN vs Physics Difference =",difference)
print("PINN vs Physics Error (%) =",errorpercent)

#prints the final chamber-regime constraint results
#all of these checks contribute directly to design_feasible
print("\n----- V5.7 CONSTRAINT CHECKS (chamber regime) -----")

print("Coolant Velocity <= 100 m/s =",velocity_constraint)
print("Reynolds Number <= 3.0e6 =",reynolds_constraint)
print("Pressure Drop <= 1.8 MPa =",pressure_constraint)
print("Wall Temperature <= 825 K =",wall_constraint)
print("Outlet Temperature <= 380 K =",outlet_constraint)
print("Channel Width 0.5-2.0 mm =",width_constraint)
print("Channel Height 1.0-8.0 mm =",height_constraint)
print("Aspect Ratio 1.0-10.0 =",aspect_constraint)
print("Correlation valid (Gnielinski or Dittus-Boelter) =",correlation_constraint)

#prints warnings for any chamber-regime constraint that fails
print("\n----- CONSTRAINT WARNINGS -----")

if not velocity_constraint:
  print("WARNING: Coolant velocity exceeds 100 m/s")
if not reynolds_constraint:
  print("WARNING: Reynolds number exceeds 3.0e6")
if not pressure_constraint:
  print("WARNING: Pressure drop exceeds 1.8 MPa")
if not wall_constraint:
  print("WARNING: Wall temperature exceeds 825 K")
if not outlet_constraint:
  print("WARNING: Coolant outlet temperature exceeds 380 K")
if not width_constraint:
  print("WARNING: Channel width is outside the allowed range")
if not height_constraint:
  print("WARNING: Channel height is outside the allowed range")
if not aspect_constraint:
  print("WARNING: Channel aspect ratio is outside the allowed range")
if not correlation_constraint:
  print("WARNING: neither Gnielinski nor Dittus-Boelter is valid here")

#runs the finalized design through the throat gas-side stress-test range
#this does not change the chamber feasibility result
print("\n----- THROAT STRESS TEST (NOT used in optimization or feasibility) -----")
print("Same geometry and coolant-side h as above. Only hgas changes, from")
print("the chamber-section Bartz estimate to the throat-section estimate.")
print("This was never seen by the NN during training or by the optimizer")
print("during search -- a failure here is a finding about the design's")
print("throat behavior, not evidence the optimization above is wrong.\n")

for label,hg,qf,tw_,ok in [
    ("low  (hgas=18000 W/m2K)",hgasthroat_low,qflux_throat_low,Twall_throat_low,throat_ok_low),
    ("mid  (hgas=26500 W/m2K)",hgasthroat_mid,qflux_throat_mid,Twall_throat_mid,throat_ok_mid),
    ("high (hgas=35000 W/m2K)",hgasthroat_high,qflux_throat_high,Twall_throat_high,throat_ok_high),
]:
  status="OK" if ok else "EXCEEDS 825K"
  print(f"{label}: q''={float(qf)/1e6:.2f} MW/m2, Twall={float(tw_):.1f} K -- {status}")

#classifies the finalized design using only the chamber-regime hard constraints
print("\n----- FINAL DESIGN CLASSIFICATION -----")

if design_feasible:
  print("CHAMBER-REGIME DESIGN STATUS = FEASIBLE")
  print("All v5.7 hard constraints are satisfied in the chamber regime")
else:
  print("CHAMBER-REGIME DESIGN STATUS = INFEASIBLE")
  print("One or more v5.7 hard constraints are violated in the chamber regime")

print("\n----- CHAMBER vs THROAT CONCLUSION -----")
throat_all_ok=throat_ok_low and throat_ok_mid and throat_ok_high
if design_feasible and not throat_all_ok:
  print("Within the modeled geometry, operating conditions, heat-transfer")
  print("correlations, and hydraulic constraints, regenerative cooling was")
  print("capable of satisfying the specified wall-temperature constraint")
  print("in the chamber region, but was insufficient to maintain the same")
  print("constraint under modeled throat-level heat-transfer conditions.")
  print("This motivates investigation of supplemental throat cooling, such")
  print("as film cooling, rather than indicating an error in the")
  print("regenerative-cooling optimization itself.")
elif design_feasible and throat_all_ok:
  print("This design satisfies the wall-temperature constraint in both the")
  print("chamber regime and across the full throat h_gas stress-test range.")
else:
  print("Chamber-regime result did not satisfy all hard constraints --")
  print("revisit the v5.2 design space or optimizer objective before")
  print("drawing conclusions about throat behavior.")

print("\n(Conclusion is limited to the modeled geometry, the fixed total")
print("coolant flow assumption, the Gnielinski/Dittus-Boelter correlation")
print("range, constant coolant properties, and the Bartz-derived h_gas")
print("ranges used here -- not a general claim about regenerative cooling")
print("in liquid rocket engines.)")

#prints everything v6 needs to build the first OpenFOAM/ParaView case
#this is a handoff from the 0-D model, not a claim that the design is CFD-validated
print("\n========== V6 HANDOFF (OpenFOAM / ParaView) ==========\n")

print("----- GEOMETRY TO MESH -----")
print(f"Channel length L = {float(L):.4f} m")
print(f"Channel width w = {float(w)*1000:.4f} mm")
print(f"Channel height H = {float(H)*1000:.4f} mm")
print(f"Hydraulic diameter Dh = {float(Dh)*1000:.4f} mm")
print(f"Number of channels nch = {float(nch):.1f}")
print(f"Wall thickness (chamber liner, hot-gas side to channel) = {tw*1000:.2f} mm")

print("\n----- INLET BOUNDARY CONDITIONS -----")
print(f"Coolant inlet temperature Tin = {Tin} K")
print(f"Per-channel mass flow inlet = {float(mdot_channel):.5f} kg/s")
print(f"(equivalently, inlet velocity = {float(v):.3f} m/s for this cross-section)")

print("\n----- WALL / THERMAL BOUNDARY CONDITIONS -----")
print(f"Hot-gas-side driving temperature T_hot = {T_hot} K")
print(f"Wall thermal conductivity kwall = {kwall} W/mK")
print("Two h_gas regimes to run as SEPARATE cases (this is exactly what")
print("v6 should resolve properly with an axially-varying wall boundary")
print("instead of two fixed lumped values):")
print(f"  chamber regime: h_gas = {hgaschamber} W/m2K")
print(f"  throat regime:  h_gas = {hgasthroat_low}-{hgasthroat_high} W/m2K")

print("\n----- WHAT THIS 0-D MODEL COULD NOT RESOLVE (v6 should) -----")
print("1. Axial heat-flux profile: this model used a single fixed")
print("   AVG_FLUX_FACTOR=0.25 to approximate the chamber-average flux")
print("   relative to the throat peak. A real CFD case with the actual")
print("   chamber/throat/nozzle contour will produce a real axial q''")
print("   profile instead of this ratio assumption.")
print("2. Temperature/pressure-dependent coolant properties: rho, Cp, k,")
print("   mu were held constant. Liquid methane is transcritical in this")
print("   pressure/temperature range (critical point 190.56K, 4.599MPa)")
print("   -- Cp in particular spikes sharply near the pseudo-critical")
print("   point. OpenFOAM with a real-fluid property table (or a")
print("   NIST/CoolProp-derived lookup) should replace these constants.")
print("3. h_gas itself: both chamber and throat values here are Bartz-")
print("   correlation estimates from a representative operating class,")
print("   not derived from the actual simulated combustion flow field.")
print("   A conjugate heat transfer (CHT) simulation in OpenFOAM removes")
print("   the need for an assumed h_gas entirely.")
print("4. This is a single representative design point, not a full")
print("   validated design -- treat the v5.2-v5.7 result as the starting")
print("   geometry for CFD, not as a final answer.")

print("\n========== END V5.7 FINAL AUDIT ==========\n")
