"""v5.3_FullCoolingPerformance- corrected

Tests the optimized cooling channel design from v5.2 under chamber
and throat conditions to see how the design performs.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
    return self.network(x)
model=CoolingNN()

#load the trained v5.1 model
checkpoint=torch.load("PINN_Modelv5_1_complete.pth")
model.load_state_dict(checkpoint["model"])

#load the input normalization values used during training
xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]

#load the output normalization values used during training
ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]

model.eval()
print("Model loaded")

#load the optimized design found by v5.2
designcheckpoint=torch.load(
    "PINN_Modelv5_2_optimized_design.pth"
)

bestdesign=designcheckpoint["bestdesign"]

print(bestdesign)

#the design has length, width, height, and number of channels
L=bestdesign[0]
w=bestdesign[1]
H=bestdesign[2]
nch=bestdesign[3]

print("L = ",L)
print("w = ",w)
print("H = ",H)
print("nch = ",nch)

#liquid methane properties used in the corrected model
rho=422          #density (kg/m^3)
Cp=3480          #specific heat (J/kg-K)
mu=1.2e-4        #dynamic viscosity (Pa-s)
k=0.187          #thermal conductivity (W/m-K)
Pr=(Cp*mu)/k     #calculate Pr from the other methane properties

#wall properties for the chamber liner
kwall=320   #wall thermal conductivity (W/m-K)
tw=0.001    #wall thickness (m)

#operating temperatures
Tin=115      #coolant inlet temperature (K)
             #corrected from 150K to a more realistic subcooled methane range
T_hot=3500   #hot-gas-side temperature (K)

#total coolant flow is fixed and split between the channels
totalmdot=20

#use an average heat flux for the outlet temperature calculation
#instead of assuming the peak heat flux happens over the entire channel
avgfluxfactor=0.25

#chamber h_gas is the condition used in v5.1 and v5.2
#the throat values are much higher and are only tested here after
#the optimization is finished
hgaschamber=5500          #chamber-section h_gas (W/m^2-K)
hgasthroat_low=18000      #low throat h_gas value
hgasthroat_mid=26500      #middle throat h_gas value
hgasthroat_high=35000     #high throat h_gas value

print("Properties defined")

#calculate the channel cross-sectional area
A=w*H

#calculate hydraulic diameter for the rectangular channel
Dh=(2*w*H)/(w+H)

#calculate mass flow in one channel from the total mass flow
#then use it to calculate coolant velocity
mdotchannel=totalmdot/nch
v=mdotchannel/(rho*A)

#calculate Reynolds number for the channel flow
Re=(rho*v*Dh)/mu

#use Gnielinski as the main heat transfer correlation
#Dittus-Boelter is kept as a fallback for cases outside Gnielinski's range
f=(0.79*torch.log(torch.clamp(Re,min=1.0))-1.64)**-2
nu_gnielinski=(f/8)*(Re-1000)*Pr/(1+12.7*torch.sqrt(f/8)*(Pr**(2/3)-1))
nu_db=0.023*(Re**0.8)*(Pr**0.4)
gnielinski_valid=(Re>=3000)&(Re<=5e6)&(Pr>=0.5)&(Pr<=2000)
db_valid=(Re>=10000)&(Pr>=0.6)&(Pr<=160)
corrvalid=gnielinski_valid|db_valid
nu=torch.where(gnielinski_valid,nu_gnielinski,torch.where(db_valid,nu_db,torch.tensor(float("nan"))))

#calculate the coolant-side heat transfer coefficient
#this stays the same for both chamber and throat tests because
#the coolant flow and channel geometry do not change
h=(nu*k)/Dh

print("Hydraulic Diameter:", Dh)
print("Velocity:", v)
print("Re:", Re)
print("Pr:", Pr)
print("Nu (gnielinski):", nu)
print("Coolant k:", k)
print("Heat Transfer Coefficient h =", h)

#calculate the total cooling channel surface area
As=2*L*(w+H)

#use the trained PINN to predict wall temperature for the optimized design
#the PINN was only trained using chamber conditions
designnorm=(bestdesign-xmean)/xstd
Twallnorm=model(designnorm)
print("Normalized Twall prediction:", Twallnorm)
Twall_pinn_chamber=Twallnorm*ystd+ymean
print("Unscaled Twall prediction (chamber, PINN):", Twall_pinn_chamber)
print("Expected training range:", ymean-3*ystd, "to", ymean+3*ystd)

#calculate the wall temperature and heat flux under chamber conditions
#these are the same conditions used during the v5.2 optimization
U_chamber=1.0/(1.0/hgaschamber+tw/kwall+1.0/h)
qflux_chamber=(T_hot-Tin)*U_chamber
Twall_chamber=T_hot-qflux_chamber/hgaschamber

#use the average heat flux to calculate coolant outlet temperature
Q_chamber=qflux_chamber*avgfluxfactor*As
Tout_chamber=Tin+Q_chamber/(mdotchannel*Cp)

#calculate cooling effectiveness
epsilon_chamber=(T_hot-Twall_chamber)/(T_hot-Tin)

#calculate pressure loss using the same friction factor
f_pressure=(0.79*torch.log(torch.clamp(Re,min=1.0))-1.64)**-2
dP=f_pressure*(L/Dh)*(rho*v**2/2)

print("Surface Area (m^2) = ", As)
print("Wall Temperature, chamber regime (K) = ", Twall_chamber)
print("Heat flux, chamber regime (MW/m2) = ", qflux_chamber/1e6)
print("Cooling Effectiveness (chamber) = ", epsilon_chamber)
print("Friction Factor = ", f_pressure)
print("Pressure Loss (Pa) = ", dP)
print("Pressure Loss (MPa) = ", dP/1e6)

#test the same design with the higher gas-side heat transfer values
#only hgas changes between these tests
def throat_eval(hgasthroat):
  U_t=1.0/(1.0/hgasthroat+tw/kwall+1.0/h)
  qflux_t=(T_hot-Tin)*U_t
  Twall_t=T_hot-qflux_t/hgasthroat
  return qflux_t,Twall_t

qflux_throat_low,Twall_throat_low=throat_eval(hgasthroat_low)
qflux_throat_mid,Twall_throat_mid=throat_eval(hgasthroat_mid)
qflux_throat_high,Twall_throat_high=throat_eval(hgasthroat_high)

MAXWALLTEMP=825.0

print("\n========== CFD/PINN DIAGNOSTIC REPORT ==========\n")

print("----- DESIGN (chamber-optimized in v5.2) -----")
print("Best Design =", bestdesign)

print("\n----- GEOMETRY -----")
print("Length L =", L)
print("Width w =", w)
print("Height H =", H)
print("Number of channels nch =", nch)
print("Surface Area As =", As)
print("Hydraulic Diameter Dh =", Dh)
print("Channel Cross Section Area =", A)

print("\n----- COOLANT PROPERTIES -----")
print("Density rho =", rho)
print("Specific Heat Cp =", Cp)
print("Thermal Conductivity k =", k)
print("Dynamic Viscosity mu =", mu)
print("Prandtl Number Pr =", Pr)

print("\n----- FLOW -----")
print("Total mass flow totalmdot =", totalmdot)
print("Per-channel mass flow mdotchannel =", mdotchannel)
print("Velocity =", v)
print("Reynolds Number Re =", Re)
print("Correlation valid =", corrvalid)

print("\n----- HEAT TRANSFER (coolant side, same for both regimes) -----")
print("Nusselt Number Nu =", nu)
print("Heat Transfer Coefficient h =", h)

print("\n----- PINN OUTPUT (chamber regime -- the only regime it was trained on) -----")
print("Normalized Wall Temperature =", Twallnorm)
print("Wall Temperature Twall (PINN, chamber) =", Twall_pinn_chamber)

print("\n----- CHAMBER-REGIME RESULT (what v5.2 optimized for) -----")
print("hgaschamber =", hgaschamber, "W/m2K")
print("Heat flux =", qflux_chamber.item()/1e6, "MW/m2")
print("Wall Temperature (physics, chamber) =", Twall_chamber.item(), "K")
print("Coolant Outlet Temperature =", Tout_chamber.item(), "K")
print("Cooling Effectiveness =", epsilon_chamber.item())
print("Wall temp <= 825K ?", bool(Twall_chamber.item()<=MAXWALLTEMP))

print("\n----- THROAT STRESS TEST (NOT used in optimization) -----")
print("This design was never evaluated against throat conditions during")
print("v5.1 training or v5.2 optimization. The result below is a pure")
print("post-hoc check -- if it fails, that's a finding, not a bug.\n")
for label,hg,qf,tw_ in [
    ("low  (hgas=18000)",hgasthroat_low,qflux_throat_low,Twall_throat_low),
    ("mid  (hgas=26500)",hgasthroat_mid,qflux_throat_mid,Twall_throat_mid),
    ("high (hgas=35000)",hgasthroat_high,qflux_throat_high,Twall_throat_high),
]:
  status="OK" if tw_.item()<=MAXWALLTEMP else "EXCEEDS 825K"
  print(f"{label}: q''={qf.item()/1e6:.2f} MW/m2, Twall={tw_.item():.1f} K -- {status}")

#compare the chamber result against the three throat stress tests
chamber_ok=bool(Twall_chamber.item()<=MAXWALLTEMP)
throat_ok_low=bool(Twall_throat_low.item()<=MAXWALLTEMP)
throat_ok_mid=bool(Twall_throat_mid.item()<=MAXWALLTEMP)
throat_ok_high=bool(Twall_throat_high.item()<=MAXWALLTEMP)

print("\n----- CHAMBER vs THROAT COMPARISON -----")
print("Chamber regime satisfies 825K constraint:", chamber_ok)
print("Throat regime (low h_gas) satisfies 825K constraint:", throat_ok_low)
print("Throat regime (mid h_gas) satisfies 825K constraint:", throat_ok_mid)
print("Throat regime (high h_gas) satisfies 825K constraint:", throat_ok_high)

print("\n----- CONCLUSION -----")
if chamber_ok and not (throat_ok_low and throat_ok_mid and throat_ok_high):
  print("Within the modeled geometry, operating conditions, heat-transfer")
  print("correlations, and hydraulic constraints, regenerative cooling was")
  print("capable of satisfying the specified wall-temperature constraint")
  print("in the chamber region, but was insufficient to maintain the same")
  print("constraint under modeled throat-level heat-transfer conditions.")
  print("This motivates investigation of supplemental throat cooling, such")
  print("as film cooling, rather than indicating an error in the")
  print("regenerative-cooling optimization itself.")
elif chamber_ok and throat_ok_low and throat_ok_mid and throat_ok_high:
  print("Within the modeled geometry, operating conditions, heat-transfer")
  print("correlations, and hydraulic constraints, this regenerative-cooling")
  print("design satisfies the specified wall-temperature constraint in both")
  print("the chamber and the full stress-tested throat h_gas range.")
else:
  print("Chamber-regime result did not satisfy the wall-temperature")
  print("constraint -- this indicates the optimizer itself should be")
  print("re-run or the design space reconsidered, since v5.2's hard")
  print("constraint check should have excluded this design.")

print("\n(Conclusion is limited to the modeled geometry, the fixed")
print("total coolant flow and channel-count assumptions, the Gnielinski/")
print("Dittus-Boelter correlation range, the assumed constant coolant")
print("properties, and the Bartz-derived h_gas ranges used here -- not a")
print("general claim about regenerative cooling in liquid rocket engines.)")

print("\n========== END REPORT ==========\n")
