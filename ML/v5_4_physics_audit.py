# -*- coding: utf-8 -*-
"""v5.4—physics audit and PINN validation

compares the trained PINN's chamber-regime wall temperature against the
corrected physics-based resistance-network result and validates the
heat-removal calculation.
"""

import torch
import torch.nn as nn

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

#load the v5.1 trained model and its normalization values so this file
#uses the exact same trained surrogate and input/output scaling as v5.1
checkpoint=torch.load("PINN_Modelv5_1_complete.pth")

model.load_state_dict(checkpoint["model"])

xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]

ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]

model.eval()

print("Complete Model v5.1 loaded")

#load the optimized design selected by v5.2 so the physics validation
#is performed on the same design that was actually chosen by the optimizer
designcheckpoint=torch.load(
"PINN_Modelv5_2_optimized_design.pth"
)

bestdesign=designcheckpoint["bestdesign"]

print("Optimized design loaded")
print(bestdesign)

#extract the four design variables from the v5.2 optimized design
#the fourth variable is n_channels, replacing the old independent mdot
L=bestdesign[0]
w=bestdesign[1]
H=bestdesign[2]
nch=bestdesign[3]

print("L =",L)
print("w =",w)
print("H =",H)
print("nch =",nch)

#use the same corrected liquid methane properties introduced in v5.1
#and carried consistently through the downstream validation files
rho=422
Cp=3480
mu=1.2e-4
k=0.187
Pr=(Cp*mu)/k

#use the same chamber liner properties as the corrected resistance network
kwall=320
tw=0.001

#corrected subcooled liquid methane inlet temperature
Tin=115

#hot-gas-side driving temperature used by the corrected resistance network
T_hot=3500

#fixed total coolant flow, with per-channel flow derived from n_channels
totalmdot=20

#use the same chamber-section gas-side film coefficient as v5.1/v5.2
#this validation is intentionally limited to the regime the PINN was
#trained on rather than evaluating the separate throat stress-test case
hgaschamber=5500

print("Properties Defined")

#calculate channel flow area from the optimized geometry
A=w*H

#calculate hydraulic diameter for the rectangular channel
Dh=(2*w*H)/(w+H)

#derive per-channel mass flow and coolant velocity from total flow,
#channel count, and channel geometry instead of using an independent mdot
mdotchannel=totalmdot/nch
v=mdotchannel/(rho*A)

#calculate reynolds number using the corrected methane viscosity
Re=(rho*v*Dh)/mu

#use gnielinski with the petukhov friction factor as the primary
#heat-transfer correlation, with dittus-boelter only as a fallback
f=(0.79*torch.log(torch.clamp(Re,min=1.0))-1.64)**-2
nu_gnielinski=(f/8)*(Re-1000)*Pr/(1+12.7*torch.sqrt(f/8)*(Pr**(2/3)-1))
nu_db=0.023*(Re**0.8)*(Pr**0.4)
gnielinski_valid=(Re>=3000)&(Re<=5e6)&(Pr>=0.5)&(Pr<=2000)
db_valid=(Re>=10000)&(Pr>=0.6)&(Pr<=160)
corrvalid=gnielinski_valid|db_valid
nu=torch.where(gnielinski_valid,nu_gnielinski,torch.where(db_valid,nu_db,torch.tensor(float("nan"))))

#calculate the coolant-side heat transfer coefficient from the selected
#nusselt correlation and corrected methane thermal conductivity
h=(nu*k)/Dh

#use the same petukhov friction factor as the gnielinski correlation
#so pressure loss and heat transfer are based on the same friction model
dP=f*(L/Dh)*(rho*v**2/2)

#calculate the total wetted cooling-channel surface area
As=2*L*(w+H)

#calculate the physics reference wall temperature using the corrected
#three-resistance network: hot gas -> gas-side film -> wall conduction
#-> coolant-side convection. qflux is now derived rather than imposed.
U=1.0/(1.0/hgaschamber+tw/kwall+1.0/h)
qflux=(T_hot-Tin)*U
Tphysics=T_hot-qflux/hgaschamber

#normalize the optimized design using the exact v5.1 training
#distribution, then run it through the loaded v5.1 surrogate
designnorm=(bestdesign-xmean)/xstd
with torch.no_grad():
  Twallnorm=model(designnorm)

#convert the normalized PINN prediction back into Kelvin
Twall=Twallnorm*ystd+ymean

#calculate heat removed directly from the corrected physics heat flux
Qphysics=qflux*As

#derive the heat flux implied by the PINN's predicted wall temperature
#using only the hot-gas-side film and wall-conduction resistances. this
#replaces the old physically inconsistent h*As*(T_hot-Twall) calculation
Uwall=1.0/(1.0/hgaschamber+tw/kwall)
qflux_pinn_implied=(T_hot-Twall)*Uwall
Qpinn=qflux_pinn_implied*As

#calculate absolute temperature disagreement and percentage error between
#the PINN prediction and the corrected physics reference
Tdifference=torch.abs(Twall-Tphysics)
Terrorpercent=(Tdifference/Tphysics)*100

print("\n========== v5.4 PHYSICS AUDIT (chamber regime) ==========\n")

print("----- DESIGN -----")
print("Best Design =",bestdesign)

print("\n----- GEOMETRY -----")
print("Length L =",L)
print("Width w =",w)
print("Height H =",H)
print("Flow Area A =",A)
print("Hydraulic Diameter Dh =",Dh)
print("Surface Area As =",As)

print("\n----- FLOW -----")
print("Total mass flow totalmdot =",totalmdot)
print("Per-channel mass flow mdotchannel =",mdotchannel)
print("Velocity =",v)
print("Reynolds Number Re =",Re)
print("Correlation valid =",corrvalid)

print("\n----- HEAT TRANSFER -----")
print("Prandtl Number Pr =",Pr)
print("Nusselt Number Nu (gnielinski) =",nu)
print("Heat Transfer Coefficient h =",h)

print("\n----- PRESSURE LOSS -----")
print("Friction Factor f =",f)
print("Pressure Loss dP =",dP)

print("\n----- TEMPERATURE (chamber regime) -----")
print("hgaschamber =",hgaschamber)
print("Heat flux qflux =",qflux)
print("PINN Wall Temperature =",Twall)
print("Physics Wall Temperature =",Tphysics)
print("PINN vs Physics Difference =",Tdifference)
print("PINN vs Physics Error (%) =",Terrorpercent)

print("\n----- HEAT REMOVAL -----")
print("PINN Heat Removed =",Qpinn)
print("Physics Heat Removed =",Qphysics)

print("\n========== END PHYSICS AUDIT ==========\n")
