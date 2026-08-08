"""
v5.6 — Physical Realism Audit

Loads the v5.1 cooling surrogate and v5.2 optimized design
to evaluate flow conditions, heat transfer, pressure loss,
thermal performance, and agreement between PINN and physics-based
wall temperature predictions.
"""
import torch
import torch.nn as nn

class CoolingNN(nn.Module):
  def __init__(self):
    super().__init__()
#recreates the same NN structure used during v5.1 training
    self.network=nn.Sequential(
    nn.Linear(4,32),
    nn.ReLU(),

    nn.Linear(32,32),
    nn.ReLU(),

    nn.Linear(32,1)
)
  def forward(self,x):
#passes the normalized design through the trained NN
    return self.network(x)

model=CoolingNN()
tcoolant=150

#loads the trained v5.1 model and normalization values
checkpoint=torch.load("PINN_Modelv5_1_complete.pth")

model.load_state_dict(checkpoint["model"])

#loads the same input normalization used during training
xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]

#loads the output normalization used during training
ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]

model.eval()

print("Complete Model v5.1 loaded")

#loads the optimized channel design found in v5.2
designcheckpoint=torch.load(
"PINN_Modelv5_2_optimized_design.pth"
)

bestdesign=designcheckpoint["bestdesign"]

print("Optimized design loaded")
print(bestdesign)

#extracts the optimized design variables
L=bestdesign[0]
w=bestdesign[1]
H=bestdesign[2]
mdot=bestdesign[3]

print("L =",L)
print("w =",w)
print("H =",H)
print("mdot =",mdot)

#coolant properties
rho=420
Cp=3500
k=0.1
mu=1.1e-5

#operating conditions
qflux=5e6
Tin=150

#chamber wall/material properties
T_hot=3500

print("Properties Defined")

#calculates the hydraulic diameter used for flow and heat transfer
Dh=(2*w*H)/(w+H)

#calculates the channel area available for coolant flow
A=w*H

#calculates coolant velocity from mass flow and channel area
v=mdot/(rho*A)

#checks the flow regime using the reynolds number
Re=(rho*v*Dh)/mu

#calculates the prandtl number from the coolant properties
Pr=(Cp*mu)/k

#estimates turbulent heat transfer using the dittus-boelter correlation
Nu=0.023*(Re**0.8)*(Pr**0.4)

#converts nusselt number into the heat transfer coefficient
h=(Nu*k)/Dh

#calculates the wetted channel surface area used for heat transfer
As=2*L*(w+H)

#estimates friction factor for pressure loss
f=0.3164*(Re**(-0.25))

#calculates pressure loss through the cooling channel
dP=f*(L/Dh)*(rho*v**2/2)

#calculates the total heat load imposed on the cooling channel
Q=qflux*As

#calculates how much the coolant temperature rises after absorbing heat
dTcoolant=Q/(mdot*Cp)

#calculates coolant outlet temperature for the energy balance check
Tout=Tin+dTcoolant

#gets predicted wall temp from NN

#normalizes the optimized design using the same scaling as v5.1
designnorm=(bestdesign-xmean)/xstd

with torch.no_grad():
  Twallnorm=model(designnorm)

#converts the NN output back into kelvin
Twall=Twallnorm*ystd+ymean

#calculates wall temperature directly from the physics model
tphysics=Tin+qflux/h

#compares the PINN prediction against the physics prediction
difference=torch.abs(Twall-tphysics)

#converts the temperature difference into a percentage error
errorpercent=(difference/tphysics)*100

print("\n========== v5.6 PHYSICAL REALISM AUDIT ==========\n")

print("----- DESIGN -----")
print("Best Design =",bestdesign)

print("\n----- GEOMETRY -----")
print("Length L =",L)
print("Width w =",w)
print("Height H =",H)
print("Flow Area A =",A)
print("Hydraulic Diameter Dh =",Dh)
print("Surface Area As =",As)

print("\n----- COOLANT PROPERTIES -----")
print("Density rho =",rho)
print("Specific Heat Cp =",Cp)
print("Thermal Conductivity k =",k)
print("Dynamic Viscosity mu =",mu)
print("Prandtl Number Pr =",Pr)

print("\n----- FLOW -----")
print("Mass Flow mdot =",mdot)
print("Velocity =",v)
print("Reynolds Number Re =",Re)

if Re<2300:
  print("Flow Regime = Laminar")
elif Re<4000:
  print("Flow Regime = Transitional")
else:
  print("Flow Regime = Turbulent")

print("\n----- HEAT TRANSFER -----")
print("Nusselt Number Nu =",Nu)
print("Heat Transfer Coefficient h =",h)

print("\n----- PRESSURE LOSS -----")
print("Friction Factor f =",f)
print("Pressure Loss dP =",dP)

print("\n----- THERMAL ENERGY BALANCE -----")
print("Heat Flux qflux =",qflux)
print("Imposed Heat Load Q =",Q)
print("Coolant Temperature Rise =",dTcoolant)
print("Coolant Outlet Temperature =",Tout)

print("\n----- WALL TEMPERATURE -----")
print("PINN Wall Temperature =",Twall)
print("Physics Wall Temperature =",tphysics)
print("PINN vs Physics Difference =",difference)
print("PINN vs Physics Error (%) =",errorpercent)

print("\n----- MODEL ASSUMPTIONS -----")

#flags conditions that may reduce physical realism
if Re>1e6:
  print("WARNING: Reynolds number is extremely high")

if v>100:
  print("WARNING: Coolant velocity exceeds 100 m/s")

if Pr<0.5:
  print("WARNING: Prandtl number is below 0.5")

#lists the major physics assumptions being audited
print("Dittus-Boelter correlation = Used")
print("Darcy-Weisbach pressure loss = Used")
print("Coolant properties = Constant")
print("Property variation with temperature = Not modeled")

print("\n----- AUDIT SUMMARY -----")

#checks whether the NN agrees closely with the physics model
if errorpercent<5:
  print("PINN vs physics agreement = Within 5%")
else:
  print("PINN vs physics agreement = Greater than 5%")

#checks whether the optimized design stays within the pressure constraint
if dP<=3e6:
  print("Pressure Loss Constraint = Satisfied")
else:
  print("Pressure Loss Constraint = Violated")

print("\n========== END PHYSICAL REALISM AUDIT ==========\n")
