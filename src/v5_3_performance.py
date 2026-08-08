"""
v5.3 — Full Cooling Performance Evaluation

Loads the v5.1 cooling surrogate and the optimized v5.2 cooling
channel design, then evaluates heat transfer, cooling effectiveness,
pressure loss, heat removal, and physics-based wall temperature.
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
model=CoolingNN()  #recreated trained NN blueprint used in v5.1 & v5.2

checkpoint=torch.load("PINN_Modelv5_1_complete.pth")
model.load_state_dict(checkpoint["model"])

xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]

ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]

model.eval()
print("Model loaded")

designcheckpoint=torch.load(
    "PINN_Modelv5_2_optimized_design.pth"
)

bestdesign=designcheckpoint["bestdesign"]

print(bestdesign)

L=bestdesign[0]
w=bestdesign[1]
H=bestdesign[2]
mdot=bestdesign[3]

print("L = ",L)
print("w = ",w)
print("H = ",H)
print("mdot = ",mdot)

#coolant properties
rho=420          #density (kg/m^3)
Cp=3500          #specific heat (J/kg*K)
mu=1.1e-5        #dynamic viscosity (Pa*s)
k=0.1            #thermal conductivity (W/m*K)

#operating conditions
qflux=5e6
Tin=150          #coolant inlet temp (K)

#chamber wall/material properties
T_hot=3500        #combustion chamber temp (K)
Tmax=4000        #max material operating temp (K)
Rth=0.001        #thermal resistance (K/W)
thickness=0.005  #wall thickness (m)

print("Properties Defined")

Dh=(2*w*H)/(w+H)
print("Hydraulic Diameter (m) = ", Dh)

#cross sectional flow area
A=w*H

#coolant velocity
v=mdot/(rho*A)

print("Flow Area (m^2) = ", A)
print("Coolant Velocity (m/s) = ", v)

Re=(rho*v*Dh)/mu

print("Reynolds Number = ", Re)

#calculate prandtl number
Pr=(Cp*mu)/k

#calculate nusselt number
Nu=0.023*(Re**0.8)*(Pr**0.4)

print("Prandtl Number = ", Pr)
print("Nusselt Number = ", Nu)

h=(Nu*k)/Dh

print("Hydraulic Diameter:", Dh)
print("Velocity:", v)
print("Re:", Re)
print("Pr:", Pr)
print("Nu:", Nu)
print("Coolant k:", k)
print("Heat Transfer Coefficient = ", h)

#cooling channel surface area
As=2*L*(w+H)

#get predicted wall temp from NN
designnorm=(bestdesign-xmean)/xstd
Twallnorm=model(designnorm)

print("Normalized Twall prediction:", Twallnorm)

Twall=Twallnorm*ystd+ymean

print("Unscaled Twall prediction:", Twall)
print("Expected training range:",
      ymean - 3*ystd,
      "to",
      ymean + 3*ystd)

#heat removed
Q=h*As*(T_hot-Twall)

print("Surface Area (m^2) = ", As)
print("Wall Temperature (K) = ", Twall)
print("Heat Removed (W) = ",Q)

#cooling effectiveness
epsilon=(T_hot-Twall)/(T_hot-Tin)

print("Cooling Effectiveness = ", epsilon)

#calculate darcy friction factor
f=0.3164*(Re**(-0.25))

#calculate pressure loss
dP=f*(L/Dh)*(rho*v**2/2)

print("Friction Factor = ", f)
print("Pressure Loss (Pa) = ", dP)
print("Pressure Loss (mPa) = ", dP/1e6)

print("\n========== CFD/PINN DIAGNOSTIC REPORT ==========\n")

#design variables
print("----- DESIGN -----")
print("Best Design =", bestdesign)

#geometry
print("\n----- GEOMETRY -----")
print("Length L =", L)
print("Width w =", w)
print("Height H =", H)
print("Surface Area As =", As)
print("Hydraulic Diameter Dh =", Dh)
print("Channel Cross Section Area =", A)

#coolant properties
print("\n----- COOLANT PROPERTIES -----")
print("Density rho =", rho)
print("Specific Heat Cp =", Cp)
print("Thermal Conductivity k =", k)
print("Dynamic Viscosity mu =", mu)
print("Prandtl Number Pr =", Pr)

#flow
print("\n----- FLOW -----")
print("Mass Flow mdot =", mdot)
print("Velocity =", v)
print("Reynolds Number Re =", Re)

#heat transfer
print("\n----- HEAT TRANSFER -----")
print("Nusselt Number Nu =", Nu)
print("Heat Transfer Coefficient h =", h)

#PINN prediction
print("\n----- PINN OUTPUT -----")
print("Normalized Wall Temperature =", Twallnorm)
print("Wall Temperature Twall =", Twall)

#operating conditions
print("\n----- OPERATING CONDITIONS -----")
print("Hot Temperature =", T_hot)
print("Coolant Inlet Temperature =", Tin)
print("Heat Flux =", qflux)

#cooling calculation
print("\n----- COOLING PERFORMANCE -----")
print("Pressure Loss =",dP)
print("Heat Removed Q =", Q)

try:
    print("Cooling Effectiveness =", epsilon)
except:
    print("Cooling Effectiveness not calculated")

print("\n========== END REPORT ==========\n")
tphysics=Tin+qflux/h

Qphysics=qflux*As

print("Physics Wall Temperature =",tphysics)
print("Physics Heat Removed =",Qphysics)
print("PINN vs Physics Difference =",abs(Twall-tphysics))

print(bestdesign)
