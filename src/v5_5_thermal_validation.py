"""
v5.5 — Thermally Consistent Cooling Validation

Loads the v5.1 cooling surrogate and v5.2 optimized design,
then validates the design using a consistent coolant energy balance
and compares the PINN prediction against the physics model.
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

checkpoint=torch.load("PINN_Modelv5_1_complete.pth")

model.load_state_dict(checkpoint["model"])

xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]

ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]

model.eval()

print("Complete Model v5.1 loaded")

designcheckpoint=torch.load(
"PINN_Modelv5_2_optimized_design.pth"
)

bestdesign=designcheckpoint["bestdesign"]

print("Optimized design loaded")
print(bestdesign)

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
mu=1.1e-5
k=0.1

#operating conditions
Tin=150
qflux=5e6

#chamber conditions
T_hot=3500

print("Properties Defined")

#cross sectional flow area
A=w*H

#hydraulic diameter
Dh=(2*w*H)/(w+H)

#coolant velocity
v=mdot/(rho*A)

#reynolds number
Re=(rho*v*Dh)/mu

#prandtl number
Pr=(Cp*mu)/k

#nusselt number
Nu=0.023*(Re**0.8)*(Pr**0.4)

#heat transfer coefficient
h=(Nu*k)/Dh

#darcy friction factor
f=0.3164*(Re**(-0.25))

#pressure loss
dP=f*(L/Dh)*(rho*v**2/2)

#cooling channel surface area
As=2*L*(w+H)

#calculate actual heat load from heat flux
Q=qflux*As

#calculate how much the coolant temperature increases
dTcoolant=Q/(mdot*Cp)

#calculate coolant outlet temperature
Tout=Tin+dTcoolant

#normalize design so the NN can use it
designnorm=(bestdesign-xmean)/xstd

with torch.no_grad():
  Twallnorm=model(designnorm)

#convert normalized prediction back to Kelvin
Twall=Twallnorm*ystd+ymean

#calculate physics wall temperature using the same heat flux
Tphysics=Tin+qflux/h

#calculate difference between PINN and physics
Tdifference=torch.abs(Twall-Tphysics)

#calculate percent error between PINN and physics
Terrorpercent=(Tdifference/Tphysics)*100

#calculate maximum heat transfer capacity of the channel
Qcapacity=h*As*(T_hot-Twall)

#calculate how much capacity is left above the required heat load
capacitymargin=Qcapacity-Q

#check that heat entering coolant matches heat load
Qcheck=mdot*Cp*(Tout-Tin)

#check if design has enough thermal capacity
thermalcapacitysufficient=Qcapacity>=Q

#check if pressure loss stays within the limit
dPmax=3e6
pressurelosswithinlimit=dP<=dPmax

print("\n========== v5.5 THERMAL VALIDATION ==========\n")

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
print("Mass Flow mdot =",mdot)
print("Velocity =",v)
print("Reynolds Number Re =",Re)

print("\n----- HEAT TRANSFER -----")
print("Prandtl Number Pr =",Pr)
print("Nusselt Number Nu =",Nu)
print("Heat Transfer Coefficient h =",h)

print("\n----- PRESSURE LOSS -----")
print("Friction Factor f =",f)
print("Pressure Loss dP =",dP)
print("Pressure Loss Limit =",dPmax)
print("Pressure Loss Within Limit =",pressurelosswithinlimit)

print("\n----- THERMAL ENERGY BALANCE -----")
print("Heat Flux qflux =",qflux)
print("Imposed Heat Load Q =",Q)
print("Coolant Temperature Rise =",dTcoolant)
print("Coolant Outlet Temperature =",Tout)

print("\n----- ENERGY BALANCE CHECK -----")
print("Heat Load Q =",Q)
print("Energy Balance Q =",Qcheck)
print("Energy Balance Difference =",torch.abs(Q-Qcheck))

print("\n----- WALL TEMPERATURE -----")
print("PINN Wall Temperature =",Twall)
print("Physics Wall Temperature =",Tphysics)
print("PINN vs Physics Difference =",Tdifference)
print("PINN vs Physics Error (%) =",Terrorpercent)

print("\n----- HEAT TRANSFER CAPACITY -----")
print("Heat Transfer Capacity =",Qcapacity)
print("Thermal Capacity Margin =",capacitymargin)
print("Thermal Capacity Sufficient =",thermalcapacitysufficient)

print("\n========== END VALIDATION ==========\n")
