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

#load the trained v5.1 model and its normalization values
checkpoint=torch.load("PINN_Modelv5_1_complete.pth")

model.load_state_dict(checkpoint["model"])

xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]

ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]

model.eval()

print("Complete Model v5.1 loaded")

#load the optimized channel design from v5.2
designcheckpoint=torch.load(
"PINN_Modelv5_2_optimized_design.pth"
)

bestdesign=designcheckpoint["bestdesign"]

print("Optimized design loaded")
print(bestdesign)

#extract the optimized design variables
#the 4th design variable is now nch instead of the old independently
#sampled mdot variable
L=bestdesign[0]
w=bestdesign[1]
H=bestdesign[2]
nch=bestdesign[3]

print("L =",L)
print("w =",w)
print("H =",H)
print("nch =",nch)

#coolant properties -- corrected liquid methane values used consistently
#throughout v5.1-v5.7
rho=422
Cp=3480
k=0.187
mu=1.2e-4
Pr=(Cp*mu)/k   #derive Pr from the corrected properties instead of using
               #the old hardcoded value

#wall (chamber liner) properties
kwall=320
tw=0.001

#operating conditions
Tin=115   #corrected from 150K to match the subcooled liquid methane
          #inlet range used throughout the corrected model

#chamber wall/material properties
T_hot=3500

#total coolant mass flow -- fixed system-level input and split across
#the optimized number of channels
totalmdot=20

#average-to-peak flux ratio -- peak flux is used for the local wall
#temperature calculation while the average flux is used for the coolant
#energy balance
avgfluxfactor=0.25

#gas-side film coefficient -- chamber regime used throughout v5.1-v5.6
#the throat is treated separately as a stress test in v5.3
hgaschamber=5500

print("Properties Defined")

#calculate hydraulic diameter used for flow and heat transfer
Dh=(2*w*H)/(w+H)

#calculate channel cross-sectional area
A=w*H

#derive per-channel flow from totalmdot/nch instead of independently
#sampling mdot like the original model did
mdotchannel=totalmdot/nch
v=mdotchannel/(rho*A)

#calculate reynolds number using the corrected viscosity and derived
#per-channel flow
Re=(rho*v*Dh)/mu

#calculate the prandtl number from the corrected coolant properties
#(already calculated above, kept here conceptually for the audit)

#gnielinski is now the primary correlation with petukhov friction factor.
#dittus-boelter is only kept as a fallback when its own validity range
#is satisfied instead of being applied blindly like before.
f=(0.79*torch.log(torch.clamp(Re,min=1.0))-1.64)**-2
nu_gnielinski=(f/8)*(Re-1000)*Pr/(1+12.7*torch.sqrt(f/8)*(Pr**(2/3)-1))
nu_db=0.023*(Re**0.8)*(Pr**0.4)
gnielinski_valid=(Re>=3000)&(Re<=5e6)&(Pr>=0.5)&(Pr<=2000)
db_valid=(Re>=10000)&(Pr>=0.6)&(Pr<=160)
corrvalid=gnielinski_valid|db_valid
nu=torch.where(gnielinski_valid,nu_gnielinski,torch.where(db_valid,nu_db,torch.tensor(float("nan"))))

#convert nusselt number into the coolant-side heat transfer coefficient
h=(nu*k)/Dh

#calculate the wetted channel surface area used for heat transfer
As=2*L*(w+H)

#use the same petukhov friction factor from above for pressure loss.
#the old model used a separate mismatched blasius-style factor.
dP=f*(L/Dh)*(rho*v**2/2)

#hot gas -> wall conduction -> coolant resistance network
#qflux emerges from the resistance network instead of being imposed as
#a fixed heat flux like in the old model
U=1.0/(1.0/hgaschamber+tw/kwall+1.0/h)
qflux=(T_hot-Tin)*U

#calculate the total heat load using the average heat flux instead of
#applying the local peak flux uniformly across the entire channel
qfluxavg=qflux*avgfluxfactor
Q=qfluxavg*As

#calculate coolant temperature rise from the corrected heat load and
#derived per-channel mass flow
dTcoolant=Q/(mdotchannel*Cp)

#calculate coolant outlet temperature
Tout=Tin+dTcoolant

#normalize the optimized design using the same normalization values
#stored with the v5.1 trained model
designnorm=(bestdesign-xmean)/xstd

with torch.no_grad():
  Twallnorm=model(designnorm)

#convert the normalized NN prediction back into kelvin
Twall=Twallnorm*ystd+ymean

#calculate wall temperature directly from the corrected resistance
#network using the hot-gas-side wall temperature
tphysics=T_hot-qflux/hgaschamber

#compare the PINN prediction against the physics result
difference=torch.abs(Twall-tphysics)

#convert the temperature difference into percentage error
errorpercent=(difference/tphysics)*100

print("\n========== v5.6 PHYSICAL REALISM AUDIT (chamber regime) ==========\n")

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
print("Total mass flow totalmdot =",totalmdot)
print("Per-channel mass flow mdotchannel =",mdotchannel)
print("Velocity =",v)
print("Reynolds Number Re =",Re)

if Re<2300:
  print("Flow Regime = Laminar")
elif Re<4000:
  print("Flow Regime = Transitional")
else:
  print("Flow Regime = Turbulent")

print("\n----- HEAT TRANSFER -----")
print("Nusselt Number Nu (gnielinski) =",nu)
print("Heat Transfer Coefficient h =",h)
print("Correlation valid (gnielinski or dittus-boelter) =",corrvalid)

print("\n----- PRESSURE LOSS -----")
print("Friction Factor f =",f)
print("Pressure Loss dP =",dP)

print("\n----- THERMAL ENERGY BALANCE -----")
print("hgaschamber =",hgaschamber)
print("Peak Heat Flux qflux =",qflux)
print("Average Heat Flux qfluxavg =",qfluxavg)
print("Imposed Heat Load Q =",Q)
print("Coolant Temperature Rise =",dTcoolant)
print("Coolant Outlet Temperature =",Tout)

print("\n----- WALL TEMPERATURE -----")
print("PINN Wall Temperature =",Twall)
print("Physics Wall Temperature =",tphysics)
print("PINN vs Physics Difference =",difference)
print("PINN vs Physics Error (%) =",errorpercent)

print("\n----- MODEL ASSUMPTIONS -----")

#these warnings are now meaningful because the corrected coolant
#properties and channel-flow calculation produce realistic reynolds
#and prandtl numbers instead of the old extreme values
if Re>3e6:
  print("WARNING: Reynolds number exceeds the 3.0e6 v5.7 constraint")

if v>100:
  print("WARNING: Coolant velocity exceeds 100 m/s")

if Pr<0.5:
  print("WARNING: Prandtl number is below 0.5 (outside gnielinski/DB range)")

#flag designs where neither available heat-transfer correlation is valid
if not corrvalid:
  print("WARNING: neither gnielinski nor dittus-boelter is valid for this Re/Pr")

#list the major physics assumptions used by the corrected model
print("Gnielinski correlation = Used (dittus-boelter fallback where gnielinski invalid)")
print("Petukhov friction factor / Darcy-Weisbach pressure loss = Used")
print("Hot gas -> wall conduction -> coolant resistance network = Used")
print("Coolant properties = Constant (no temperature/pressure dependence)")
print("Gas-side h = chamber-regime assumption only (throat checked separately in v5.3)")
print("Property variation with temperature = Not modeled")

print("\n----- AUDIT SUMMARY -----")

#check whether the PINN agrees closely with the corrected physics model
if errorpercent<5:
  print("PINN vs physics agreement = Within 5%")
else:
  print("PINN vs physics agreement = Greater than 5%")

#check whether the optimized design stays within the corrected 1.8MPa
#pressure-loss constraint
dPmax=1.8e6
if dP<=dPmax:
  print("Pressure Loss Constraint = Satisfied")
else:
  print("Pressure Loss Constraint = Violated")

print("\n========== END PHYSICAL REALISM AUDIT ==========\n")
