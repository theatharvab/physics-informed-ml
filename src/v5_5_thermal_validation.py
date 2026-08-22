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
nch=bestdesign[3]   #4th design variable is now n_channels instead of mdot,
                    #matching the corrected v5.1 training and v5.2 optimizer

print("L =",L)
print("w =",w)
print("H =",H)
print("nch =",nch)

#coolant properties -- corrected liquid methane values, matching the
#corrected values introduced in v5.1 and used consistently downstream
rho=422
Cp=3480
mu=1.2e-4
k=0.187
Pr=(Cp*mu)/k   #derived from the corrected properties instead of being
               #hardcoded, removing the old Pr inconsistency

#wall (chamber liner) properties -- added as part of the corrected
#hot-gas -> wall -> coolant resistance network introduced in v5.1
kwall=320
tw=0.001

#operating conditions -- Tin corrected from 150K to 115K to match the
#subcooled liquid methane inlet range used throughout the corrected model
Tin=115

#chamber conditions
T_hot=3500

#total coolant mass flow -- fixed system-level input. per-channel flow is
#derived from totalmdot/nch instead of independently sampling mdot
totalmdot=20

#average-to-peak flux ratio -- added in v5.2 to separate the LOCAL/peak
#heat flux used for the wall-temperature constraint from the average
#heat flux used for the coolant energy balance. applying peak flux over
#the full channel length would overstate total heat absorption.
avgfluxfactor=0.25

#gas-side film coefficient -- chamber regime used consistently with the
#v5.1 training data and v5.2 optimization. v5.3 separately stress-tests
#the finalized design under throat-level hgas values.
hgaschamber=5500

print("Properties Defined")

#cross sectional flow area
A=w*H

#hydraulic diameter
Dh=(2*w*H)/(w+H)

#per-channel mass flow and coolant velocity -- both are now derived from
#the fixed system flow and channel count introduced in v5.1
mdotchannel=totalmdot/nch
v=mdotchannel/(rho*A)

#reynolds number -- now uses corrected liquid-methane viscosity and the
#physically derived per-channel flow instead of the old independently
#sampled mdot
Re=(rho*v*Dh)/mu

#gnielinski correlation is now the primary heat-transfer correlation,
#using the petukhov friction factor. dittus-boelter is retained only as
#an explicit fallback when its own validity range is satisfied.
f=(0.79*torch.log(torch.clamp(Re,min=1.0))-1.64)**-2
nu_gnielinski=(f/8)*(Re-1000)*Pr/(1+12.7*torch.sqrt(f/8)*(Pr**(2/3)-1))
nu_db=0.023*(Re**0.8)*(Pr**0.4)
gnielinski_valid=(Re>=3000)&(Re<=5e6)&(Pr>=0.5)&(Pr<=2000)
db_valid=(Re>=10000)&(Pr>=0.6)&(Pr<=160)
corrvalid=gnielinski_valid|db_valid
nu=torch.where(gnielinski_valid,nu_gnielinski,torch.where(db_valid,nu_db,torch.tensor(float("nan"))))

#heat transfer coefficient -- calculated from the valid heat-transfer
#correlation instead of applying dittus-boelter unconditionally
h=(nu*k)/Dh

#darcy friction factor -- uses the same petukhov friction factor as the
#gnielinski correlation, replacing the old separate mismatched
#blasius-style pressure-loss calculation
dP=f*(L/Dh)*(rho*v**2/2)

#cooling channel surface area
As=2*L*(w+H)

#hot gas -> wall conduction -> coolant resistance network. qflux is the
#LOCAL/peak chamber heat flux and now emerges from the thermal resistance
#network instead of using the old imposed qflux=5e6.
U=1.0/(1.0/hgaschamber+tw/kwall+1.0/h)
qflux=(T_hot-Tin)*U

#calculate actual heat load from heat flux -- uses the AVERAGE heat flux
#instead of applying the local peak flux uniformly over the entire
#channel length, matching the v5.2/v5.3 energy-balance treatment
qfluxavg=qflux*avgfluxfactor
Q=qfluxavg*As

#calculate how much the coolant temperature increases -- uses the
#derived per-channel mass flow and corrected coolant specific heat
dTcoolant=Q/(mdotchannel*Cp)

#calculate coolant outlet temperature from the corrected energy balance
Tout=Tin+dTcoolant

#normalize design so the NN can use it -- the v5.1 checkpoint expects the
#corrected four-variable design vector [L,w,H,nch]
designnorm=(bestdesign-xmean)/xstd

with torch.no_grad():
  Twallnorm=model(designnorm)

#convert normalized prediction back to Kelvin
Twall=Twallnorm*ystd+ymean

#calculate physics wall temperature -- uses the corrected resistance
#network and the hot-gas-side wall temperature that governs the 825K limit
Tphysics=T_hot-qflux/hgaschamber

#calculate difference between PINN and physics
Tdifference=torch.abs(Twall-Tphysics)

#calculate percent error between PINN and physics
Terrorpercent=(Tdifference/Tphysics)*100

#calculate the heat flux that corresponds to exactly 825K at the fixed
#chamber hgas. this replaces the old Qcapacity calculation, which mixed
#the coolant-side h with the hot-gas driving temperature and was not a
#physically coherent heat-capacity check.
#for fixed hgaschamber, higher qflux means a lower gas-side wall
#temperature because more of the total temperature drop occurs across
#the gas film. therefore positive qflux-qfluxat825 means the wall is
#below the 825K limit.
MAXWALLTEMP=825.0
qfluxat825=(T_hot-MAXWALLTEMP)*hgaschamber
fluxmargin=qflux-qfluxat825   #positive means Twall is safely under 825K

#check that heat entering the coolant matches the calculated heat load.
#this is definitionally satisfied because Tout was calculated directly
#from Q using this same relationship, so it is an arithmetic
#self-consistency check rather than an independent energy validation.
Qcheck=mdotchannel*Cp*(Tout-Tin)

#check if design has enough thermal margin before reaching the 825K wall
#temperature limit using the corrected flux-margin formulation
thermalcapacitysufficient=fluxmargin>=0

#pressure-loss limit unified to the corrected 1.8MPa engineering
#constraint. the old value here was 3e6 and did not match the corrected
#value propagated through the later model.
dPmax=1.8e6
pressurelosswithinlimit=dP<=dPmax

print("\n========== v5.5 THERMAL VALIDATION (chamber regime) ==========\n")

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
print("Pressure Loss Limit =",dPmax)
print("Pressure Loss Within Limit =",pressurelosswithinlimit)

print("\n----- THERMAL ENERGY BALANCE -----")
print("hgaschamber =",hgaschamber)
print("Peak Heat Flux qflux =",qflux)
print("Average Heat Flux qfluxavg =",qfluxavg)
print("Imposed Heat Load Q =",Q)
print("Coolant Temperature Rise =",dTcoolant)
print("Coolant Outlet Temperature =",Tout)

print("\n----- ENERGY BALANCE CHECK (definitional, see comment above) -----")
print("Heat Load Q =",Q)
print("Energy Balance Q =",Qcheck)
print("Energy Balance Difference =",torch.abs(Q-Qcheck))

print("\n----- WALL TEMPERATURE -----")
print("PINN Wall Temperature =",Twall)
print("Physics Wall Temperature =",Tphysics)
print("PINN vs Physics Difference =",Tdifference)
print("PINN vs Physics Error (%) =",Terrorpercent)

print("\n----- FLUX CAPACITY (vs 825K limit, chamber regime) -----")
print("NOTE: for fixed hgaschamber, higher qflux means a COOLER wall --")
print("more total flux shifts more of the temperature drop onto the gas")
print("film itself. Safe when actual qflux is ABOVE qfluxat825.")
print("Flux at 825K threshold qfluxat825 =",qfluxat825)
print("Actual Flux qflux =",qflux)
print("Flux Margin (qflux - qfluxat825) =",fluxmargin)
print("Thermal Capacity Sufficient =",thermalcapacitysufficient)

print("\n========== END VALIDATION ==========\n")
