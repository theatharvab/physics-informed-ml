"""
v5.2 — Constrained Cooling Channel Optimization

Loads the v5.1 cooling surrogate and searches the design space
for regenerative cooling channel geometries that minimize predicted
wall temperature while satisfying a coolant pressure-loss constraint.
"""

import torch
import torch.nn as nn
import numpy as np

#make optimization results reproducible
np.random.seed(39)
torch.manual_seed(39)

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

tcoolant=150

checkpoint=torch.load("PINN_Modelv5_1_complete.pth")

model.load_state_dict(checkpoint["model"])

xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]

ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]

model.eval()

print("Complete Model v5.1 loaded")

#design variable ranges for optimization
Lmin=0.5    #defines allowable channel length range for optimization
Lmax=2.0    #meters

wmin=0.001  #defines possible channel width range
wmax=0.01   #meters

Hmin=0.001  #defins possible channel height range
Hmax=0.01   #meters

mdotmin=0.1 #defines coolant mass flow rate range for design exploration
mdotmax=2.0 #kg/s

print("Design space defined")

numdesigns=10000 #number of channel designs to test
designs=torch.tensor(
    np.column_stack([
        np.random.uniform(Lmin,Lmax,numdesigns), #generates design variables in allowed ranges
        np.random.uniform(wmin,wmax,numdesigns),
        np.random.uniform(Hmin,Hmax,numdesigns),
        np.random.uniform(mdotmin,mdotmax,numdesigns)
    ]),
    dtype=torch.float32
)

print(designs[:5])
print(designs.shape)  #displays total number of design and input variables

designsnorm=(designs-xmean)/xstd #normalizes generated cooling channels w/ same scaling as v5.1
print(designsnorm[:5])

with torch.no_grad():
    predictionsnorm=model(designsnorm)

print(predictionsnorm[:5])
predictions=predictionsnorm*ystd+ymean

#prevent nonphysical cooling predictions
rho=420
mu=1.1e-5
Cp=3500
k=0.1

L=designs[:,0:1]
w=designs[:,1:2]
H=designs[:,2:3]
mdot=designs[:,3:4]

Dh=(2*w*H)/(w+H)

A=w*H

v=mdot/(rho*A)

Re=(rho*v*Dh)/mu

Pr=(Cp*mu)/k

Nu=0.023*(Re**0.8)*(Pr**0.4)

h=(Nu*k)/Dh

f=0.3164*(Re**(-0.25))

dP=f*(L/Dh)*(rho*v**2/2)
#pressure loss constraint

dPmax=3e6

feasible=(predictions>=tcoolant)&(dP<=dPmax)

score=torch.where(
    feasible,
    predictions,
    torch.full_like(predictions,float("inf"))
)

print(predictions[:5])

torch.save(
    {
        "designs":designs,
        "predictions":predictions,
        "pressure_loss":dP,
        "feasible":feasible,
        "score":score,
        "dPmax":dPmax
    },
    "PINN_Modelv5_2_optimization_results.pth"
)

print("Optimization results saved")

bestindex=torch.argmin(score)        #index of minimum temp
bestdesign=designs[bestindex]        #retrieves best measurements
besttemp=predictions[bestindex]      #retrieves the lowest temp (K)
bestpressureloss=dP[bestindex]
bestscore=score[bestindex]

print("Best Design:")
print(bestdesign)

print("Best Design Pressure Loss:")
print(bestpressureloss)

print("Pressure Loss Limit:")
print(dPmax)

print("Predicted Wall Temperature:")
print(besttemp)

print("Pressure Loss:")
print(bestpressureloss)

print("Optimization Score:")
print(bestscore)

torch.save(
    {
        "bestdesign":bestdesign,
        "besttemp":besttemp,
        "bestpressureloss":bestpressureloss,
        "bestscore":bestscore
    },
    "PINN_Modelv5_2_optimized_design.pth"
)

print("Optimized design saved")
