"""
v5.1 — Physics-Based Dataset and Cooling Surrogate

Generates a physics-based dataset for rectangular regenerative
cooling channels and trains a neural-network surrogate to predict
combustion-chamber wall temperature.
"""

import torch
import matplotlib.pyplot as plt

#liquid methane properties
rho=420
Cp=3500
k=0.1
mu=1.1e-5
Pr=1
#cooling environment
tcoolant=150
qflux=5e6

N=1000
L=torch.rand(N,1)*1.5+0.5 #gen lengths between 0.5m & 2m
w=torch.rand(N,1)*0.009+0.001 #gen widths between 1mm & 10mm
H=torch.rand(N,1)*0.009+0.001 #gen heights between 1mm & 10mm
mdot=torch.rand(N,1)*1.9+0.1 #gen flow rates between 0.1kg/s & 2kg/s

x=torch.cat((L,w,H,mdot),dim=1)  #combines channel geometries & flow variables into 1 input dataset
print(x[:5])

#calculate hydraulic diameter of recatngular cooling channel
Dh=(2*w*H)/(w+H) #estimates channel size that affects coolant flow and heat transfer

#calculate cooling velocity from mass flow rate
v=mdot/(rho*w*H) #calculates coolant speed through channel

#calculate reynolds number
Re=(rho*v*Dh)/mu #describes coolant flow behavior

#calculate nusselt number using dittus-boelter correlation
Nu=0.023*(Re**0.8)*(Pr**0.4) #estimates heat transfer enhancement from coolant flow

#calculate heat transfer coefficient
h=(Nu*k)/Dh #measures efficency of coolant removing heat

#calculate wall temperatures
twall=tcoolant+qflux/h #estimates wall temp

print(twall[:5])

#normalize inputs so all features have similar scales
xmean=x.mean(dim=0)
xstd=x.std(dim=0)
xnorm=(x-xmean)/xstd
#normalize output temp
ymean=twall.mean(dim=0)
ystd=twall.std(dim=0)
ynorm=(twall-ymean)/ystd

print(xnorm[:5])
print(ynorm[:5])

import torch.nn as nn

class CoolingNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.network=nn.Sequential(
        nn.Linear(4,32), #inputs into 32 features
        nn.ReLU(),
        nn.Linear(32,32), #32 into 32 reinforced features
        nn.ReLU(),
        nn.Linear(32,1) #32 reinforced features into 1 prediction
    )
  def forward(self,x):
    return self.network(x)
model=CoolingNN()
print(model)

loss_fn=nn.MSELoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(1000):
  prediction=model(xnorm)
  loss=loss_fn(prediction,ynorm)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if epoch%100==0:
    print(epoch,loss)

#new cooling channel desigh
newdesign=torch.tensor([[1.2,0.004,0.006,1.5]])
#L=1.2m, w=4mm, H=6mm, mdot=1.5kg/s
print(newdesign)

newdesignnorm=(newdesign-xmean)/xstd
print(newdesignnorm) #applies same normalization used during training to match NN input format

model.eval() #switches NN into eval mode for testing predictions
with torch.no_grad(): #disables gradient calculations since model is only predicting not training
  predictionnorm=model(newdesignnorm) #uses trained NN to predict normalized wall temmp of new channel design

prediction=predictionnorm*ystd+ymean #converts normalized prediction back into actual wall temp (Kelvin)
print(prediction)
print("Normalized prediction:", predictionnorm)
print("Mean/std:", ymean, ystd)
print("Final Twall:", prediction)

#extract the design variables listed before

Lnew=newdesign[:,0:1]
wnew=newdesign[:,1:2]
Hnew=newdesign[:,2:3]
mdotnew=newdesign[:,3:4]

#recalculate physics
Dhnew=(2*wnew*Hnew)/(wnew+Hnew)    #new hydraulic diameter
vnew=mdotnew/(rho*wnew*Hnew)       #new voolant velocity
Renew=(rho*vnew*Dhnew)/mu          #reynolds number to characterize flow behavior
Nunew=0.023*(Renew**0.8)*(Pr**0.4) #estimates heat transfer w/ nusselt number correlation
hnew=(Nunew*k)/Dhnew               #calculates heat transfer coefficient
tphysics=tcoolant+qflux/hnew       #calculates physics-based wall temp used as validation reference

print(tphysics)
torch.save(
    {
        "model": model.state_dict(),
        "xmean": xmean,
        "xstd": xstd,
        "ymean": ymean,
        "ystd": ystd
    },
    "PINN_Modelv5_1_complete.pth"
)

Dh=(2*w*H)/(w+H)

A=w*H

v=mdot/(rho*A)

Re=(rho*v*Dh)/mu

Pr=(Cp*mu)/k

Nu=0.023*(Re**0.8)*(Pr**0.4)

h=(Nu*k)/Dh

tphysics=tcoolant+qflux/h
print("PINN Twall =", prediction)
print("Physics Twall =", tphysics[:5])

print("Shapes:")
print("L:", L.shape)
print("w:", w.shape)
print("H:", H.shape)
print("mdot:", mdot.shape)
