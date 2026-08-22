"""
v5.1 — Physics-Based Dataset and Cooling Surrogate

Generates a physics-based dataset for rectangular regenerative
cooling channels and trains a neural-network surrogate to predict
combustion-chamber wall temperature.
"""

import torch
import matplotlib.pyplot as plt

#liquid methane properties -- mu was 1.1e-5 before (~11x too low),
#which was inflating Re and h
rho=422
Cp=3480
k=0.187
mu=1.2e-4
Pr=(Cp*mu)/k #derived here instead of hardcoded to 1

#wall (chamber liner) properties
kwall=320  #wall thermal conductivity (W/m-K)
tw=0.001   #wall thickness (m)

#hot-gas-side film coefficient -- chamber section only
#modified Bartz correlation for Pc=6-10MPa, LOX/methane,
#c*~1800-1850m/s, Dt~0.10-0.15m
#chamber h_gas is roughly 4000-7000 W/m2K, so this uses the midpoint
#throat h_gas is much higher and will be tested separately in v5.3
#this file only represents the chamber section
hgaschamber=5500

#operating conditions
Tin=115     #coolant inlet temp (K), corrected from 150K
            #subcooled liquid methane is around 110-120K
T_hot=3500  #hot-gas-side driving temp (K)

#total coolant mass flow, split across the channels
#replaces the old independently-sampled mdot
totalmdot=20

N=20000
L=torch.rand(N,1)*1.5+0.5      #gen lengths between 0.5m & 2m
w=torch.rand(N,1)*0.0015+0.0005 #gen widths between 0.5mm & 2mm
AR=torch.rand(N,1)*9+1          #gen aspect ratio between 1 & 10
H=w*AR                          #derive height from width*AR
nch=torch.rand(N,1)*250+50      #gen channel count between 50 & 300

#drop rows where H is outside 1-8mm
keep=(H>=0.001)&(H<=0.008)
L=L[keep].reshape(-1,1)
w=w[keep].reshape(-1,1)
H=H[keep].reshape(-1,1)
nch=nch[keep].reshape(-1,1)

x=torch.cat((L,w,H,nch),dim=1)  #4th column is nch, not mdot
print(x[:5])

#calculate hydraulic diameter
Dh=(2*w*H)/(w+H)

#per-channel mass flow and velocity from total flow + geometry
mdotchannel=totalmdot/nch
v=mdotchannel/(rho*w*H)

#calculate reynolds number
Re=(rho*v*Dh)/mu

#gnielinski correlation with petukhov friction factor
#valid roughly 3000<Re<5e6 and 0.5<Pr<2000
#dittus-boelter is only used as a fallback
f=(0.79*torch.log(torch.clamp(Re,min=1.0))-1.64)**-2
Nu_g=(f/8)*(Re-1000)*Pr/(1+12.7*torch.sqrt(f/8)*(Pr**(2/3)-1))
Nu_db=0.023*(Re**0.8)*(Pr**0.4)

gnielinski_valid=(Re>=3000)&(Re<=5e6)&(Pr>=0.5)&(Pr<=2000)
db_valid=(Re>=10000)&(Pr>=0.6)&(Pr<=160)
corrvalid=gnielinski_valid|db_valid
Nu=torch.where(gnielinski_valid,Nu_g,torch.where(db_valid,Nu_db,torch.full_like(Nu_g,float("nan"))))

#calculate heat transfer coefficient
h=(Nu*k)/Dh

#hot gas -> wall -> coolant resistance network
#q'' comes from this instead of being imposed directly
U=1.0/(1.0/hgaschamber+tw/kwall+1.0/h)
qflux=(T_hot-Tin)*U
twall=T_hot-qflux/hgaschamber   #hot-gas-side wall temp used to train the NN

#drop rows with invalid correlations before training
keep2=corrvalid.reshape(-1,1)&(v<=300)
L=L[keep2].reshape(-1,1)
w=w[keep2].reshape(-1,1)
H=H[keep2].reshape(-1,1)
nch=nch[keep2].reshape(-1,1)
twall=twall[keep2].reshape(-1,1)

x=torch.cat((L,w,H,nch),dim=1)
print(twall[:5])

#normalize inputs so the features have similar scales
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
        nn.Linear(32,32), #32 into 32 features
        nn.ReLU(),
        nn.Linear(32,1) #32 features into 1 prediction
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

for epoch in range(1500):
  prediction=model(xnorm)
  loss=loss_fn(prediction,ynorm)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if epoch%100==0:
    print(epoch,loss)

#new cooling channel design
newdesign=torch.tensor([[1.2,0.0012,0.006,150.0]])
#L=1.2m, w=1.2mm, H=6mm, nch=150
print(newdesign)

newdesignnorm=(newdesign-xmean)/xstd
print(newdesignnorm) #same normalization used during training

model.eval() #switch NN to eval mode for testing
with torch.no_grad(): #no gradient calculations needed for prediction
  predictionnorm=model(newdesignnorm) #predict normalized wall temp

prediction=predictionnorm*ystd+ymean #convert back to wall temp in K
print(prediction)
print("Normalized prediction:", predictionnorm)
print("Mean/std:", ymean, ystd)
print("Final Twall:", prediction)

#extract the design variables

Lnew=newdesign[:,0:1]
wnew=newdesign[:,1:2]
Hnew=newdesign[:,2:3]
nchnew=newdesign[:,3:4]

#recalculate physics
Dhnew=(2*wnew*Hnew)/(wnew+Hnew)     #new hydraulic diameter
mdotchannelnew=totalmdot/nchnew    #new per-channel flow
vnew=mdotchannelnew/(rho*wnew*Hnew) #new coolant velocity
Renew=(rho*vnew*Dhnew)/mu           #reynolds number

fnew=(0.79*torch.log(torch.clamp(Renew,min=1.0))-1.64)**-2
Nunew=(fnew/8)*(Renew-1000)*Pr/(1+12.7*torch.sqrt(fnew/8)*(Pr**(2/3)-1)) #gnielinski
hnew=(Nunew*k)/Dhnew                #heat transfer coefficient

Unew=1.0/(1.0/hgaschamber+tw/kwall+1.0/hnew)
qfluxnew=(T_hot-Tin)*Unew
tphysics=T_hot-qfluxnew/hgaschamber #physics-based wall temp for validation

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

print("PINN Twall =", prediction)
print("Physics Twall =", tphysics)

print("Shapes:")
print("L:", L.shape)
print("w:", w.shape)
print("H:", H.shape)
print("nch:", nch.shape)
