"""
v5.2 — Cooling Channel Design Optimization

Uses the v5.1 neural network and physics calculations to test
many cooling channel designs and find a design that balances
wall temperature and pressure loss while meeting the engineering
constraints.
"""
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

Tin=115   #coolant inlet temperature in K, changed from 150K to a more realistic 110-120K range

checkpoint=torch.load("PINN_Modelv5_1_complete.pth")

model.load_state_dict(checkpoint["model"])

xmean=checkpoint["xmean"]
xstd=checkpoint["xstd"]

ymean=checkpoint["ymean"]
ystd=checkpoint["ystd"]

model.eval()

print("Complete Model v5.1 loaded")

#liquid methane properties
#mu was previously much lower, which made Re and h unrealistically high
rho=422        #density (kg/m^3)
Cp=3480        #specific heat (J/kg-K)
k=0.187        #thermal conductivity (W/m-K)
mu=1.2e-4      #dynamic viscosity (Pa-s)
Pr=(Cp*mu)/k   #calculate Pr from the properties instead of using a separate fixed value

#wall properties for the chamber liner
kwall=320      #wall thermal conductivity (W/m-K)
tw=0.001       #wall thickness (m)

#hot-gas-side heat transfer coefficient for the chamber section
#this uses the middle of the estimated 4000-7000 W/m^2-K range
#the throat has much higher heat transfer, so it will be checked separately in v5.3
hgaschamber=5500     #W/m^2-K

T_hot=3500     #hot gas temperature used for the wall heat transfer calculation

#total coolant flow rate
#the total flow stays fixed and is divided between the cooling channels
totalmdot=20   #kg/s, split across n_channels below

#used to estimate the average heat absorbed by the coolant
#the peak heat flux from the resistance model is not applied over the whole channel
avgfluxfactor=0.25

print("Properties defined")

#design variable ranges for the optimization
Lmin=0.5     #meters
Lmax=2.0

wmin=0.0005  #meters (0.5mm)
wmax=0.0020  #meters (2.0mm)

Hmin=0.0010  #meters (1.0mm)
Hmax=0.0080  #meters (8.0mm)

ARmin=1.0    #aspect ratio H/w
ARmax=10.0

nchmin=50    #minimum number of channels
nchmax=300   #maximum number of channels

print("Design space defined")

numdesigns=20000  #number of random designs to test

L=np.random.uniform(Lmin,Lmax,numdesigns)
w=np.random.uniform(wmin,wmax,numdesigns)
AR=np.random.uniform(ARmin,ARmax,numdesigns)  #randomly choose the aspect ratio
H=w*AR                                         #calculate height from width and aspect ratio
nch=np.random.uniform(nchmin,nchmax,numdesigns) #randomly choose the number of channels

#remove designs where the calculated height is outside the allowed range
keep=(H>=Hmin)&(H<=Hmax)
L,w,H,nch=L[keep],w[keep],H[keep],nch[keep]

designs=torch.tensor(
    np.column_stack([L,w,H,nch]),
    dtype=torch.float32
)
#the fourth input is now the number of channels instead of mass flow

print(designs[:5])
print(designs.shape)

designsnorm=(designs-xmean)/xstd
print(designsnorm[:5])

with torch.no_grad():
    predictionsnorm=model(designsnorm)

print(predictionsnorm[:5])

predictions=predictionsnorm*ystd+ymean   #convert the NN prediction back to wall temperature in K

#calculate the physics again for each design
#this lets us check the NN result and calculate things like pressure loss

Lc=designs[:,0:1].numpy()
wc=designs[:,1:2].numpy()
Hc=designs[:,2:3].numpy()
nchc=designs[:,3:4].numpy()

Dh=(2*wc*Hc)/(wc+Hc)
A=wc*Hc
mdotchannel=totalmdot/nchc     #mass flow through each channel
v=mdotchannel/(rho*A)          #velocity based on the flow rate and channel area

Re=(rho*v*Dh)/mu

#use Gnielinski as the main correlation for turbulent flow
#Dittus-Boelter is kept as a backup when Gnielinski is outside its valid range
f=(0.79*np.log(np.clip(Re,1.0,None))-1.64)**-2
nu_gnielinski=(f/8)*(Re-1000)*Pr/(1+12.7*np.sqrt(f/8)*(Pr**(2/3)-1))
nu_db=0.023*(Re**0.8)*(Pr**0.4)

gnielinski_valid=(Re>=3000)&(Re<=5e6)&(Pr>=0.5)&(Pr<=2000)
db_valid=(Re>=10000)&(Pr>=0.6)&(Pr<=160)
corrvalid=gnielinski_valid|db_valid
nu=np.where(gnielinski_valid,nu_gnielinski,np.where(db_valid,nu_db,np.nan))

h=(nu*k)/Dh   #calculate the coolant-side heat transfer coefficient

#calculate heat transfer from the hot gas through the wall and into the coolant
#heat flux is calculated from the thermal resistance instead of being fixed
U=1.0/(1.0/hgaschamber+tw/kwall+1.0/h)
qflux=(T_hot-Tin)*U
twallgas=T_hot-qflux/hgaschamber    #hot-gas-side wall temperature
twallcool=Tin+qflux/h        #coolant-side wall temperature

#calculate pressure loss through each cooling channel
#use the same friction factor from the heat transfer calculation
dP=f*(L.reshape(-1,1)/Dh)*(rho*v**2/2)

#estimate coolant outlet temperature using the average heat flux
As=2*L.reshape(-1,1)*(wc+Hc)
qfluxavg=qflux*avgfluxfactor
Q=qfluxavg*As
Tout=Tin+Q/(mdotchannel*Cp)

#check how far each design is from the training data distribution
#designs more than 3 standard deviations away are treated as unreliable NN predictions
designznorm=((designs-xmean)/xstd).numpy()
oodscore=np.max(np.abs(designznorm),axis=1,keepdims=True)
oodthreshold=3.0

#maximum allowed pressure loss
dPmax=1.8e6

#engineering limits that a design has to meet
maxvelocity=100.0
maxreynolds=3.0e6
maxwalltemp=825.0
maxoutlettemp=380.0

#additional ranges used to make sure the results stay within realistic values
#these are based on methane regenerative cooling CFD data
Gmin=3027    #minimum mass flux (kg/s-m^2)
Gmax=35000   #maximum mass flux (kg/s-m^2)
twallliteraturemin=230   #minimum wall temperature from the dataset
twallliteraturemax=1482  #maximum wall temperature from the dataset

#the literature data has higher velocities than the 100 m/s limit used here
#the 100 m/s value is kept because it is the engineering limit being used in this project

G=rho*v   #mass flux through the cooling channels

velocityok=v<=maxvelocity
reynoldsok=Re<=maxreynolds
pressureok=dP<=dPmax
walltempok=twallgas<=maxwalltemp
outletok=Tout<=maxoutlettemp
oodok=oodscore<=oodthreshold
massfluxplausible=(G>=Gmin)&(G<=Gmax)
walltempplausible=(twallgas>=twallliteraturemin)&(twallgas<=twallliteraturemax)

#only designs that pass all of the constraints are considered feasible
feasible=(corrvalid.reshape(-1,1)&velocityok&reynoldsok&pressureok&walltempok
          &outletok&oodok&massfluxplausible&walltempplausible)

print("feasible designs:",feasible.sum(),"/",len(feasible))

#the old objective only minimized wall temperature, which pushed the optimizer
#toward designs with very high heat transfer and unrealistic results
#
#now the score balances pressure loss and wall temperature
#only designs that pass all of the constraints can receive a score
dPnorm=dP/dPmax
twallnorm=twallgas/maxwalltemp
score=np.where(feasible,0.6*dPnorm+0.4*twallnorm,np.inf)

torch.save(
    {
        "designs":designs,
        "predictions":predictions,
        "twallgas_physics":torch.tensor(twallgas,dtype=torch.float32),
        "pressure_loss":torch.tensor(dP,dtype=torch.float32),
        "outlet_temp":torch.tensor(Tout,dtype=torch.float32),
        "feasible":torch.tensor(feasible),
        "oodscore":torch.tensor(oodscore,dtype=torch.float32),
        "score":torch.tensor(score,dtype=torch.float32),
        "dPmax":dPmax
    },
    "PINN_Modelv5_2_optimization_results.pth"
)

print("Optimization results saved")

bestindex=int(np.argmin(score))         #find the design with the lowest score
bestdesign=designs[bestindex]           #get the dimensions of the best design
besttemp=predictions[bestindex]         #get the NN wall temperature prediction
bestphysicstemp=float(twallgas[bestindex,0])
bestpressureloss=float(dP[bestindex,0])
bestoutlettemp=float(Tout[bestindex,0])
bestscore=float(score[bestindex,0])
bestood=float(oodscore[bestindex,0])

print("Best Design:")
print(bestdesign)

print("Best Design Pressure Loss:")
print(bestpressureloss)

print("Pressure Loss Limit:")
print(dPmax)

print("Predicted Wall Temperature (NN):")
print(besttemp)

print("Predicted Wall Temperature (physics):")
print(bestphysicstemp)

print("Outlet Temperature:")
print(bestoutlettemp)

print("Out-of-distribution score (threshold",oodthreshold,"):")
print(bestood)

print("Mass flux G (literature band",Gmin,"-",Gmax,"kg/s-m2):")
print(float(G[bestindex,0]))

print("Optimization Score:")
print(bestscore)

torch.save(
    {
        "bestdesign":bestdesign,
        "besttemp":besttemp,
        "bestphysicstemp":bestphysicstemp,
        "bestpressureloss":bestpressureloss,
        "bestoutlettemp":bestoutlettemp,
        "bestscore":bestscore,
        "bestood":bestood
    },
    "PINN_Modelv5_2_optimized_design.pth"
)

print("Optimized design saved")
