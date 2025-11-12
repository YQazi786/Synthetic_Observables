import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pencil as pc
import h5py

#This program will calculate the angles and bins for every point in the array
#This only has to be done once per configuration to save huge amount of computation time!
#Can't believe I didn't think of this before

#Number of beams for theta
nbinst = 300
thetas = np.linspace(-np.pi,np.pi,nbinst)
binst = np.zeros(nbinst-1)

#Number of beams for phi
nbinsp = 300
phis = np.linspace(-0.5*np.pi,0.5*np.pi,nbinsp)
binsp = np.zeros(nbinsp-1)

x = np.linspace(-0.764,0.764,384)
y = np.linspace(-0.764,0.764,384)
z = np.linspace(-1.598,1.598,800)[208:592]

#Find the actual angles of each point
ThetaPoints = np.zeros([384,384,384])
PhiPoints = np.zeros([384,384,384])
for k in np.arange(0,len(z)):
    for i in np.arange(0,len(x)):
        for j in np.arange(0,len(y)):
            ThetaPoints[k,i,j] = np.atan2(y[j],x[i])
            PhiPoints[k,i,j] = np.atan2(z[k],np.sqrt(x[i]**2+y[j]**2))
    print(k) #Purely for progess update

#Saves the files as numpy files to be read in to Line_of_sight.py
np.save('ThetaPoints 2x2.npy',ThetaPoints)
np.save('PhiPoints 2x2.npy',PhiPoints)


#Save which bins each point goes into
ThetaBins = np.zeros([384,384,384])
PhiBins = np.zeros([384,384,384])

for k in np.arange(0,len(z)):
    for i in np.arange(0,len(x)):
        for j in np.arange(0,len(y)):
            for t in np.arange(0,nbinst-1):
                if thetas[t]<=ThetaPoints[k,i,j]<thetas[t+1]:
                    ThetaBins[k,i,j] = t
            for p in np.arange(0,nbinsp-1):
                if phis[p]<=PhiPoints[k,i,j]<phis[p+1]:
                    PhiBins[k,i,j] = p
    print(k)

#Again, saves the files
np.save('ThetaBins 2x2.npy', ThetaBins)
np.save('PhiBins 2x2.npy',PhiBins)



#Distance of each point from origin
#No current use of this but could be useful
DistBins = np.zeros([384,384,384])

for k in np.arange(0,len(z)):
    for i in np.arange(0,len(x)):
        for j in np.arange(0,len(y)):
            DistBins[k,i,j] = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
    print(k)

np.save('DistBins 2x2.npy',DistBins)













