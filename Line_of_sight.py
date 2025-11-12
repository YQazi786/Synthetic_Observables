import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pencil as pc
import h5py

#Theta and phi values at each point
ThetaPoints = np.load('2x2 Cubes/ThetaPoints 2x2.npy')
PhiPoints = np.load('2x2 Cubes/PhiPoints 2x2.npy')

#Bins that each point goes into
ThetaBins = np.load('2x2 Cubes/ThetaBins 2x2.npy')
PhiBins = np.load('2x2 Cubes/PhiBins 2x2.npy')


def LoS(Func,z,x,y,theta,phi,ThetaBins = ThetaBins,PhiBins = PhiBins,ThetaPoints = ThetaPoints,PhiPoints = PhiPoints):
    #Func is the integrand to be calculated at each point
    #Func MUST be defined to be a function of z,x,y (i.e. k,i,j) and theta, phi ONLY
    #Shape is the shape of the array
    #z, x and y are the dimensions of the box
    #theta and phi are the bins for each line of sight
    #Line of sight originates from (0,0,0)
    Final = np.zeros([len(theta),len(phi)])
    for k in np.arange(0,len(z)):    
        for i in np.arange(0,len(x)):
            for j in np.arange(0,len(y)):
                #Reads in which bins each point goes in from a pre-calculated array
                p = int(PhiBins[k,i,j])
                t = int(ThetaBins[k,i,j])

                #Reads in actual angles for calculation 
                phiP = PhiPoints[k,i,j]
                thetaP = ThetaPoints[k,i,j]

                Final[p,t] += Func(k,i,j,thetaP,phiP)
        print(k)
    return Final


#Variables within the cube
T = np.load('2x2 Cubes/T 2x2.npy')[208:592,:,:]
B = np.load('2x2 Cubes/B 2x2.npy')[:,208:592,:,:]
S = np.load('2x2 Cubes/S 2x2.npy')[208:592,:,:]
ncr = np.load('2x2 Cubes/ncr 2x2.npy')[208:592,:,:]
par = pc.read.param()
#Dimensions of cube, trimmed to only contain the sphere
x = np.linspace(-0.764,0.764,384)
y = np.linspace(-0.764,0.764,384)
z = np.linspace(-1.598,1.598,800)[208:592]

#Reads in and masks NE
NE = np.load('2x2 Cubes/NE 2x2.npy')[208:592:,:]
SMask = ma.masked_greater(S,23e8)
MaskHot = ma.getmask(SMask)
NEMask = ma.array(NE, mask=MaskHot)

#Theta and phi bins
nbinst = 300
thetas = np.linspace(-np.pi,np.pi,nbinst)
binst = np.zeros(nbinst-1)

nbinsp = 300
phis = np.linspace(-0.5*np.pi,0.5*np.pi,nbinsp)
binsp = np.zeros(nbinsp-1)



#Redefining the functions to take only arguments 
#Rotation measure RM
def RM(k,i,j,theta,phi,NE=NEMask,B=B):
    const = 2.6304012093737727e-13
    d = [np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]
    ne = NE[k,i,j]
    b = B[:,k,i,j]
    if ne is not ma.masked:
        return const*ne*np.dot(d,np.roll(b,-1))
    else:
        return 0

#Synchrotron radiation SR
def SR(k,i,j,theta,phi,B=B,ncr=ncr):
    B = B[:,k,i,j]
    ncr = ncr[k,i,j]
    #d is unit vector in the direction of the line of sight
    d = [np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]
    #BPerp is the magnetic field perpendicular to the LOS
    #Note we roll B as the output gives (Bz,Bx,By) whereas we want (Bx,By,Bz)
    BPerp = np.roll(B,-1) - np.dot(d,np.dot(np.roll(B,-1),d))

    return ncr*np.dot(BPerp,BPerp)

#Psi to calculate Stokes
def Psi(k,i,j,theta,phi,RM=BinFinalRM,wavelength=1,B=B,PhiBins=PhiBins,ThetaBins=ThetaBins):
    #Reads in which bin each point is in and thus which value of RM is valid
    p = int(PhiBins[k,i,j])
    t = int(ThetaBins[k,i,j])
    rm = RM[p,t]

    #Other factors in calculating psi
    b = B[:,k,i,j]
    d = [np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]
    Bpar = np.dot(np.dot(np.roll(b,1),d),d)
    Bperp = np.roll(b,-1) - np.dot(np.dot(np.roll(b,1),d),d)

    psi0 = np.arctan(np.sqrt(Bpar.dot(Bpar)),np.sqrt(Bperp.dot(Bperp)))

    return psi0 + (wavelength**2 * rm)

def StokesQ(k,i,j,theta,phi,B=B,pm=0.75):
    b = B[:,k,i,j]
    d = [np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]
    Bperp = np.roll(b,-1) - np.dot(np.dot(np.roll(b,1),d),d)
    return 0.75*ncr[k,i,j] * Bperp.dot(Bperp) * np.cos(2*Psi(k,i,j,theta,phi))

def StokesU(k,i,j,theta,phi,B=B,pm=0.75):
    b = B[:,k,i,j]
    d = [np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]
    Bperp = np.roll(b,-1) - np.dot(np.dot(np.roll(b,1),d),d)
    return 0.75*ncr[k,i,j] * Bperp.dot(Bperp) * np.sin(2*Psi(k,i,j,theta,phi))





#Actual calculations
#RM 
#should take approx 9 minutes
BinFinalRM = LoS(RM,z,x,y,thetas,phis)
RMConv = par.unit_length/1000
BinFinalRM *= RMConv
#Code for plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
Lon,Lat = np.meshgrid(thetas,phis)
vmin = min(BinFinalRM.min(),-BinFinalRM.max())
vmax = -vmin
im = ax.pcolormesh(Lon,Lat,BinFinalRM,cmap='seismic',vmin=vmin,vmax=vmax)
plt.colorbar(im,location='bottom',orientation='horizontal',label=r'rad m$^{-2}$')
plt.xticks([])
plt.yticks([])
plt.title(r'The Rotation Measure RM')
plt.show()


#SR
#Should take approx 15 minutes
BinFinalSR = LoS(SR,z,x,y,thetas,phis)
#Code for plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
Lon,Lat = np.meshgrid(thetas,phis)
im = ax.pcolormesh(Lon,Lat,BinFinalSR,cmap='grey',norm=mc.LogNorm())
plt.colorbar(im,location='bottom',orientation='horizontal')
plt.xticks([])
plt.yticks([])
plt.title(r'The Synchrotron Radiation SR')
plt.show()


#Psi for sanity check
BinFinalPsi = LoS(Psi,z,x,y,thetas,phis)

#Stokes Q
#Read in BinFinalRM if not already ran
BinFinalRM = np.load('2x2 Cubes/RM 2x2.npy')
BinFinalQ = LoS(StokesQ,z,x,y,thetas,phis)
#Code for plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
Lon,Lat = np.meshgrid(thetas,phis)
im = ax.pcolormesh(Lon,Lat,BinFinalQ,cmap='seismic')
plt.colorbar(im,location='bottom',orientation='horizontal')
plt.xticks([])
plt.yticks([])
plt.title(r'The Stokes Parameter $Q$')
plt.show()


#Stokes U
#Read in BinFinalRM if not already run
BinFinalRM = np.load('2x2 Cubes/RM 2x2.npy')
BinFinalU = LoS(StokesU,z,x,y,thetas,phis)
#Code for Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
Lon,Lat = np.meshgrid(thetas,phis)
im = ax.pcolormesh(Lon,Lat,BinFinalU,cmap='seismic')
plt.colorbar(im,location='bottom',orientation='horizontal')
plt.xticks([])
plt.yticks([])
plt.title(r'The Stokes Parameter $U$')
plt.show()


#Calculates P from SR, U and Q
#Reads in the above if not already calculated directly
BinFinalSR = np.load('2x2 Cubes/SR 2x2.npy')
BinFinalQ = np.load('2x2 Cubes/Stokes Q 2x2.npy')
BinFinalU = np.load('2x2 Cubes/Stokes U 2x2.npy')

#Calculate the modulus and argument of P = (Q+iU)/I
P = np.zeros(BinFinalQ.shape)
ArgP = np.zeros(BinFinalQ.shape)
for p in np.arange(0,P.shape[0]):
    for t in np.arange(0,P.shape[1]):
        ArgP[p,t] = 0.5*np.arctan(BinFinalU[p,t]/BinFinalQ[p,t])
        #Exclude when I=0 to avoid divide by 0 errors
        if BinFinalSR[p,t] != 0:
            P[p,t] = np.sqrt(BinFinalQ[p,t]**2 + BinFinalU[p,t]**2)/BinFinalSR[p,t]


#Code for plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
Lon,Lat = np.meshgrid(thetas,phis)
im = ax.pcolormesh(Lon,Lat,P,cmap='grey',norm=mc.LogNorm())
plt.colorbar(im,location='bottom',orientation='horizontal')
plt.xticks([])
plt.yticks([])
plt.title(r'The Modulus $|P|$ of P')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
Lon,Lat = np.meshgrid(thetas,phis)
im = ax.pcolormesh(Lon,Lat,ArgP,cmap='seismic')
plt.colorbar(im,location='bottom',orientation='horizontal')
plt.xticks([])
plt.yticks([])
plt.title(r'The Argument $\Psi$ of P')
plt.show()









