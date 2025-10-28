import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pencil as pc
import h5py

#input bounds and number of grid points for x and y
#make sure observer, does not sit on a grid point

#computes gradients of lines going through all points and removes duplicates
#the actual number of bins is one less than nbins since -pi and pi are the same angle
nbinst = 300
thetas = np.linspace(-np.pi,np.pi,nbinst)
binst = np.zeros(nbinst-1)

nbinsp = 300
phis = np.linspace(-0.5*np.pi,0.5*np.pi,nbinsp)
binsp = np.zeros(nbinsp-1)

#for each point determine which angle it is closest to
def BinT(Px,Py):
    #P is a point = [Px,Py,Pz]
    thetaP = np.atan2(Py,Px)
    for i in np.arange(0,nbinst-1):
        if thetas[i]<=thetaP<thetas[i+1]:
            return i

def BinP(Px,Py,Pz):
    #P is a point = [Px,Py,Pz]
    r = np.sqrt(Px**2+Py**2)
    phiP = np.atan2(Pz,r)
    for j in np.arange(0,nbinsp-1):
        if phis[j]<=phiP<phis[j+1]:
            return j

#copied stuff from N_e.py to calculate the final ne and B
G0 = 1
xiCR = 1
xiXR = 1
xit = xiCR + xiXR
phiPAH = 0.5
Zd = 1

var = pc.read.var('VAR1.h5',trimall=True,magic=['tt','bb'])
par = pc.read.param()

#B, T and ncr from observations
B = var.bb
T = var.tt*par.unit_temperature
#ncr actually reads ecr, energy of cosmic rays however the distinction is not important
#for our purposes
var2 = h5py.File('data/allprocs/var.h5','r+')
ncr = np.array(var2['data/ecr'][()])
ncr = ncr[2:802,2:194,2:194]


#Option of shift of position (in kpc)
xshift = 0
yshift = 0
zshift = 0

x = var.x + xshift
y = var.y + yshift
z = var.z + zshift



def Ne(xit,T,G0,Zd,phiPAH):
    '''
    inputs:
        Total ionisation rate, xit
        Temperature, T
        Scaled intensity of FUV interstellar radiation field, G0
        Metallicity, Zd
        Scaling parameter of the PAH collision rates, phiPAH
    '''
    T2 = T/100

    return (2.4e-3 * np.sqrt(xit) * T2**(0.25) * np.sqrt(G0)) / (np.sqrt(Zd) * phiPAH)

NE = Ne(xit,T,G0,Zd,phiPAH)
Ent = var.ss*par.unit_entropy
EntMask = ma.masked_greater(Ent,23e8)
MaskHot = ma.getmask(EntMask)
NEMask = ma.array(NE, mask=MaskHot)

#function that calculates the integrand of RM at  every point
def RM(B,ne,theta,phi):
    #Calculates RM at a given point (x,y,z) given magnetic field B at that point
    #var gives vectors in the form (Bz, Bx, By) so we roll it once backwards to get
    #(Bx,By,Bz)
    #and line of sight direction d = (sin(phi)cos(theta),sin(phi)sin(theta),cos(phi))
    #as well as various physical constants
    e = 1.602e-19 #Coulombs
    e_0 = 8.854e-12 #Farads per metre
    m_e = 9.109e-31 #Kilograms
    c = 2.998e8 #Metres per second
    k = (e**3)/(8 * np.pi**2 * e_0 * m_e**2 * c**3) #nasty constant in front of integral

    d = [np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]
    if ne is not ma.masked:
        return k*ne*np.dot(d,np.roll(B,-1))
    else:
        return 0

def SR(B,ncr,theta,phi):
    #d is unit vector in the direction of the line of sight
    d = [np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]
    #BPerp is the magnetic field perpendicular to the LOS
    #Note we roll B as the output gives (Bz,Bx,By) whereas we want (Bx,By,Bz)
    BPerp = np.roll(B,-1) - np.dot(d,np.dot(np.roll(B,-1),d))
    
    return ncr*np.dot(BPerp,BPerp)


#add that point's data to the corresponding point in the array
#BinFinal contains all the data in bins by (theta,phi) for plotting purposes
BinFinalRM = np.zeros([nbinst,nbinsp])
BinFinalSR = np.zeros([nbinst,nbinsp])

#actual code


#Calculates the RM
for i in np.arange(0,len(x)):
    for j in np.arange(0,len(y)):
        for k in np.arange(0,len(z)):
            t,p = BinT(x[i],y[j]), BinP(x[i],y[j],z[k])
            BinFinalRM[t,p] += RM(B[:,k,i,j],NEMask[k,i,j],np.atan2(y[j],x[i]),np.atan2(z[k],np.sqrt(x[i]**2+y[j]**2)))
            if i==len(x)-1 & j==len(y)-1:
                print(k)

#Calculates the Synchrotron Radiation
for i in np.arange(0,len(x)):
    for j in np.arange(0,len(y)):
        for k in np.arange(0,len(z)):
            t,p = BinT(x[i],y[j]), BinP(x[i],y[j],z[k])
            BinFinalSR[t,p] += SR(B[:,k,i,j],ncr[k,i,j],np.atan2(y[j],x[i]),np.atan2(z[k],np.sqrt(x[i]**2+y[j]**2)))
            if i==len(x)-1 & j==len(y)-1:
                print(k)

#code for plotting the RM on a Mollweide projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
Lon,Lat = np.meshgrid(thetas,phis)
vmin = min(BinFinalRM.min(),-BinFinalRM.max())
vmax = -vmin
im = ax.pcolormesh(Lon,Lat,BinFinalRM.T,cmap='seismic',vmin=vmin,vmax=vmax)
plt.colorbar(im)
plt.xticks([])
plt.yticks([])
plt.title(r'The RM of the simulated ISM')
plt.show()


#code for plotting SR on a Mollweide projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
Lon,Lat = np.meshgrid(thetas,phis)
vmin = 0
vmax = BinFinalSR.max()
im = ax.pcolormesh(Lon,Lat,BinFinalSR.T,cmap='grey',vmin=vmin,vmax=vmax)
plt.colorbar(im)
plt.xticks([])
plt.yticks([])
plt.title(r'The Synchrotron Radiation of the simulated ISM')
plt.show()

