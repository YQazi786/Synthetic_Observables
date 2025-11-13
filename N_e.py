import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pencil as pc
import matplotlib.colors as mc

#Uses Wolfire et al. 'Neutral Atomic Phases of the ISM in the Galaxy' to describe the number of electrons in the cold ionised medium (CIM) and warm ionised medium (WIM)
#Values of parameters are all for the solar circle, R=8.5kpc
#G0 from Table 2 Wolfire

#Variables from simulation
var = pc.read.var('VAR1.h5',trimall=True,magic=['tt','bb'])
par = pc.read.param()
T = var.tt*par.unit_temperature
S = var.ss*par.unit_entropy
rho = var.rho
x = var.x
y = var.y
z = var.z


def Ne(T,S,rho,xit=2,G0=1,Zd=1,phiPAH=0.5):
    HotMask = ma.getmask(ma.masked_less(S,23.2e8))
    WarmMask = ma.getmask(ma.masked_outside(S,3.7e8,23.3e8))
    ColdMask = ma.getmask(ma.masked_greater(S,3.7e8))

    rhoHot = ma.array(rho,mask=HotMask)
    THot, TWarm, TCold = ma.array(T,mask=HotMask), ma.array(T,mask=WarmMask), ma.array(T,mask=ColdMask)
    
    NCold = (2.4e-3 * np.sqrt(xit) * (TCold/100)**(0.25) * np.sqrt(G0)) / (np.sqrt(Zd) * phiPAH)
    NWarm = (2.4e-3 * np.sqrt(xit) * (TWarm/100)**(0.25) * np.sqrt(G0)) / (np.sqrt(Zd) * phiPAH)
    NHot = rhoHot

    return ma.MaskedArray.filled(NCold,0), ma.MaskedArray.filled(NWarm,0), ma.MaskedArray.filled(NHot,0)
 

#Converts the Temperature data cube into an electron number data cube
NCold, NWarm, NHot = Ne(T,S,rho)

NTotal = NCold + NWarm + NHot
#Setting plot size and resolution
fig = plt.gcf()
fig.set_dpi(100)
fig.set_size_inches(10,16.18)
#ratio of 1:golden ratio :)
plt.pcolormesh(y,z,NTotal[:,96,:],cmap='Blues',norm=mc.LogNorm())
plt.colorbar()
plt.title(r'Electron Number Density $n_e$ per cm$^3$ excluding hot gas')
plt.xlabel(r'$y$ in pc')
plt.ylabel(r'$z$ in pc')
plt.show()

#Separate the three phases
#Hot gas, S > 23.2e8
#Warm gas, 23.2e8 > S > 3.7e8
#Cold gas, 3.7e8 > S
HotMask = ma.getmask(ma.masked_less(S,23.2e8))
WarmMask = ma.getmask(ma.masked_outside(S,3.7e8,23.3e8))
ColdMask = ma.getmask(ma.masked_greater(S,3.7e8))

SHot, SWarm, SCold = ma.array(S,mask=HotMask), ma.array(S,mask=WarmMask), ma.array(S,mask=ColdMask)
THot, TWarm, TCold = ma.array(T,mask=HotMask), ma.array(T,mask=WarmMask), ma.array(T,mask=ColdMask) 
rhoHot = ma.array(rho,mask=HotMask)












