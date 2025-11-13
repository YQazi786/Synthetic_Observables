import numpy as np
import pencil as pc
import h5py
import numpy.ma as ma



#Var input
var1 = pc.read.var('VAR10.h5',trimall=True,magic=['tt','bb'])
var2 = pc.read.var('VAR12.h5',trimall=True,magic=['tt','bb'])
var3 = pc.read.var('VAR14.h5',trimall=True,magic=['tt','bb'])
var4 = pc.read.var('VAR16.h5',trimall=True,magic=['tt','bb'])
par = pc.read.param()

#For NE we need entropy (ss), temperature (T) and rho
S1, T1, rho1 = var1.ss*par.unit_entropy, var1.tt*par.unit_temperature, var1.rho
S2, T2, rho2 = var2.ss*par.unit_entropy, var2.tt*par.unit_temperature, var2.rho
S3, T3, rho3 = var3.ss*par.unit_entropy, var3.tt*par.unit_temperature, var3.rho
S4, T4, rho4 = var4.ss*par.unit_entropy, var4.tt*par.unit_temperature, var4.rho

#For RM we need B
B1 = var1.bb*par.unit_magnetic
B2 = var1.bb*par.unit_magnetic
B3 = var1.bb*par.unit_magnetic
B4 = var1.bb*par.unit_magnetic

#For SR we need ecr
var1a = h5py.File('data/allprocs/VAR10.h5','r+')
var2a = h5py.File('data/allprocs/VAR11.h5','r+')
var3a = h5py.File('data/allprocs/VAR12.h5','r+')
var4a = h5py.File('data/allprocs/VAR13.h5','r+')

ncr1 = np.array(var1a['data/ecr'][2:802,2:194,2:194])*par.unit_energy_density
ncr2 = np.array(var2a['data/ecr'][2:802,2:194,2:194])*par.unit_energy_density
ncr3 = np.array(var3a['data/ecr'][2:802,2:194,2:194])*par.unit_energy_density
ncr4 = np.array(var4a['data/ecr'][2:802,2:194,2:194])*par.unit_energy_density



T = np.dstack([np.hstack([T1,T2]),np.hstack([T3,T4])])
S = np.dstack([np.hstack([S1,S2]),np.hstack([S3,S4])])
B = np.concatenate([np.dstack([B1,B2]),np.dstack([B3,B4])],axis=3)
ncr = np.dstack([np.hstack([ncr1,ncr2]),np.hstack([ncr3,ncr4])])
rho = np.dstack([np.hstack([rho1,rho2]),np.hstack([rho3,rho4])])

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

NCold, NWarm, NHot = Ne(T,S,rho)
NE = NCold + NWarm + NHot
plt.pcolormesh(NE[:,50,:])
plt.colorbar()
plt.show()

np.save('2x2 Cubes/NE 2x2.npy',NE)





'''
#the 9 individual arrays, make sure they are all the same size.
#the final grid looks like
#T1 T2 T3
#T4 T5 T6
#T7 T8 T9
#For example, we are using some var files
var1 = pc.read.var('VAR1.h5',trimall=True,magic=['tt'])
var2 = pc.read.var('VAR1.h5',trimall=True,magic=['tt'])
var3 = pc.read.var('VAR1.h5',trimall=True,magic=['tt'])
var4 = pc.read.var('VAR.h5',trimall=True,magic=['tt'])
var5 = pc.read.var('VAR.h5',trimall=True,magic=['tt'])
var6 = pc.read.var('VAR.h5',trimall=True,magic=['tt'])
var7 = pc.read.var('var.h5',trimall=True,magic=['tt'])
var8 = pc.read.var('var.h5',trimall=True,magic=['tt'])
var9 = pc.read.var('var.h5',trimall=True,magic=['tt'])

T1 = var1.tt
T2 = np.roll(var2.tt,100)
T3 = np.roll(var3.tt,-100)
T4 = var4.tt
T5 = np.roll(var5.tt,100)
T6 = np.roll(var6.tt,-100)
T7 = var7.tt
T8 = np.roll(var8.tt,100)
T9 = np.roll(var9.tt,-100)

#First combining three boxes in the x-direction
Ta = np.hstack([T1,T2,T3])
Tb = np.hstack([T4,T5,T6])
Tc = np.hstack([T7,T8,T9])

#Second combining the three slabs into a 3x3 cube
T = np.dstack([Ta,Tb,Tc])
'''
