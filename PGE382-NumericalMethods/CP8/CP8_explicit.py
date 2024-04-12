#%%
from math import pi, sin, cos, exp
import numpy as np
from numpy import linspace, zeros, arange
from numpy import ix_ as ix
np.set_printoptions(threshold=10000, linewidth=10000)
from numpy import exp, linspace, vectorize
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

XMAX = 100
YMAX = 25
dx = XMAX/10 ; dy = YMAX/10
Tf = 80 ; Nt=160
dt = Tf/Nt ; Nt = Nt + 1
X = np.arange(0,XMAX+dx,dx) ; Ni = len(X)
Y = np.arange(0,YMAX+dy,dy) ; Nj = len(Y)
Nij = Ni * Nj

beta = 0.5
Pr = 0.733

# Global index
def _(i,j) : return j + Nj*i

#
# Assign BCs to solution vectors
# 
def init_bcs() :
    global Unij, Vnij, Tnij
    Unij = zeros( [Nt,Ni,Nj] )
    Vnij = zeros( [Nt,Ni,Nj] )
    Tnij = zeros( [Nt,Ni,Nj] )

    Unij[0,:,:] = 0 # ic
    Vnij[0,:,:] = 0 # ic
    Tnij[0,:,:] = 0 # ic

    Unij[:,:,0] = 0 # BC , Y=0
    Vnij[:,:,0] = 0 # BC , Y=0
    Tnij[:,:,0] = 1 # BC , Y=0

    Unij[:,:,-1] = 0 # BC , Y=inf
    Tnij[:,:,-1] = 0 # BC , Y=inf

    Unij[:,0,:] = 0 # BC , X=0
    Vnij[:,0,:] = 0 # BC , X=0
    Tnij[:,0,:] = 0 # BC , X=0


#
#
# 
#  MAIN FLOW
#
#
#

# Global solution vector
init_bcs()

for n in arange(1,Nt) :
    print(f"Solving timestep {n} ...")

    T = Tnij[n-1,:,:]
    U = Unij[n-1,:,:]
    V = Vnij[n-1,:,:]
    # Solve U
    for i in arange(1,Ni) :
        for j in arange(1,Nj-1) :
            Unij[n,i,j] = U[i,j] + dt * (
                            + T[i,j]                                # T
                            + (U[i,j-1]-2*U[i,j]+U[i,j+1])/dy/dy    # Uyy
                            - U[i,j]*(U[i,j]-U[i-1,j])/dx           # - U Ux
                            - V[i,j]*(U[i,j]-U[i,j-1])/dy           # - U Uy
                        )

    # Solve T
    for i in arange(1,Ni) :
        for j in arange(1,Nj-1) :
            Tnij[n,i,j] = T[i,j] + dt*(
                + 1/Pr*(T[i,j-1]-2*T[i,j]+T[i,j+1])/dy/dy           # Tyy/Pr
                - U[i,j]*(T[i,j]-T[i-1,j])/dx                       # -U Tx
                - V[i,j]*(T[i,j]-T[i,j-1])/dy                       # -V Ty
            )

    # Solve V
    for i in arange(1,Ni) :
        for j in arange(1,Nj) :
            Vnij[n,i,j] = Vnij[n,i,j-1] - dy/dx*( Unij[n,i,j] - Unij[n,i-1,j])


    
#%%
for n in arange(0,Nt,10) :
    fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=[10,5])    
    ax1.imshow( Tnij[n,:,:] )
    ax1.set_title(f"T @ {n}")
    ax2.imshow( Unij[n,:,:] )
    ax2.set_title(f"U @ {n}")
    ax3.imshow( Vnij[n,:,:] )
    ax3.set_title(f"V @ {n}")


# %%
plt.imshow( JAC > 0 )
