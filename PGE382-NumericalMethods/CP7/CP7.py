#%%
from math import pi, sin, cos, exp
import numpy as np
from numpy import linspace, zeros, arange
from numpy import ix_ as ix
np.set_printoptions(threshold=10000, linewidth=10000)
from numpy import exp, linspace, vectorize
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

dx = 0.125 ; dy = dx ; dz = dx
Tf = 0.25 ; Nt=10
mu = 0.5

X = np.arange(0,1+dx,dx) ; Ni = len(X)
Y = np.arange(0,1+dy,dy) ; Nj = len(Y)
Z = np.arange(0,1+dz,dz) ; Nk = len(Z)

Nijk = Ni * Nj * Nk

dt = Tf/Nt ; Nt = Nt + 1

# Global index
def _(i,j,k) : return i + Ni*j+ Ni*Nj*k
def exact( t,x,y,z) :
    return 1/( 1+exp(x/2/mu + y/2/mu + z/2/mu - 3*t/4/mu) ) 

def build_exact() :
    global Uexact, Nt, Ni, Nj, Nk
    Uexact = np.zeros_like(U)
    for n in arange(1,Nt) :
        for i in arange(Ni) :
            for j in arange(Nj) :
                for k in arange(Nk) :
                        Uexact[n,i,j,k] = exact(n*dt,i*dx,j*dy,k*dz)

# The list of free dofs
def build_Df() :
    global Df, Nt, Ni, Nj, Nk

    Df=[]
    for i in arange(1,Ni-1) :
        for j in arange(1,Nj-1) :
            for k in arange(1,Nk-1) :
                Df.append( _(i,j,k) )

# Global solution vector
U = zeros( [Nt,Ni,Nj,Nk] )
build_exact()
build_Df()

for n in arange(1,Nt) :
    print(f"Solving timestep {n} ...")

    # Solution from the previous TS
    Un = U[n-1,:,:,:].copy()
    Uk = Un.copy()

    nk = 0 #newton loop index
    while(1) :      
        # Jacobians and functions
        J = zeros([ Nijk, Nijk ] )
        F = zeros( Nijk )

        for i in arange(1,Ni-1) :
            for j in arange(1,Nj-1) :
                for k in arange(1,Nk-1) :
                    _ijk = _(i,j,k)
                    _0jk = _(i-1,j,k)
                    _1jk = _(i+1,j,k)
                    _i0k = _(i,j-1,k)
                    _i1k = _(i,j+1,k)
                    _ij0 = _(i,j,k-1)
                    _ij1 = _(i,j,k+1)

                    J[_ijk,_ijk] += -2*Uk[i,j,k]*(1/dx + 1/dy + 1/dz) +\
                                     Uk[i-1,j,k]/dx + Uk[i,j-1,k]/dy + Uk[i,j,k-1]/dz +\
                                     (-2*mu)*(1/dx/dx + 1/dy/dy + 1/dz/dz) +\
                                     (-1/dt)

                    J[_ijk,_0jk] += Uk[i,j,k]/dx + mu/dx/dx
                    J[_ijk,_1jk] += mu/dx/dx
                    J[_ijk,_i0k] += Uk[i,j,k]/dy + mu/dy/dy
                    J[_ijk,_i1k] += mu/dy/dy
                    J[_ijk,_ij0] += Uk[i,j,k]/dz + mu/dz/dz
                    J[_ijk,_ij1] += mu/dz/dz

                    F[_ijk] += Un[i,j,k]/dt

        nk += 1
        if nk > 10 : break
                    

#%%
#
#
#
def linear_solve( J1u, J2u, J1v, J2v, F1, F2 ) :
    global Ni, free_dofs
    Kff_ix = ix( free_dofs, free_dofs )
    Ff_ix = ix( free_dofs )

    # Assemble global linear system
    K = zeros( [ 2*Ni, 2*Ni ] )
    F = zeros( 2*Ni )

    # Only free dofs - as we have forced the initial to the right value, the delta is zero (homogeneous)
    Kff = K[Kff_ix]
    Ff  = F[Ff_ix]   

    # Solve for the unknowns, update vectors
    df = np.linalg.solve( Kff, -Ff )

    # Collect solution 
    dUV = np.zeros_like(F);
    dUV[Ff_ix] = df
    
    err = np.linalg.norm(dUV)

    return dUk, dVk, err

#
# Solver
#
for n in arange(1,Nt) :
    print(f"Solving timestep {n} ...")

    # Solution from the previous TS
    Un = Uni[n-1,:].copy()
    Vn = Vni[n-1,:].copy()
    Uk = Un.copy()
    Vk = Vn.copy()

    # Newton loops
    k = 0
    err = 999
    while(1) :      
        # Jacobians and functions
        J1u = zeros([ Ni, Ni ] )
        J1v = zeros([ Ni, Ni ] )
        J2u = zeros([ Ni, Ni ] )
        J2v = zeros([ Ni, Ni ] )
        F1 = zeros(Ni) 
        F2 = zeros(Ni) 
        
        # Jacobians and residuals F
        for i in arange(1,Ni-1) :
            # Shortcuts
            h2 = dx*dx ; h=dx
            u = Uk[i]; v = Vk[i]
            u0 = Uk[i-1]; u1 = Uk[i+1] ; v0 = Vk[i-1]; v1 = Vk[i+1]
            du = u - u0   ; dv = v-v0 ; d2u = u1-2*u+u0 ; d2v = v1-2*v+v0
            v2 = v**2 ; u2=u**2

            F1[i] += (Un[i]-u) / dt
            F1[i] += v2 / h2 * d2u
            F1[i] += 2*v/h2 * dv * du
            F1[i] += -u*v + u2 + 10
    
            J1u[i,i]   += -2*v2/h2 + 2*v*dv/h2 - v + 2*u - 1/dt
            J1u[i,i-1] += v2/h2    - 2*v*dv/h2
            J1u[i,i+1] += v2/h2
    
            J1v[i,i]   += 2*v*d2u/h2 + 4*du*v/h2 - 2*v0/h2*du - u
            J1v[i,i-1] += -2*v/h2*du
            
            F2[i] += (Vn[i]-v)/dt
            F2[i] += u2/h2 * d2v
            F2[i] += 2*u*du*dv/h2
            F2[i] += d2u/h2
            F2[i] += u*v
            F2[i] += -v2
    
            J2u[i,i]   += 2*u*d2v/h2   +   4*u*dv/h2  -2*u0*dv/h2 -   2/h2  +  v
            J2u[i,i-1] += -2*u*dv/h2   +   1/h2
            J2u[i,i+1] += 1/h2
            
            J2v[i,i]   +=  -1/dt - 2*u2/h2   +   2*u*du/h2   +   u   +   -2*v
            J2v[i,i-1] +=  u2/h2   - 2*u*du/h2
            J2v[i,i+1] += u2/h2

        # Neumann BC @ x=1 (i=N)
        # u_x + sin(uv) = 1/2   @ x=1
        u = Uk[-1] ; v = Vk[-1] ; uv = u*v ;
        J1u[-1,-1] += 1/dx + v * cos(uv)
        J1u[-1,-2] += -1/dx
        J1v[-1,-1] += u * cos( uv )
        F1[-1] = ( u - Uk[-2] )/dx + sin(uv) - 1/2
        # v_x - cos(uv) = 1     @ x=1
        J2v[-1,-1] += 1/dx + u * sin( uv )
        J2v[-1,-2] += -1/dx
        J2u[-1,-1] += v * sin( uv )
        F2[-1] = ( v - Vk[-2] )/dx - cos(uv) - 1

        dUk, dVk, err = linear_solve( J1u, J2u, J1v, J2v, F1, F2 )
        Uk += dUk
        Vk += dVk

        # Finish newton loop?
        print(f"   Newton iteration #{k} ... (err={err:.3e})")
        if k > 50 : break   # max iterations ?
        if err < 1e-15 : break # min error ?
        # Continue
        k += 1

    # Left the newton loop, update solution
    Uni[n,:] = Uk.copy()
    Vni[n,:] = Vk.copy()

import matplotlib.pyplot as plt 
fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots( 2, 2, figsize=(15,10) )
for n in arange(0,Nt,1) :
    ax1.plot( X, Uni[n,:], label=f"t={T[n]:.2f}" )
    ax2.plot( X, Vni[n,:], label=f"t={T[n]:.2f}" )

for i in arange(0,Ni,5) :
    ax3.plot( T, Uni[:,i], label=f"x={X[i]:.2f}" )
    ax4.plot( T, Vni[:,i], label=f"x={X[i]:.2f}" )
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

ax1.set_title("U")
ax2.set_title("V")
ax1.set_xlabel("X")
ax2.set_xlabel("X")
ax3.set_xlabel("T")
ax4.set_xlabel("T")
fig.tight_layout()

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.imshow(Kff!=0, interpolation='none')