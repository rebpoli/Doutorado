#%%
from sympy import *

H, U1, p, s, rho, iota = symbols(r'H U1 p s \rho \iota', real=True)
cv, theta, gamma = symbols(r'C_v \theta \gamma', real=True)
p0,rho0 = symbols(r'p0 \rho0', real=True)
U1, U2, U3, U4, U5 = symbols(r'U1 U2 U3 U4 U5', cls=Function)

x, y, z = symbols(r'x y z', real=True)
U1 = U1(x,y,z)
U2 = U2(x,y,z)
U3 = U3(x,y,z)
U4 = U4(x,y,z)
U5 = U5(x,y,z)

E, nu = symbols(r'e nu', real=True)

rho = U1
u1 = U2 / rho
u2 = U3 / rho
u3 = U4 / rho
E = U5 / rho
u_sq = u1**2 + u2**2 + u3**2
iota = E - 1/2*(u_sq)
theta = iota / cv

p = (gamma - 1) * rho * iota
s = ln( p/p0 * (rho/rho0)**(-gamma))
H = -s * rho

nu = iota + p/rho - iota*s

V1, V2, V3, V4, V5 = symbols(r'V1 V2 V3 V4 V5', real=True)
V1 = diff(H, U1)
V2 = diff(H, U2)
V3 = diff(H, U3)
V4 = diff(H, U4)
V5 = diff(H, U5)

_V1, _V2, _V3, _V4, _V5 = symbols(r'_V1 _V2 _V3 _V4 _V5', real=True)
# Test equalities
_V1 = 1/iota * ( nu - 1/2 * u_sq )
_V2 = 1/iota * ( u1 )
_V3 = 1/iota * ( u2 )
_V4 = 1/iota * ( u3 )
_V5 = 1/iota * ( -1 )

print("Test equalities - should be all zeros.")
display( simplify(V1 - _V1) )
display( simplify(V2 - _V2) )
display( simplify(V3 - _V3) )
display( simplify(V4 - _V4) )
display( simplify(V5 - _V5) )

#%% EXERCISE 4.2i j")
import numpy as np

kappa, q1, q2, q3 = symbols(r'\kappa q_1 q_2 q_3', real=True)
F1, F2, F3 = symbols(r'F_1 F_2 F_3', real=True)
F1v, F2v, F3v = symbols(r'F_1^v F_2^v F_3^v', real=True)
F1h, F2h, F3h = symbols(r'F_1^h F_2^h F_3^h', real=True)

b1, b2, b3, r = symbols(r'b1 b2 b3 r', real=True)

mu, lamb = symbols(r'\mu \lambda')

eps = Matrix([ 
    [ 1/2*(diff(u1,x) + diff(u1,x)), 1/2*(diff(u1,y) + diff(u2,x)), 1/2*(diff(u1,z) + diff(u3,x)) ],
    [ 1/2*(diff(u2,x) + diff(u1,y)), 1/2*(diff(u2,y) + diff(u2,y)), 1/2*(diff(u2,z) + diff(u3,y)) ],
    [ 1/2*(diff(u3,x) + diff(u1,z)), 1/2*(diff(u3,y) + diff(u2,z)), 1/2*(diff(u3,z) + diff(u3,z)) ] ])

epskk = eps[0,0] + eps[1,1] + eps[2,2] 
T11 = 2 * mu * eps[0,0] + lamb*epskk
T12 = 2 * mu * eps[0,1] + lamb*epskk
T13 = 2 * mu * eps[0,2] + lamb*epskk
T21 = 2 * mu * eps[1,0] + lamb*epskk
T22 = 2 * mu * eps[1,1] + lamb*epskk
T23 = 2 * mu * eps[1,2] + lamb*epskk
T31 = 2 * mu * eps[2,0] + lamb*epskk
T32 = 2 * mu * eps[2,1] + lamb*epskk
T33 = 2 * mu * eps[2,2] + lamb*epskk
TAU = Matrix([[T11,T12,T13], [T21,T22,T23], [T31,T32,T33] ])

# theta_x, theta_y, theta_z = symbols(r"\theta_1 \theta_2 \theta_3")

V = Matrix( [ V1, V2, V3, V4, V5 ] )
U = Matrix( [ U1, U2, U3, U4, U5 ] )

F1v = Matrix([ 0, T11, T21, T31, T11*u1+T12*u2+T13*u3])
F2v = Matrix([ 0, T12, T22, T32, T21*u1+T22*u2+T23*u3])
F3v = Matrix([ 0, T13, T23, T33, T31*u1+T32*u2+T33*u3])

q1 = - -kappa * diff(theta,x)
q2 = - -kappa * diff(theta,y)
q3 = - -kappa * diff(theta,z)

F1h = Matrix([ 0, 0, 0, 0, -q1 ])
F2h = Matrix([ 0, 0, 0, 0, -q2 ])
F3h = Matrix([ 0, 0, 0, 0, -q3 ])

F1 = u1*U + p * Matrix( [0, 1, 0, 0, u1] )
F2 = u2*U + p * Matrix( [0, 0, 1, 0, u2] )
F3 = u3*U + p * Matrix( [0, 0, 0, 1, u3] )

Fs = Matrix( [ 0, rho*b1, rho*b2, rho*b3, rho*(b1*u1+b2*u2+b3*u3+r) ] )

print("Check the if V \cdot Fi,i = Huii (should be zero):")
vfii = V.dot(diff(F1,x)+diff(F2,y)+diff(F3,z))
Huii = diff(H*u1,x)+diff(H*u2,y)+diff(H*u3,z)
display( simplify( vfii - Huii ) )


print("Check the if V \cdot F1^visc = 0 (should all be zeros):")
display( simplify(V.dot(F1v)) )
display( simplify(V.dot(F2v)) )
display( simplify(V.dot(F3v)) )

print("Check the if V \cdot F1^heat = qi/cv/theta should all be zeros):")
display( simplify(V.dot(F1h) - q1/cv/theta ) ) 
display( simplify(V.dot(F2h) - q2/cv/theta ) ) 
display( simplify(V.dot(F3h) - q3/cv/theta ) ) 

print("Check V \cdot Fscript - rho*r/cv/theta):")
display( simplify(V.dot(Fs) + rho*r/cv/theta ) ) 

#%% EXERCISE 4.3
V = simplify(V)
Fh = Matrix([F1h, F2h, F3h])
Fv = Matrix([F1v, F2v, F3v])

Fhv = simplify( Fh + Fv )
D = 0
D += simplify( diff(V,x).dot(F1h + F1v) )
# D += simplify( diff(V,y).dot(F2h + F2v) )
# D += simplify( diff(V,z).dot(F3h + F3v) )

# eps = Matrix([ 
#     [ 1/2*(diff(u1,x) + diff(u1,x)), 1/2*(diff(u1,y) + diff(u2,x)), 1/2*(diff(u1,z) + diff(u3,x)) ],
#     [ 1/2*(diff(u2,x) + diff(u1,y)), 1/2*(diff(u2,y) + diff(u2,y)), 1/2*(diff(u2,z) + diff(u3,y)) ],
#     [ 1/2*(diff(u3,x) + diff(u1,z)), 1/2*(diff(u3,y) + diff(u2,z)), 1/2*(diff(u3,z) + diff(u3,z)) ] ])

# eps_kk = eps[0,0] + eps[1,1] + eps[2,2]
# eps[0,0] += eps_kk
# eps[1,1] += eps_kk
# eps[2,2] += eps_kk

# sol = eps.copy()
# for i in range(3) :
#     for j in range(3) :
#         sol[i,j] = eps[i,j] * eps[i,j]

# a, b, c = symbols("a b c", real=true)

# div_u_sq = simplify( ( diff(u1,x) + diff(u2,y) + diff(u3,z) )**2 )
# mod_grad_theta_sq = simplify( diff(theta,x)**2 + diff(theta,y)**2 + diff(theta,z)**2 )

# solve( D - a*sol - b*div_u_sq - c*mod_grad_theta_sq, a,b,c, dict=True)
# eps.dot(eps)
# %%
