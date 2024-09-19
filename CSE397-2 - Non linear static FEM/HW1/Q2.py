
#%%
from sympy import *
import numpy as np
x, xi, u = symbols(r'x \xi u', real=True)

d1, d2, d3 = symbols(r'd_1 d_2 d_3', real=True)

Xi = [ -1/sqrt(3), 1/sqrt(3) ]
W = [ 1, 1 ]

kappa = Function(r"\kappa")(u,x)

Na = [
1/2*xi*(xi-1),
1-xi*xi,
1/2*xi*(xi+1)
]

dN1_xi = diff(Na[0], xi)
dN2_xi = diff(Na[1], xi)
dN3_xi = diff(Na[2], xi)

# assuming element from 0-1 and a point in the middle
X = Na[1] * 0.5 + Na[2]
X = simplify(X)
print("dNe_dxi")
display(dN1_xi)
display(dN2_xi)
display(dN3_xi)

dx_dxi = diff(X,xi)
dxi_dx = 1/dx_dxi
print("dx_dxi")
display(dx_dxi)
print("dxi_dx")
display(dxi_dx)

def build_N_x( xi_ ) :
    # Derivatives in X space
    N1_x = dN1_xi.subs(xi, xi_) * dxi_dx
    N2_x = dN2_xi.subs(xi, xi_) * dxi_dx
    N3_x = dN3_xi.subs(xi, xi_) * dxi_dx
    return N1_x, N2_x, N3_x

db = [ d1, d2, d3 ]
ne = [ 0, 0, 0 ]

f = symbols(r'f_1 f_2 f_3', real=True)
h = symbols(r'h', real=True)
fe = [ 0, 0, 0 ]
dna_ddb = zeros( 3, 3)
for xi_, W_ in zip( Xi, W ) :
    Na_x_ = build_N_x(xi_)
    x_ = X.subs(xi, xi_)
    kappa_ = kappa.subs( x, x_ )

    dkappa_du = diff(kappa_, u)
    display(dkappa_du)
    print(kappa_)
    Na_ = [0, 0, 0]
    for a in range(3) : Na_[a] = Na[a].subs(xi, xi_)
    
    q = 0
    for b in range(3) : q += db[b] * Na_x_[b]
    for a in range(3) :
        ne[a] += W_ * dx_dxi * Na_x_[a] * q * kappa_
        fe[a] += W_ * dx_dxi * Na_[a] * f[a]

    for a in range(3) :
        for b in range(3) :
            dna_ddb[a,b] += W_ * dx_dxi * Na_x_[a] * Na_[b] * q * dkappa_du
            dna_ddb[a,b] += W_ * dx_dxi * kappa_ * Na_x_[a] * Na_x_[b]

    fe[0] += W_ * h * Na_[0]
    
print("n1, n2, n3")
for a in range(3) : ne[a] = simplify(ne[a])
for a in range(3) : display(ne[a])

print("f1, f2, f3")
for a in range(3) : fe[a] = simplify(fe[a])
for a in range(3) : display(fe[a])

print("dna_ddb")
for a in range(3) :
    for b in range(3) :
        dna_ddb[a,b] = simplify(dna_ddb[a,b])
        display(dna_ddb[a,b])


