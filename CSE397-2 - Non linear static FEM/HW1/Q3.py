#%%

import numpy as np
from scipy.optimize import fsolve
from sympy import *


# Derivatives
x = symbols(r'x', real=True)
N1, N2 = symbols(r'N_1 N_2', real=True)
d1, d2, d3 = symbols(r'd_1 d_2 d_3', real=True)
N1 = x * d1 / ( 10 - d1) - 0.5 * d2**2
N2 = d2 - d1
print("dN1/dd1")
display( diff( N1, d1) )
print("dN1/dd2")
display( diff( N1, d2) )
print("dN2/dd1")
display( diff( N2, d1) )
print("dN2/dd2")
display( diff( N2, d2) )

# Load steps
F1 = 0
delta_F1 = 0.25

x = 15
# x = 25

line_search_1 = False
line_search_2 = False
line_search_maxit = 10
modif_nr = False
line_search_fast_s = True
ls_s=1

# Load steps
d = np.array( [ 0. , 0. ] )
for n in range(40) :
    print(f"n={n} -- F1={F1:.3f}")
    F1 += delta_F1
    F2 = 0

    # Newton loop
    for i in range(15) :
        N1 = x * d[0] / (10 - d[0]) - 0.5 * d[1]**2
        N2 = d[1] - d[0]

        # Check convergence
        R1 = F1 - N1
        R2 = F2 - N2
        R = sqrt( R1**2 + R2**2 )
        if not i : 
            if R > 1e-5 : R0=R
            else : R0 = 1e-5

        if R > 1e50 : break  # Crash!
        print(f"{i:3d}: R={R:.5e} R/R0={R/R0:.5f}")
        
        if R/R0 < 1e-4 : break

        # Update the tangent in the first interacion only if modified NR is chosen
        if not modif_nr or not i :
            dN1_dd1 = d[0] * x / (10 - d[0])**2 + x / (10 - d[0])
            dN1_dd2 = - d[1]
            dN2_dd1 = -1
            dN2_dd2 = 1

        K = [ [ dN1_dd1, dN1_dd2],
              [ dN2_dd1, dN2_dd2 ] ]
        K = np.array(K)
        delta_d = np.linalg.inv(K).dot( [R1, R2] )

        if line_search_1 :
            ls_s = delta_d.dot( [R1,R2] ) / delta_d.dot(K.dot(delta_d))
        elif line_search_2 :
            def foo(s):
                global G0
                d_ = d + s * delta_d
                Fi1 = x * d_[0] / (10 - d_[0]) - 0.5 * d_[1]**2
                Fi2 = d_[1] - d_[0]
                f = delta_d.dot( np.array([F1,F2]) - np.array([Fi1,Fi2]) )

                print(f"     {f:.5e} s={s[0]:.4e} (G0={G0:.4e})")
                
                if line_search_fast_s :
                    if abs(f) < abs(G0/2) : 
                        # interrupt.
                        print("FAST S!")
                        return 0 
                
                return f
            G0 = 0
            G0 = foo([0])            
            print("Find s ...")
            ls_s = fsolve(foo,0, maxfev=line_search_maxit)
            print(f"ls_s: [{ls_s}]")
            test_f = abs(foo(ls_s))
            if (test_f >= abs(G0/2)) :
                print(f"[FAILED] Cannot find acceptable s in line search - test_f[{test_f:.4e} > abs(G0/2)[{abs(G0/2):.4e}]]")
                break
            print("[Done]")

        d += delta_d * ls_s
    
    print(d)


#%%
import matplotlib.pyplot as plt
import numpy as np
def cN1(x,d) :
    N1 = x * d[0] / (10 - d[0]) - 0.5 * d[1]**2
    return N1

d0 = np.arange(10)
d0 = [ d0, d0 ]
N1 = cN1(15,d0)
plt.plot( d0[0], N1 )