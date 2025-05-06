'''
This script was used in obtaining the statics equations used to calculate torsion bar deflections. Sympy seems incapable of solving the system of equations for case 2. If you run this, be prepared to wait a few minutes before the solver gives up and moves on.
'''

# %%

import sympy as sp
from sympy import sin, cos

# %%
# case 1: reaction forces act through center of outer edge

 #####      #      #####   #######              #     
#     #    # #    #     #  #                  ###     
#         #   #   #        #                    #     
#        #######   #####   ####                 #     
#        #     #        #  #                    #     
#     #  #     #  #     #  #                    #     
 #####   #     #   #####   #######           #######  

Rx, Ry, M, b, kb, kt, Fx, Fy, T, Fsx, Fsy, Fs, Ms, theta, delta = sp.symbols('R_x R_y M b k_b k_t F_x F_y T F_sx F_sy F_s M_s theta delta')

eq1 = sp.Eq(Rx + Fx, 0)
eq2 = sp.Eq(Ry + Fy, 0)
eq3 = sp.Eq(M + T + Rx * (delta * sin(theta) + b * sin(2 * theta)) - Ry * (b + delta * cos(theta) + b * cos(2 * theta)), 0)
eq4 = sp.Eq(Rx + Fsx, 0)
eq5 = sp.Eq(Ry + Fsy, 0)
eq6 = sp.Eq(M + Ms - Ry * b, 0)
eq7 = sp.Eq(Fx - Fsx, 0)
eq8 = sp.Eq(Fy - Fsy, 0)
eq9 = sp.Eq(T - Ms + Fy * b * cos(2 * theta) - Fx * b * sin(2 * theta), 0)
eq10 = sp.Eq(delta, kb * (Fsx * cos(theta) + Fsy * sin(theta)))
eq11 = sp.Eq(theta, kt * Ms)

Fx_expression = sp.solve(eq1, Fx)[0]
Fy_expression = sp.solve(eq2, Fy)[0]
Fsx_expression = sp.solve(eq4, Fsx)[0]
Fsy_expression = sp.solve(eq5, Fsy)[0]
Ms_expression = sp.solve(eq6, Ms)[0]
T_expression = sp.solve(eq9, T)[0]
delta_expression = sp.solve(eq10, delta)[0]
theta_expression = sp.solve(eq11, theta)[0]

solution = sp.solve([eq1, eq2, eq4, eq5, eq6, eq9, eq10, eq11], [Fx, Fy, T, Fsx, Fsy, Ms, theta, delta])

print(solution)

# %%
# case 2: reaction forces act through center of outer edge, and top edges are touching

 #####      #      #####   #######            #####   
#     #    # #    #     #  #                 #     #  
#         #   #   #        #                      #   
#        #######   #####   ####                 #     
#        #     #        #  #                  #       
#     #  #     #  #     #  #                 #        
 #####   #     #   #####   #######           #######  

Rx, Ry, M, b, h, kb, kt, Fx, Fy, T, Fsx, Fsy, Px, Py, Fs, Ms, theta, delta = sp.symbols('R_x R_y M b h k_b k_t F_x F_y T F_sx F_sy P_x P_y F_s M_s theta delta')

eq1 = sp.Eq(delta, 2 * h * sin(theta))
eq2 = sp.Eq(Rx + Fx, 0)
eq3 = sp.Eq(Ry + Fy, 0)
eq4 = sp.Eq(M + T + Rx * (delta * sin(theta) + b * sin(2 * theta)) - Ry * (b + delta * cos(theta) + b * cos(2 * theta)), 0)
eq5 = sp.Eq(Rx + Fsx + Px, 0)
eq6 = sp.Eq(Ry + Fsy + Py, 0)
eq7 = sp.Eq(M + Ms - Ry * b - Px * h, 0)
# eq8 = sp.Eq(Fx - Fsx - Px, 0)
# eq9 = sp.Eq(Fy - Fsy - Py, 0)
eq10 = sp.Eq(T - Ms + Fy * b * cos(2 * theta) - Fx * b * sin(2 * theta) + Py * h * sin(2 * theta) + Px * h * cos(2 * theta), 0)
eq11 = sp.Eq(delta, kb * (Fsx * cos(theta) + Fsy * sin(theta)))
eq12 = sp.Eq(theta, kt * Ms)

solution = sp.solve([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq10, eq11, eq12], [Fx, Fy, T, Fsx, Fsy, Ms, Px, Py, theta, delta])

print(solution)

# %%
# case 3: same as case 2, but using small angle approximations

 #####      #      #####   #######            #####   
#     #    # #    #     #  #                 #     #  
#         #   #   #        #                       #  
#        #######   #####   ####                ####   
#        #     #        #  #                       #  
#     #  #     #  #     #  #                 #     #  
 #####   #     #   #####   #######            #####   

# sin(theta) = theta
# sin(2 * theta) = 2 * theta
# cos(theta) = 1
# cos(2 * theta) = 1

A, Rx, Ry, M, b, h, kb, kt, Fx, Fy, T, Fsx, Fsy, Px, Py, Fs, Ms, theta, delta = sp.symbols('A R_x R_y M b h k_b k_t F_x F_y T F_sx F_sy P_x P_y F_s M_s theta delta')

eq1 = sp.Eq(delta, 2 * h * theta)
eq2 = sp.Eq(Rx + Fx, 0)
eq3 = sp.Eq(Ry + Fy, 0)
eq4 = sp.Eq(M + T + (Rx * delta * theta + 2 * b * theta) - Ry * (2 * b + delta), 0)
eq5 = sp.Eq(Rx + Fsx + Px, 0)
eq6 = sp.Eq(Ry + Fsy + Py, 0)
eq7 = sp.Eq(M + Ms - Ry * b - Px * h, 0)
# eq8 = sp.Eq(Fx - Fsx - Px, 0)
# eq9 = sp.Eq(Fy - Fsy - Py, 0)
eq10 = sp.Eq(T - Ms + Fy * b - Fx * b * 2 * theta + Py * h * 2 * theta + Px * h, 0)
eq11 = sp.Eq(delta, kb * (Fsx + Fsy * theta))
eq12 = sp.Eq(theta, kt * Ms)

print(sp.latex(eq1))
print(sp.latex(eq2))
print(sp.latex(eq3))
print(sp.latex(eq4))
print(sp.latex(eq5))
print(sp.latex(eq6))
print(sp.latex(eq7))
# print(sp.latex(eq8))
# print(sp.latex(eq9))
print(sp.latex(eq10))
print(sp.latex(eq11))
print(sp.latex(eq12))

solution = sp.solve([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq10, eq11, eq12], [Fx, Fy, T, Fsx, Fsy, Ms, Px, Py, theta, delta])

print(solution)
# %%
