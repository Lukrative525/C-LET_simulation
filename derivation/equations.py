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

equations: list[sp.Eq] = []

equations.append(sp.Eq(Rx + Fx, 0))
equations.append(sp.Eq(Ry + Fy, 0))
# equations.append(sp.Eq(M + T + Rx * (delta * sin(theta) + b * sin(2 * theta)) - Ry * (b + delta * cos(theta) + b * cos(2 * theta)), 0))
equations.append(sp.Eq(Rx + Fsx, 0))
equations.append(sp.Eq(Ry + Fsy, 0))
equations.append(sp.Eq(M + Ms - Ry * b, 0))
# equations.append(sp.Eq(Fx - Fsx, 0))
# equations.append(sp.Eq(Fy - Fsy, 0))
equations.append(sp.Eq(T - Ms + Fy * b * cos(2 * theta) - Fx * b * sin(2 * theta), 0))
equations.append(sp.Eq(delta, kb * (Fsx * cos(theta) + Fsy * sin(theta))))
equations.append(sp.Eq(theta, kt * Ms))

for i in range(len(equations)):
    equations[i] = equations[i].subs([(Rx, 0), (Ry, 0)])

solution = sp.solve(equations, [Fx, Fy, T, Fsx, Fsy, Ms, theta, delta])

for expression in solution[0]:
    print(expression)

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

equations: list[sp.Eq] = []

equations.append(sp.Eq(delta, 2 * h * sin(theta)))
equations.append(sp.Eq(Rx + Fx, 0))
equations.append(sp.Eq(Ry + Fy, 0))
equations.append(sp.Eq(M + T + Rx * (delta * sin(theta) + b * sin(2 * theta)) - Ry * (b + delta * cos(theta) + b * cos(2 * theta)), 0))
equations.append(sp.Eq(Rx + Fsx + Px, 0))
equations.append(sp.Eq(Ry + Fsy + Py, 0))
equations.append(sp.Eq(M + Ms - Ry * b - Px * h, 0))
# equations.append(sp.Eq(Fx - Fsx - Px, 0))
# equations.append(sp.Eq(Fy - Fsy - Py, 0))
equations.append(sp.Eq(T - Ms + Fy * b * cos(2 * theta) - Fx * b * sin(2 * theta) + Py * h * sin(2 * theta) + Px * h * cos(2 * theta), 0))
equations.append(sp.Eq(delta, kb * (Fsx * cos(theta) + Fsy * sin(theta))))
equations.append(sp.Eq(theta, kt * Ms))

for i in range(len(equations)):
    equations[i] = equations[i].subs([(Rx, 0), (Ry, 0)])

solution = sp.solve(equations, [Fx, Fy, T, Fsx, Fsy, Ms, Px, Py, theta, delta])

for expression in solution[0]:
    print(expression)

# %%
# case 3: same as case 2, but using small angle approximations

 #####      #      #####   #######            #####   
#     #    # #    #     #  #                 #     #  
#         #   #   #        #                       #  
#        #######   #####   ####                ####   
#        #     #        #  #                       #  
#     #  #     #  #     #  #                 #     #  
 #####   #     #   #####   #######            #####   

# approximations:
# sin(theta) = theta
# sin(2 * theta) = 2 * theta
# cos(theta) = 1
# cos(2 * theta) = 1

Rx, Ry, M, b, h, kb, kt, Fx, Fy, T, Fsx, Fsy, Px, Py, Fs, Ms, theta, delta = sp.symbols('R_x R_y M b h k_b k_t F_x F_y T F_sx F_sy P_x P_y F_s M_s theta delta')

equations: list[sp.Eq] = []

equations.append(sp.Eq(delta, 2 * h * sin(theta)))
equations.append(sp.Eq(Rx + Fx, 0))
equations.append(sp.Eq(Ry + Fy, 0))
equations.append(sp.Eq(M + T + Rx * (delta * sin(theta) + b * sin(2 * theta)) - Ry * (b + delta * cos(theta) + b * cos(2 * theta)), 0))
equations.append(sp.Eq(Rx + Fsx + Px, 0))
equations.append(sp.Eq(Ry + Fsy + Py, 0))
equations.append(sp.Eq(M + Ms - Ry * b - Px * h, 0))
# equations.append(sp.Eq(Fx - Fsx - Px, 0))
# equations.append(sp.Eq(Fy - Fsy - Py, 0))
equations.append(sp.Eq(T - Ms + Fy * b * cos(2 * theta) - Fx * b * sin(2 * theta) + Py * h * sin(2 * theta) + Px * h * cos(2 * theta), 0))
equations.append(sp.Eq(delta, kb * (Fsx * cos(theta) + Fsy * sin(theta))))
equations.append(sp.Eq(theta, kt * Ms))

for i in range(len(equations)):
    # equations[i] = equations[i].subs([(Rx, 0), (Ry, 0)])
    equations[i] = equations[i].subs([(sin(theta), theta), (cos(theta), 1), (sin(2 * theta), 2 * theta), (cos(2 * theta), 1)])

solution = sp.solve(equations, [Fx, Fy, T, Fsx, Fsy, Ms, Px, Py, theta, delta])

for expression in solution[0]:
    print(expression)
# %%
