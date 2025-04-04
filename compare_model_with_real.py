from matplotlib import pyplot as plt
import simulation.arraymodel as am
from validation.joint_components import *
from validation.deflection_data import tests

E = am.msiToPa(9.8)
G = am.msiToPa(18.4)
Sy = am.msiToPa(120e-3)

available_length = 140e-3
num_series = 12
num_parallel = 2
bond_region_length = 20e-3
L = round((available_length - num_parallel * bond_region_length) / num_parallel, 9)
torsion_bar_width = 10e-3
torsion_bar_thickness = 0.018 * 25.4e-3
h = torsion_bar_width / 2
b = torsion_bar_thickness / 2

array = am.LetArray(b, h, L, E, G, Sy, num_series)

Fx = -0.8262
Fy = 0 # (ignored)
T = 0.20694
loading = [Fx, Fy, T]
array.graduallyReposition(Fx, Fy, T, 10)

figure = plt.figure()
axes = figure.add_subplot()
axes.set_aspect("equal")

array.plotArray(axes)

joint: Joint = tests[0]
joint.scale(25.4e-3)
joint.rotateAndCenter()
plotJoint(axes, joint)

plt.show()