from matplotlib import pyplot as plt
import simulation.arraymodel as am
from validation.joint_components import *
from validation.deflection_data import tests

def positionError(model: am.LetArray, data: Joint):

    error = []

    for i in range(model.series):
        if i % 2 == 1:
            model_position = Point(*model.getPosition(i))
            data_position = data.sections[int((i + 1) / 2)].centroid()
            difference = model_position - data_position

            error.append(vectorNorm(difference))

    return error

def angularError(model: am.LetArray, data: Joint):

    error = []

    for i in range(model.series):
        if i % 2 == 1:
            model_rotation = model.getRotation(i)
            point_1, point_2 = data.sections[int((i + 1) / 2)].majorAxis()
            data_rotation = vectorAngle(point_2 - point_1)

            error.append(abs(findDifferenceOfAngles(model_rotation, data_rotation)))

    return error

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

# ===============================================
Fx = -0.8262
Fy = 0 # (ignored)
T = 0.20694
test_index = 0
# ===============================================

loading = [Fx, Fy, T]
array.graduallyReposition(Fx, Fy, T, 10)

figure = plt.figure()
axes = figure.add_subplot()
axes.set_aspect("equal")

array.plotArray(axes)

joint: Joint = tests[test_index]
joint.scale(25.4e-3)
joint.rotateAndCenter()
plotJoint(axes, joint)

print(positionError(array, joint))
print(angularError(array, joint))

plt.show()