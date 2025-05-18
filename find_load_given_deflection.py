from matplotlib import pyplot as plt
import simulation.arraymodel as am
from numpy import radians
from validation.visualization_components import *
from validation.deflection_measurements import tests

def sequentiallyMinimizeDeflectionError(iterations: int, array: am.LetArray, target_deflection: list, loading: list, step_sizes: list):

    for i in range(iterations):
        loading = takeStep(0, array, target_deflection, loading, step_sizes)
        loading = takeStep(1, array, target_deflection, loading, step_sizes)
        loading = takeStep(2, array, target_deflection, loading, step_sizes)

    return loading

def takeStep(index: int, array: am.LetArray, target_deflection: list, loading: list, step_sizes):

    deflection = [0, 0, 0]

    updateDeflection(array, loading, deflection)

    initial_error = abs(deflection[index] - target_deflection[index])

    loading[index] += step_sizes[index]

    updateDeflection(array, loading, deflection)

    new_error = abs(deflection[index] - target_deflection[index])

    if new_error < initial_error:
        return loading
    else:
        loading[index] -= 2 * step_sizes[index]

    updateDeflection(array, loading, deflection)

    new_error = abs(deflection[index] - target_deflection[index])

    if new_error < initial_error:
        return loading
    else:
        loading[index] += step_sizes[index]

    return loading

def updateDeflection(array: am.LetArray, loading: list, deflection: list):

    M = array.getReactions()[2]

    array.calculateStaticsInverse(loading[0], loading[1], loading[2], M)
    deflection[0], deflection[1] = array.getEndPosition()
    print(array.getEndPosition())
    deflection[2] = array.getEndRotation()

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
target_x = -0.0007868328311495939
target_y = 0.013409784928369698
target_theta = radians(182.66006427766135)
Fx = -0.8262
Fy = 0 # (ignored)
T = 0.2069
test_index = 0
step_sizes = [0.0001, 0.0, 0.0001]
# ===============================================

iterations = 10

loading = [Fx, Fy, T]
array.graduallyReposition(Fx, Fy, T, 10)

target_deflection = [target_x, target_y, target_theta]

loading = sequentiallyMinimizeDeflectionError(iterations, array, target_deflection, loading, step_sizes)
print(loading)

figure = plt.figure()
axes = figure.add_subplot()
axes.set_aspect("equal")

array.graduallyReposition(Fx, Fy, T, 10)
array.plotArray(axes)

joint: Joint = tests[test_index]
joint.scale(25.4e-3)
joint.rotateAndCenter()

plotJoint(axes, joint)

plt.show()