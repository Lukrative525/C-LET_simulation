from matplotlib import pyplot as plt
import simulation.arraymodel as am
from numpy import linspace, radians
from scipy.optimize import minimize
from validation.joint_components import *
from validation.deflection_data import tests

# def sequentiallyMinimizeDeflectionError(array: am.LetArray, target_deflection: list, loading: list, step_sizes: list):

#     for i in range(100):
#         loading = takeStep(0, array, target_deflection, loading, step_sizes)
#         loading = takeStep(1, array, target_deflection, loading, step_sizes)
#         loading = takeStep(2, array, target_deflection, loading, step_sizes)

#     return loading

# def takeStep(index: int, array: am.LetArray, target_deflection: list, loading: list, step):

#     deflection = [0, 0, 0]

#     updateDeflection(array, loading, deflection)

#     initial_error = deflection[index] - target_deflection[index]

#     loading[index] += step

#     updateDeflection(array, loading, deflection)

#     new_error = deflection[index] - target_deflection[index]

#     if new_error < initial_error:
#         return loading
#     else:
#         loading[index] -= 2 * step

#     updateDeflection(array, loading, deflection)

#     new_error = deflection[index] - target_deflection[index]

#     if new_error < initial_error:
#         return loading
#     else:
#         loading[index] += step

#     return loading

# def updateDeflection(array: am.LetArray, loading: list, deflection: list):

#     array.calculateStaticsForward(*loading)
#     deflection[0], deflection[1] = array.getEndPosition()
#     deflection[2] = array.getEndRotation()



















def rootMeanSquareError(params, args):

    Rx, Ry, M = params
    target_x, target_y, target_theta, steps = args

    array.graduallyReposition(Rx, Ry, M, steps)

    actual_x, actual_y = array.getEndPosition()
    actual_theta = restrictAngle(array.getEndRotation())

    error = [actual_x - target_x, actual_y - target_y, actual_theta - target_theta]

    rms_error = rootMeanSquare(error)

    print(rms_error)

    return rms_error

def rootMeanSquare(values: list):

    rms = 0

    for value in values:
        rms += value ** 2

    rms = rms / len(values)

    rms = rms ** (1 / 2)

    return rms

E = am.msiToPa(9.8)
G = am.msiToPa(18.4)
Sy = am.msiToPa(120e-3)

available_length = 140e-3
num_series = 12
num_parallel = 2
bond_region_length = 20e-3
L = round((available_length - num_parallel * bond_region_length) / num_parallel, 9)
torsion_bar_width = 10.3e-3
torsion_bar_thickness = 0.018 * 25.4e-3
h = torsion_bar_width / 2
b = torsion_bar_thickness / 2

array = am.LetArray(b, h, L, E, G, Sy, num_series)

target_x = -0.0007868328311495939
target_y = 0.013409784928369698
target_theta = radians(182.66006427766135)

solution_steps = 10

Rx = 0.98
Ry = 1.23
M = -0.2172445396523411
initial_guess = [Rx, Ry, M]
target_deflection = [target_x, target_y, target_theta]

result = minimize(rootMeanSquareError, initial_guess, [target_x, target_y, target_theta, solution_steps], tol=1e-9)
Rx, Ry, M = result.x
print(result)

# loading = sequentiallyMinimizeDeflectionError(array, )

figure = plt.figure()
axes = figure.add_subplot()
axes.set_aspect("equal")

array.graduallyReposition(Rx, Ry, M, 10)
array.plotArray(axes)

joint: Joint = tests[0]
joint.scale(25.4e-3)
joint.rotateAndCenter()

# point_1, point_2 = joint.sections[0].majorAxis()
# print(point_2 - point_1)

plotJoint(axes, joint)

plt.show()