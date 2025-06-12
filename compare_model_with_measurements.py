from matplotlib import pyplot as plt
import simulation.arraymodel as am
from validation.deflection_measurements import tests
from validation.error_calculation import *
from validation.visualization_components import *
from numpy import array, degrees, set_printoptions

set_printoptions(legacy="1.25")

def positionError(model: am.LetArray, data: Joint):

    error = []

    for i in range(model.series):
        if i % 2 == 1:
            model_position = Point(*(model.getPosition(i)))
            data_position, data_rotation = data.getTotalDeflection(int((i + 1) / 2))
            difference = model_position - data_position

            error.append(vectorNorm(difference))

    return error

def angularError(model: am.LetArray, data: Joint):

    error = []

    for i in range(model.series):
        if i % 2 == 1:
            model_rotation = model.getRotation(i)
            data_position, data_rotation = data.getTotalDeflection(int((i + 1) / 2))

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

let_array = am.LetArray(b, h, L, E, G, Sy, num_series)

# ===============================================
Fx = 0.0244
Fy = 1.5769
T = 0.2134
test_index = 1
# ===============================================

loading = [Fx, Fy, T]
let_array.graduallyReposition(Fx, Fy, T, 10)

figure = plt.figure()
axes = figure.add_subplot()

let_array.plotArray(axes)

joint: Joint = tests[test_index]
joint.scale(25.4e-3)
joint.rotateAndCenter()

position_error = array(positionError(let_array, joint))
rotation_error = angularError(let_array, joint)

print("Position Error (mm):")
print(position_error * 1e3)
print("Rotation Error (deg):")
print(degrees(rotation_error))

rms_position_error = rootMeanSquareError(position_error)
rms_rotation_error = rootMeanSquareError(rotation_error)

print("RMS Position Error (mm):")
print(rms_position_error * 1e3)
print("RMS Position Error / Torsion Bar Width")
print(rms_position_error / torsion_bar_width)
print("RMS Rotation Error (deg)")
print(degrees(rms_rotation_error))

joint.scale(1000)
plotJoint(axes, joint)

plt.show()