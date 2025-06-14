import numpy as np
from matplotlib import pyplot as plt
from simulation.arraymodel import LetArray
from simulation.visualization import ArrayPlayer
from scipy.optimize import minimize, fsolve

def deflectionError(params, args):

    [torque] = params

    let_array: LetArray = args[0]
    target_deflection = args[1]

    let_array.calculateStaticsForward(0, 0, -torque)
    actual_deflection = let_array.getEndRotation()
    error = target_deflection - actual_deflection

    return error ** 2

def findTorqueRequiredForDeflection(target_deflection, initial_guess, let_array: LetArray):

    let_array.verbose = False

    result = minimize(deflectionError, initial_guess, args=[let_array, target_deflection])
    required_torque = result.x[0]

    let_array.verbose = True

    return required_torque

def stressError(params, let_array: LetArray):

    [torque] = params

    let_array.verbose = False

    let_array.calculateStaticsForward(0, 0, -torque)
    max_bending = let_array.sigmaZZ(let_array.b, 0, 0)
    max_torsion = let_array.sigmaZY(let_array.b, 0, 0)
    max_von_mises = let_array.vonMisesStress3D(0, 0, max_bending, 0, 0, max_torsion)

    error = (let_array.Sy - max_von_mises) ** 2

    let_array.verbose = True

    return error

def findMaxTorqueAndDeflectionForTorsionSegment(b, h, L, E, G, Sy, gap):

    test_array = LetArray(b, h, L, E, G, Sy, 1, gap)
    result = minimize(stressError, 0, test_array)
    max_torque = abs(result.x[0])
    max_deflection = test_array.getEndRotation()

    return [max_torque, max_deflection]

def calculateAForSquare(terms=10):

    series_sum = 0
    for n in range(1, 2 * terms + 1, 2):
        summand = 1 / (n ** 2 * np.cosh(n * np.pi / 2))
        series_sum += summand * np.logical_not(np.isinf(summand))

    return series_sum

def calculatek1ForSquare(terms=10):

    series_sum = 0
    for n in range(1, 2 * terms + 1, 2):
        summand = np.tanh(n * np.pi / 2) / (n ** 5)
        series_sum += summand * np.logical_not(np.isinf(summand))

    k1 = (1 / 3) * (1 - (192 / np.pi ** 5) * series_sum)

    return k1

if __name__ == "__main__":

    A = calculateAForSquare(10)
    k1 = calculatek1ForSquare(10)

    # material properties for generic steel:
    E = 200e9
    G = 80e9
    Sy = 350e6

    L = 40e-3
    torsion_segment_width = 10e-3
    torsion_segment_thickness = 0.5e-3
    h = torsion_segment_width / 2
    b = torsion_segment_thickness / 2

    target_deflection = np.pi

    max_torque, max_deflection = findMaxTorqueAndDeflectionForTorsionSegment(b, h, L, E, G, Sy, 0)
    C_LET_series_number = int(np.ceil(target_deflection / max_deflection))
    C_LET_array = LetArray(b, h, L, E, G, Sy, C_LET_series_number, 0)

    C_LET_torque = findTorqueRequiredForDeflection(np.pi, max_torque, C_LET_array)
    print(np.round(C_LET_torque, 12))
    C_LET_player = ArrayPlayer(C_LET_array, moment_res=1e-12)

    side_length = np.cbrt(C_LET_torque * (1 - 8 * A / np.pi ** 2) / (k1 * (Sy / np.sqrt(3))))
    LET_series_number = int(np.ceil(side_length ** 4 * G * k1 * target_deflection / (C_LET_torque * L)))
    gap = side_length * (np.sqrt(2) * np.cos(np.pi / 4 - (target_deflection / LET_series_number)) - 1)
    LET_array = LetArray(side_length / 2, side_length / 2, L, E, G, Sy, LET_series_number, gap)

    LET_torque = findTorqueRequiredForDeflection(np.pi, max_torque, LET_array)
    print(np.round(LET_torque, 12))
    print(side_length / 2)
    LET_player = ArrayPlayer(LET_array, moment_res=1e-12)
