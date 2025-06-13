import numpy as np
from matplotlib import pyplot as plt
from simulation.arraymodel import LetArray
from simulation.visualization import ArrayPlayer
from scipy.optimize import minimize, fsolve

def deflectionError(parameters, args):

    [torque] = parameters

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

def stressError(let_array: LetArray, target_deflection):

    let_array.verbose = False

    required_torque = findTorqueRequiredForDeflection(target_deflection, 0, let_array)

    let_array.calculateStaticsForward(0, 0, -required_torque)
    max_bending = let_array.sigmaZZ(let_array.b, 0, 0)
    max_torsion = let_array.sigmaZY(let_array.b, 0, 0)
    max_von_mises = let_array.vonMisesStress3D(0, 0, max_bending, 0, 0, max_torsion)

    error = let_array.Sy - max_von_mises

    let_array.verbose = True

    return error

def findNumberTorsionSegmentsRequiredForDeflection(target_deflection, initial_guess, parameters: list):

    [b, h, L, E, G, Sy, gap] = parameters

    for series in range(initial_guess, initial_guess + 100):
        let_array = LetArray(b, h, L, E, G, Sy, series, gap)
        if stressError(let_array, target_deflection) > 0:
            break

    return series

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

def testSideLengthAndSeriesNumber(variables, args):

    side_length, series = variables
    torque, target_deflection, L, G, Sy, k1, A = args

    test = np.zeros(2)

    test[0] = side_length ** 4 - (series * torque * L) / (G * k1 * target_deflection)
    test[1] = Sy / np.sqrt(3) - (target_deflection * side_length * G * (1 - (8 * A / np.pi ** 2)) / (series * L))

    return test

def findSideLengthAndSeriesNumber(torque, target_deflection, initial_guess: list, parameters: list):

    L, G, Sy, k1, A = parameters

    side_length, series = fsolve(testSideLengthAndSeriesNumber, initial_guess, [torque, target_deflection, L, G, Sy, k1, A], xtol=1e-9)

    return [side_length, series]

if __name__ == "__main__":

    A = calculateAForSquare(10)
    k1 = calculatek1ForSquare(10)

    # material properties for generic steel:
    E = 200e9
    G = 80e9
    Sy = 350e6

    L = 40e-3
    torsion_bar_width = 10e-3
    torsion_bar_thickness = 0.2e-3
    h = torsion_bar_width / 2
    b = torsion_bar_thickness / 2

    target_deflection = np.pi

    series_C_LET = 5
    series_C_LET = findNumberTorsionSegmentsRequiredForDeflection(target_deflection, series_C_LET, [b, h, L, E, G, Sy, 0])

    array_C_LET = LetArray(b, h, L, E, G, Sy, series_C_LET)
    torque = findTorqueRequiredForDeflection(target_deflection, 0, array_C_LET)
    print(np.round(torque, 12))
    player_C_LET = ArrayPlayer(array_C_LET)

    side_length = np.cbrt(torque * (1 - 8 * A / np.pi ** 2) / (k1 * (Sy / np.sqrt(3))))
    series_LET = side_length ** 4 * G * k1 * target_deflection / (torque * L)

    array_LET = LetArray(side_length / 2, side_length / 2, L, E, G, Sy, int(np.ceil(series_LET)), side_length * (np.sqrt(2) - 1))
    player_LET = ArrayPlayer(array_LET)
