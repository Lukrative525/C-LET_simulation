import simulation.arraymodel as am
from numpy import linspace

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

target_x = -0.004979247795435776
target_y = 0.8657012066622016
target_theta = 1.602984235253494

def error(params, args):

    Rx, Ry, M = params
    target_x, target_y, target_theta = args

    steps = 50

    increments_Rx = linspace(0, Rx, steps)
    increments_Ry = linspace(0, Ry, steps)
    increments_M = linspace(0, M, steps)

    for i, j, k in zip(increments_Rx, increments_Ry, increments_M):
        array.calculateStaticsForward(i, j, k)

    actual_x, actual_y = array.getEndPosition()
    theta = array.getEndRotation()