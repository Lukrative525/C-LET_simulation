import numpy as np
from arraymodel import LetArray
from visualization import ArrayPlayer

def msiToPa(msi):

    psi = msi * 1e6
    Pa = psi * 6894.757

    return(Pa)

if __name__ == "__main__":

    E = msiToPa(9.8)
    G = msiToPa(18.4)
    Sy = msiToPa(120e-3)

    available_length = 101.6e-3
    num_series = 12
    num_parallel = 3
    bond_region_length = 10e-3
    L = (available_length - num_parallel * bond_region_length) / num_parallel
    torsion_bar_width = 10e-3
    torsion_bar_thickness = 0.018 * 25.4e-3
    h = torsion_bar_width / 2
    b = torsion_bar_thickness / 2

    array = LetArray(b, h, L, E, G, Sy, num_series)

    player = ArrayPlayer(array)