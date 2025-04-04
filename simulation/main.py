from simulation.arraymodel import LetArray, msiToPa
from simulation.visualization import ArrayPlayer

if __name__ == "__main__":

    E = msiToPa(9.8)
    G = msiToPa(18.4)
    Sy = msiToPa(120e-3)

    available_length = 140e-3
    num_series = 12
    num_parallel = 2
    bond_region_length = 20e-3
    L = round((available_length - num_parallel * bond_region_length) / num_parallel, 9)
    torsion_bar_width = 10.3e-3
    torsion_bar_thickness = 0.018 * 25.4e-3
    h = torsion_bar_width / 2
    b = torsion_bar_thickness / 2

    array = LetArray(b, h, L, E, G, Sy, num_series)

    player = ArrayPlayer(array)