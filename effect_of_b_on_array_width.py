import numpy as np
from matplotlib import pyplot as plt

def numberTorsionSegmentsGivenYieldStrength(half_thickness, half_width, length, total_deflection, shear_modulus, yield_strength, discrete_torsion_segments=True, n=10):

    if half_thickness > half_width:
        raise Exception("Unknown behavior when the thickness is greater than the width")

    number_torsion_segments = total_deflection * shear_stress_per_radian(half_thickness, half_width, length, shear_modulus) / (yield_strength / np.sqrt(3))
    if discrete_torsion_segments:
        number_torsion_segments = np.ceil(number_torsion_segments)

    return number_torsion_segments

def numberTorsionSegmentsGivenTorque():

    return 0

def shear_stress_per_radian(half_thickness, half_width, length, shear_modulus, n=10):

    series_sum = 0
    for i in range(1, 2 * n + 1, 2):
        summand = 1 / (i ** 2 * np.cosh(i * np.pi * half_width / (2 * half_thickness)))
        series_sum += summand * np.logical_not(np.isinf(summand))

    result = (2 * shear_modulus * half_thickness / length) - (16 * shear_modulus * half_thickness / (length * np.pi ** 2)) * series_sum

    return result

if __name__ == "__main__":

    # material properties for generic steel:
    E = 200e9
    G = 80e9
    Sy = 350e6

    num_series = 1
    num_parallel = 1
    L = 0.04
    torsion_bar_width = 10e-3
    torsion_bar_thickness = np.array(list(range(10, 101))) / 10000
    h = torsion_bar_width / 2
    b = torsion_bar_thickness / 2

    number_torsion_segments = np.zeros(len(b))
    widths = np.zeros(len(b))
    for i in range(len(widths)):
        number_torsion_segments[i] = numberTorsionSegmentsGivenYieldStrength(b[i], h, L, np.pi / 2, G, Sy, True)
        # number_torsion_segments[i] = numberTorsionSegmentsGivenYieldStrength(b[i], h, L, np.pi / 2, G, Sy, False)
        widths[i] = 2 * b[i] * number_torsion_segments[i]

    number_torsion_segments_to_width_ratio = number_torsion_segments[-1] / widths[-1]

    fig = plt.figure()
    axes = fig.add_subplot()

    x_ticks = [0, torsion_bar_thickness[0], torsion_bar_width]
    y_ticks = [0]
    axes.tick_params(axis="both", which="major")
    axes.set_xticks(x_ticks)
    axes.set_yticks(y_ticks)
    axes.grid(True, which="major", axis="both")
    axes.set_axisbelow(True)

    axes.step(torsion_bar_thickness, number_torsion_segments, where="post")
    # axes.plot(torsion_bar_thickness, number_torsion_segments)
    axes.plot(torsion_bar_thickness, widths * number_torsion_segments_to_width_ratio)

    plt.show()