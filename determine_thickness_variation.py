from validation.visualization_components import *
from validation.deflection_measurements import tests
from validation.error_calculation import *

thicknesses = []

test: Joint
section: Section
for test in tests:
    test_thicknesses = []
    for section in test.sections:
        point_1, point_2 = section.minorAxis()
        line = point_2 - point_1
        test_thicknesses.append(vectorNorm(line))
    thicknesses.append(test_thicknesses[1:-1])

# for test in thicknesses:
#     mean_thickness = sum(test) / len(test)
#     print(mean_thickness)
#     thickness_error = [mean_thickness - thickness for thickness in test]
#     rms_thickness_error = rootMeanSquareError(thickness_error)
#     print(f"RMS error (mm): {rms_thickness_error * 25.4}")

all_thicknesses = []
for test in thicknesses:
    all_thicknesses += test

mean_thickness = sum(all_thicknesses) / len(all_thicknesses)
thickness_error = [mean_thickness - thickness for thickness in all_thicknesses]
rms_thickness_error = rootMeanSquareError(thickness_error)
print(f"RMS error (mm): {rms_thickness_error * 25.4}")
print(f"Max thickness (mm): {max(all_thicknesses) * 25.4}")
print(f"Min thickness (mm): {min(all_thicknesses) * 25.4}")
print(f"Percent error: {100 * rms_thickness_error / mean_thickness}")