from matplotlib import pyplot as plt
from validation.visualization_components import *
from validation.deflection_measurements import tests
from numpy import degrees

joint: Joint = tests[0]
joint.scale(25.4e-3)
joint.rotateAndCenter()

figure = plt.figure()
axes = figure.add_subplot()
axes.set_aspect("equal")

translation_vector, rotation_angle = joint.getTotalDeflection(len(joint.sections) - 1)
print(translation_vector)
print(degrees(rotation_angle) + 360)

plotJoint(axes, joint)

plt.show()