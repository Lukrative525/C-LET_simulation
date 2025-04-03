from matplotlib import pyplot as plt
from joint_components import *
from deflection_data import tests
from numpy import degrees

joint: Joint = tests[0]
joint.rotateAndCenter()

figure = plt.figure()
axes = figure.add_subplot()
axes.set_aspect("equal")

for i in range(len(joint.sections) - 1):
    translation_vector, rotation_angle = joint.getTotalDeflection(i)
    # print(translation_vector)
    print(degrees(rotation_angle))
    axes.plot(translation_vector.x, translation_vector.y, 'o')

plotJoint(axes, joint)

plt.show()