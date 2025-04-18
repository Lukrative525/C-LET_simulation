from typing import Self
from numpy import arctan2, sin, cos, pi, argmin
from matplotlib.axes import Axes

class Joint():

    def __init__(self):

        self.sections: list[Section] = []

    def addSection(self, section_to_add):

        if isinstance(section_to_add, Section):
            self.sections.append(section_to_add)
        else:
            new_section = Section()
            for point in section_to_add:
                new_section.addPoint(point)
            self.sections.append(new_section)

    def rotateAndCenter(self, angle=0):

        translation_vector, current_angle = self.getTotalDeflection(0)

        translation_vector.x = -translation_vector.x
        translation_vector.y = -translation_vector.y

        for section in self.sections:
            section.translate(translation_vector)

        angle_to_rotate = angle - current_angle

        for section in self.sections:
            section.rotate(angle_to_rotate)

    def scale(self, scale_factor):

        for section in self.sections:
            section.scale(scale_factor)

    def getTotalDeflection(self, index: int):

        if index + 1 > len(self.sections):
            raise Exception(f"index out of bounds")

        if index + 1 == len(self.sections):
            centroid_1 = self.sections[index - 1].centroid()
            centroid_2 = self.sections[index].centroid()
            translation_vector = centroid_2
        else:
            centroid_1 = self.sections[index].centroid()
            centroid_2 = self.sections[index + 1].centroid()
            translation_vector = centroid_1

        direction_vector = centroid_2 - centroid_1
        direction_angle = vectorAngle(direction_vector)
        major_axis_point_1, major_axis_point_2 = self.sections[index].majorAxis()
        major_axis_angle = vectorAngle(major_axis_point_2 - major_axis_point_1)
        possible_rotation_angles = [restrictAngle(major_axis_angle + pi / 2), restrictAngle(major_axis_angle - pi / 2)]
        differences = [abs(findDifferenceOfAngles(possible_rotation_angles[0], direction_angle)), abs(findDifferenceOfAngles(possible_rotation_angles[1], direction_angle))]
        rotation_angle = possible_rotation_angles[argmin(differences)]

        return translation_vector, rotation_angle

class Point():

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other: Self):

        new_point = Point(self.x + other.x, self.y + other.y)

        return new_point

    def __sub__(self, other: Self):

        new_point = Point(self.x - other.x, self.y - other.y)

        return new_point

    def __mul__(self, other):

        new_point = Point(self.x * other, self.y * other)

        return new_point

    def rotate(self, angle):

        new_x = self.x * cos(angle) - self.y * sin(angle)
        new_y = self.x * sin(angle) + self.y * cos(angle)

        self.x = new_x
        self.y = new_y

    def translate(self, vector: Self):

        self.x += vector.x
        self.y += vector.y

    def scale(self, scale_factor):

        self.x *= scale_factor
        self.y *= scale_factor

    def __str__(self):

        return_string = f"{self.x}, {self.y}"

        return return_string

class Section():

    def __init__(self):

        self.points: list[Point] = []

    def addPoint(self, point_to_add):

        if isinstance(point_to_add, Point):
            self.points.append(point_to_add)
        else:
            new_point = Point()
            new_point.x = point_to_add[0]
            new_point.y = point_to_add[1]
            self.points.append(new_point)

    def centroid(self):

        centroid = calculateCentroid(self.points)

        return centroid

    def majorAxis(self):

        """
        assumes a line connecting the midpoints of the two shortest segments forms the major axis of the section
        """

        if len(self.points) <= 2:
            return self.points[0], self.points[1]

        segment_lengths = []

        segment_lengths.append(vectorNorm(self.points[0] - self.points[-1]))
        for i in range(1, len(self.points)):
            segment_lengths.append(vectorNorm(self.points[i] - self.points[i - 1]))

        index_1, index_2 = findIndicesOfTwoSmallest(segment_lengths)

        midpoint_segment_1 = calculateCentroid([self.points[index_1], self.points[index_1 - 1]])
        midpoint_segment_2 = calculateCentroid([self.points[index_2], self.points[index_2 - 1]])

        return midpoint_segment_1, midpoint_segment_2

    def rotate(self, angle):

        for point in self.points:
            point.rotate(angle)

    def translate(self, vector: Point):

        for point in self.points:
            point.translate(vector)

    def scale(self, scale_factor):

        for point in self.points:
            point.scale(scale_factor)

def calculateCentroid(points_list: list[Point]) -> Point:

    centroid_x = 0
    centroid_y = 0
    for point in points_list:
        centroid_x += point.x
        centroid_y += point.y
    centroid_x /= len(points_list)
    centroid_y /= len(points_list)

    centroid = Point(centroid_x, centroid_y)

    return centroid

def loopPoints(values: list):

    if values[0] != values[-1]:
        values.append(values[0])

def vectorAngle(vector_1: Point):

    rotation_angle = arctan2(vector_1.y, vector_1.x)

    return rotation_angle

def restrictAngle(angle):

    '''
    restricts an angle to [-pi, pi] radians
    '''

    if angle < -pi:
        angle = angle % (2 * pi)
    elif angle > pi:
        angle = angle % (-2 * pi)

    return angle

def findDifferenceOfAngles(angle_1, angle_2):

    difference = ((angle_2 - angle_1 + pi) % (2 * pi)) - pi

    return difference

def vectorNorm(vector: Point):

    norm = (vector.x ** 2 + vector.y ** 2) ** (1 / 2)

    return norm

def plotJoint(axes: Axes, joint: Joint):

    for section in joint.sections:
        plotSection(axes, section)

def plotSection(axes: Axes, section: Section):

    x_values, y_values = convertPointsToListAndTranspose(section.points)

    if len(section.points) > 2:
        loopPoints(x_values)
        loopPoints(y_values)

    axes.plot(x_values, y_values, 'k')

    center = section.centroid()

    axes.plot(center.x, center.y, 'ko')

def plotLine(axes: Axes, points: list[Point]):

    x_values, y_values = convertPointsToListAndTranspose(points)

    axes.plot(x_values, y_values)

def convertPointsToListAndTranspose(points: list[Point]):

    x_values = []
    y_values = []

    for point in points:
        x_values.append(point.x)
        y_values.append(point.y)

    return x_values, y_values

def findIndicesOfTwoSmallest(values: list):

    if len(values) < 2:
        raise ValueError("List must contain at least two elements")

    index_1, index_2 = (0, 1) if values[0] < values[1] else (1, 0)

    for i in range(2, len(values)):
        if values[i] < values[index_1]:
            index_2, index_1 = index_1, i
        elif values[i] < values[index_2]:
            index_2 = i

    return index_1, index_2