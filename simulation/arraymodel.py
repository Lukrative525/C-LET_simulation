import numpy as np
from numpy import sin, cos, linspace
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.optimize import fsolve
import warnings
warning_behavior = 'always'

def msiToPa(msi):

    psi = msi * 1e6
    Pa = psi * 6894.757

    return Pa

def se2(angle, x, y):

    transformation_matrix = np.array([
        [ cos(angle), -sin(angle), x],
        [ sin(angle),  cos(angle), y],
        [          0,           0, 1]
    ])

    return transformation_matrix

def transformConstructor(b):

    def T(theta, extension):

        transformation_matrix_1 = se2(theta, 0, 0)

        transformation_matrix_2 = se2(theta, extension + 2 * b * cos(theta), 0)

        total_transformation_matrix = transformation_matrix_1 @ transformation_matrix_2

        return total_transformation_matrix

    return T

class LetArray():

    def __init__(self, b, h, L, E, G, Sy, series, gap=0):

        self.b = b
        self.h = h
        self.L = L
        self.E = E
        self.G = G
        self.Sy = Sy
        self.series = series
        self.gap = gap

        self.Ix = (2 * self.b) * (2 * self.h) ** 3 / 12
        self.Iy = (2 * self.h) * (2 * self.b) ** 3 / 12

        self.T = transformConstructor(self.b + self.gap / 2)
        self.calculateTorsionSpringConstant()
        self.calculateBendingSpringConstant()

        self.thetas = np.zeros(self.series)
        self.gammas = np.zeros(self.series)
        self.extensions = np.zeros(self.series)
        self.Rx = np.zeros(self.series + 1)
        self.Ry = np.zeros(self.series + 1)
        self.M = np.zeros(self.series + 1)
        self.Fx_mid = np.zeros(self.series + 1)
        self.Fy_mid = np.zeros(self.series + 1)
        self.T_mid = np.zeros(self.series + 1)
        self.calculateTransforms()
        self.calculateStaticsForward(0, 0, 0)

        self.cmap = plt.get_cmap("tab10")

    def setAngle(self, indices, angle):

        if hasattr(indices, '__len__'):
            if indices[1] == -1:
                indices[1] = self.series
            indices = range(indices[0], indices[1])
        else:
            indices = range(indices, indices + 1)

        for i in indices:
            self.thetas[i] = angle
            self.gammas[i] = 2 * angle

    def setExtension(self, indices, extension):

        if hasattr(indices, '__len__'):
            if indices[1] == -1:
                indices[1] = self.series
            indices = range(indices[0], indices[1])
        else:
            indices = range(indices, indices + 1)

        for i in indices:
            self.extensions[i] = extension

    def calculateTorsionSpringConstant(self, n=10):

        if self.h < self.b:
            long_side, short_side = self.b, self.h
        else:
            short_side, long_side = self.b, self.h

        left_factor = (192 / np.pi ** 5) * (short_side / long_side)
        right_factor = 0
        for n in range(1, 2 * n + 1, 2):
            summand = np.nan_to_num((1 / n ** 5) * np.tanh(n * np.pi * long_side / (2 * short_side)))
            right_factor += summand * np.logical_not(np.isinf(summand))

        k1 = (1 - left_factor * right_factor) / 3

        self.torsion_constant = self.L / (2 * self.G * k1 * (2 * long_side) * (2 * short_side) ** 3)

    def calculateBendingSpringConstant(self):

        self.bending_constant = self.L ** 3 / (12 * self.E * self.Iy)

    def calculateTransforms(self):

        if not hasattr(self, 'transforms'):
            self.transforms = np.zeros((self.series + 1, 3, 3))
            self.transforms[0] = se2(0, 0, 0)

        for i in range(self.series):
            self.transforms[i + 1] = self.transforms[i] @ self.T(self.thetas[i], self.extensions[i])

    def calculateStaticsForward(self, Rx, Ry, M):

        self.Rx[0] = Rx - np.sum(self.Fx_mid)
        self.Ry[0] = Ry - np.sum(self.Fy_mid)
        self.M[0] = M - np.sum(self.T_mid)

        for i in range(self.series):

            F_mid = np.array([self.Fx_mid[i], self.Fy_mid[i]])
            Rotation = se2(np.sum(self.gammas[:i]), 0, 0)[:2, :2]
            F_mid = F_mid @ Rotation

            Rx = self.Rx[i] + F_mid[0]
            Ry = self.Ry[i] + F_mid[1]
            M = self.M[i] + self.T_mid[i]

            # =====================================================================================

            Ms = Ry * (self.b + self.gap / 2) - M
            theta = self.torsion_constant * Ms
            Fsx = -Rx
            Fsy = -Ry
            delta = self.bending_constant * (Fsx * cos(theta) + Fsy * sin(theta))
            Fx = -Rx
            Fy = -Ry
            T = Fx * (self.b + self.gap / 2) * sin(2 * theta) - Fy * (self.b + self.gap / 2) * cos(2 * theta) + Ms

            if self.gap == 0 and delta < 2 * self.h * sin(abs(theta)):

                def interferenceStatics(X):

                    Fx, Fy, T, Fsx, Fsy, Ms, Px, Py, delta, theta = X

                    test = np.zeros(10)

                    test[0] = delta - 2 * self.h * sin(abs(theta))
                    test[1] = Rx + Fx
                    test[2] = Ry + Fy
                    test[3] = M + T + Rx * (delta * sin(theta) + self.b * sin(2 * theta)) - Ry * (self.b + delta * cos(theta) + self.b * cos(2 * theta))
                    test[4] = Rx + Fsx + Px
                    test[5] = Ry + Fsy + Py
                    test[6] = M + Ms - Ry * self.b - Px * self.h * np.sign(theta)
                    test[7] = T - Ms + Fy * self.b * cos(2 * theta) - Fx * self.b * sin(2 * theta) + ((Py * self.h * sin(2 * theta)) + (Px * self.h * cos(2 * theta))) * np.sign(theta)
                    test[8] = delta - self.bending_constant * (Fsx * cos(theta) + Fsy * sin(theta))
                    test[9] = theta - self.torsion_constant * Ms

                    return test

                Px = 0
                Py = 0
                initial_guess = [Fx, Fy, T, Fsx, Fsy, Ms, Px, Py, delta, theta]

                if theta == 0:
                    Fx = -Rx
                    Fy = -Ry
                    T = 2 * self.b * Ry - M
                    delta = 0
                else:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter(warning_behavior)
                        Fx, Fy, T, Fsx, Fsy, Ms, Px, Py, delta, theta = fsolve(interferenceStatics, initial_guess, xtol=1e-9)
                        error = interferenceStatics([Fx, Fy, T, Fsx, Fsy, Ms, Px, Py, delta, theta])
                        if np.any(np.abs(error) > 1e-9):
                            print(f"Error values: {error}")

                if delta <= 0:
                    Fx = -Rx
                    Fy = -Ry
                    T = 2 * self.b * Ry - M
                    delta = 0
                    theta = 0

            # =====================================================================================

            F = np.array([Fx, Fy])
            F = F @ se2(2 * theta, 0, 0)[:2, :2]
            self.Rx[i + 1] = -F[0]
            self.Ry[i + 1] = -F[1]
            self.M[i + 1] = -T

            self.setAngle(i, theta)
            self.setExtension(i, delta)

        self.calculateTransforms()

    def sigmaZX(self, x, y, index, n=10):

        gamma = self.gammas[index]
        left_factor = -16 * self.G * gamma * self.b / (self.L * np.pi ** 2)
        if hasattr(x, 'shape'):
            right_factor = np.zeros(x.shape)
        else:
            right_factor = 0
        for i in range(1, 2 * n + 1, 2):
            summand = np.nan_to_num((-1) ** ((i - 1) / 2) * np.cos(i * np.pi * x / (2 * self.b)) * np.sinh(i * np.pi * y / (2 * self.b)) / (i ** 2 * np.cosh(i * np.pi * self.h / (2 * self.b))))
            right_factor += summand * np.logical_not(np.isinf(summand))

        sigma_zx = left_factor * right_factor

        return sigma_zx

    def sigmaZY(self, x, y, index, n=10):

        gamma = self.gammas[index]
        left_factor = -16 * self.G * gamma * self.b / (self.L * np.pi ** 2)
        if hasattr(x, 'shape'):
            right_factor = np.zeros(x.shape)
        else:
            right_factor = 0
        for i in range(1, 2 * n + 1, 2):
            summand = np.nan_to_num((-1) ** ((i - 1) / 2) * np.sin(i * np.pi * x / (2 * self.b)) * np.cosh(i * np.pi * y / (2 * self.b)) / (i ** 2 * np.cosh(i * np.pi * self.h / (2 * self.b))))
            right_factor += summand * np.logical_not(np.isinf(summand))

        sigma_zy = (2 * self.G * gamma * x / self.L) + (left_factor * right_factor)

        return sigma_zy

    def sigmaZZ(self, x, z, index):

        deflection_max = self.extensions[index]
        if index % 2 == 0:
            bending_stress = (12 * deflection_max * self.E * x / self.L ** 3) * (z - self.L / 2)
        else:
            bending_stress = (12 * deflection_max * self.E * x / self.L ** 3) * (self.L / 2 - z)

        return bending_stress

    def vonMisesStress3D(self, sigma_1, sigma_2, sigma_3, sigma_12, sigma_13, sigma_23):

        '''
        Returns the von Mises stress of an arbitrary stress element.
        '''

        von_mises_stress = np.sqrt(((sigma_1 - sigma_2) ** 2 + (sigma_1 - sigma_3) ** 2 + (sigma_2 - sigma_3) ** 2) / 2 + 3 * (sigma_12 ** 2 + sigma_13 ** 2 + sigma_23 ** 2))

        return von_mises_stress

    def getMaxBending(self):

        bending_stresses = np.zeros(self.series)
        for i in range(self.series):
            bending_stresses[i] = max(self.sigmaZZ(self.b, self.L, i), self.sigmaZZ(-self.b, self.L, i))

        index_max = np.argmax(bending_stresses)
        max_bending = bending_stresses[index_max]

        return index_max, max_bending

    def getMaxTorsion(self):

        torsion_stresses = np.zeros(self.series)
        for i in range(self.series):
            torsion_stresses[i] = max(self.sigmaZY(self.b, 0, i), self.sigmaZX(0, self.h, i))

        index_max = np.argmax(torsion_stresses)
        max_torsion = torsion_stresses[index_max]

        return index_max, max_torsion

    def getMaxVonMises(self):

        bending_stresses_x_faces = np.zeros(self.series)
        torsion_stresses_x_faces = np.zeros(self.series)
        torsion_stresses_y_faces = np.zeros(self.series)

        bending_stresses_x_faces = np.zeros(self.series)
        for i in range(self.series):
            bending_stresses_x_faces[i] = max(self.sigmaZZ(self.b, self.L, i), self.sigmaZZ(-self.b, self.L, i))
            torsion_stresses_x_faces[i] = self.sigmaZY(self.b, 0, i)
            torsion_stresses_y_faces[i] = self.sigmaZX(0, self.h, i)

        zero = np.zeros(self.series)
        von_mises_x_faces = self.vonMisesStress3D(bending_stresses_x_faces, zero, zero, zero, zero, torsion_stresses_x_faces)
        von_mises_y_faces = self.vonMisesStress3D(zero, zero, zero, zero, torsion_stresses_y_faces, zero)

        von_mises_maxes = np.max([von_mises_x_faces, von_mises_y_faces], axis=0)

        index_max = np.argmax(von_mises_maxes)
        max_von_mises = von_mises_maxes[index_max]

        return index_max, max_von_mises

    def getPosition(self, index):

        return self.transforms[index][:2, 2]

    def getEndPosition(self):

        return self.getPosition(-1)

    def getRotation(self, index):

        return np.sum(self.gammas[:index])

    def getEndRotation(self):

        return np.sum(self.gammas)

    def getEndLoad(self):

        rotation = self.transforms[-1, :2, :2]
        end_force = np.array([[-self.Rx[-1]], [-self.Ry[-1]]])
        end_force = np.squeeze(rotation @ end_force)
        end_torque = -self.M[-1]

        load = np.array([end_force[0], end_force[1], end_torque])

        return load

    def getReactions(self):

        Rx = self.Rx[0]
        Ry = self.Ry[0]
        M = self.M[0]

        reactions = np.array([Rx, Ry, M])

        return reactions

    def calculateStaticsInverse(self, Fx, Fy, T, guess=None):

        def calculateError(parameters):

            [M] = parameters

            self.calculateStaticsForward(-Fx, -Fy, M)

            end_load = self.getEndLoad()

            error = T - end_load[2]

            return error

        if guess is not None:
            initial_guess = guess
        else:
            initial_guess = -T

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter(warning_behavior)
            M = fsolve(calculateError, initial_guess, xtol=1e-9)[0]
            error = calculateError([M])
            if abs(error) > 1e-9:
                print(f"M error: {error}")

        self.calculateStaticsForward(-Fx, -Fy, M)

    def plotArray(self, axes: Axes):

        scale = 1000

        F_length = (self.b + self.h) / 2
        axes.clear()

        major_ticks = np.arange(-40, 41, 5)
        minor_ticks = np.arange(-40, 41, 1)
        axes.tick_params(axis="both", which="major", labelsize=18)
        axes.set_xticks(major_ticks)
        axes.set_xticks(minor_ticks, minor=True)
        axes.set_yticks(major_ticks)
        axes.set_yticks(minor_ticks, minor=True)
        axes.grid(True, "both")
        axes.grid(which='minor', alpha=0.2)
        axes.grid(which='major', alpha=0.8)
        axes.set_axisbelow(True)

        for i in range(self.series):
            end_point = self.transforms[i, 0:2, 2]
            R = self.transforms[i, 0:2, 0:2]
            points = np.zeros((5, 2))
            points[0] = end_point + (R @ np.array([2 * self.b, self.h])).T
            points[1] = end_point + (R @ np.array([0, self.h])).T
            points[2] = end_point + (R @ np.array([0, -self.h])).T
            points[3] = end_point + (R @ np.array([2 * self.b, -self.h])).T
            points[4] = end_point + (R @ np.array([2 * self.b, self.h])).T
            axes.plot(scale * points[:, 0], scale * points[:, 1], color=self.cmap(i % 10), alpha=0.5)
            axes.fill(scale * points[:, 0], scale * points[:, 1], color=self.cmap(i % 10), alpha=0.5)
            end_point = self.transforms[i + 1, 0:2, 2]
            R = self.transforms[i + 1, 0:2, 0:2]
            points = np.zeros((5, 2))
            points[0] = end_point + (R @ np.array([0, self.h])).T
            points[1] = end_point + (R @ np.array([-2 * self.b, self.h])).T
            points[2] = end_point + (R @ np.array([-2 * self.b, -self.h])).T
            points[3] = end_point + (R @ np.array([0, -self.h])).T
            points[4] = end_point + (R @ np.array([0, self.h])).T
            axes.plot(scale * points[:, 0], scale * points[:, 1], color=self.cmap(i % 10), alpha=0.5)
            axes.fill(scale * points[:, 0], scale * points[:, 1], color=self.cmap(i % 10), alpha=0.5)
        axes.plot(scale * self.transforms[:, 0, 2], scale * self.transforms[:, 1, 2], color=self.cmap(0))
        axes.plot(scale * self.transforms[:, 0, 2], scale * self.transforms[:, 1, 2], 'o', color=self.cmap(0))
        for i in range(self.series):
            end_point = self.transforms[i, 0:2, 2]
            F_mid = np.array([self.Fx_mid[i], self.Fy_mid[i]])
            F_norm = np.linalg.norm(F_mid)
            if F_norm != 0:
                axes.arrow(scale * end_point[0], scale * end_point[1], scale * F_length * F_mid[0] / F_norm, scale * F_length * F_mid[1] / F_norm, width=scale * F_length / 50, head_width=scale * F_length / 20)
        axes.set_aspect(1)

    def graduallyReposition(self, Fx, Fy, T, steps):

        steps = max(steps, 2)

        increments_Rx = linspace(0, Fx, steps)
        increments_Ry = linspace(0, Fy, steps)
        increments_M = linspace(0, T, steps)

        for i, j, k in zip(increments_Rx, increments_Ry, increments_M):
            self.calculateStaticsInverse(i, j, k)