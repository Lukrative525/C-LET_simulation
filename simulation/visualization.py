import numpy as np
from numpy.linalg import norm
from scipy.signal import convolve2d
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QDoubleSpinBox,
    QPushButton,
    QSizePolicy,
    QLabel,
    QToolButton,
    QMenu,
    QInputDialog,
    QComboBox,
    QStackedLayout,
    QSpacerItem,
    QSpinBox,
    QAbstractSpinBox
)
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import warnings
from arraymodel import LetArray

def getDecimals(number):

        '''
        returns the number of decimals places in the input number
        '''

        number_string = np.format_float_positional(number)
        decimal_index = (len(number_string) - 1)
        for i in range(len(number_string)):
            if number_string[i] == '.':
                decimal_index = i
        decimals = (len(number_string) - 1) - decimal_index

        return decimals

def roundToResolution(num, resolution):

        '''
        rounds the input num to the resolution res
        '''

        rounded_to_resolution = resolution * round(num / resolution)

        return rounded_to_resolution

class ArrayDeflectionVisualizer(QVBoxLayout):

    def __init__(self, let_array: LetArray, points, *args, **kwargs):
        super(QVBoxLayout, self).__init__(*args, **kwargs)

        self.let_array = let_array
        self.points_x, self.points_y = points

        # a figure instance to plot on
        self.figure = Figure()
        self.axes = self.figure.add_subplot()

        # this is the Canvas Widget that displays the figure
        # it takes the figure instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas)

        self.addWidget(self.toolbar)
        self.addWidget(self.canvas)

    def updatePlot(self):

        F_length = (self.let_array.b + self.let_array.h) / 2
        self.axes.clear()
        for i in range(self.let_array.series):
            end_point = self.let_array.transforms[i, 0:2, 2]
            R = self.let_array.transforms[i, 0:2, 0:2]
            points = np.zeros((5, 2))
            points[0] = end_point + (R @ np.array([2 * self.let_array.b, self.let_array.h])).T
            points[1] = end_point + (R @ np.array([0, self.let_array.h])).T
            points[2] = end_point + (R @ np.array([0, -self.let_array.h])).T
            points[3] = end_point + (R @ np.array([2 * self.let_array.b, -self.let_array.h])).T
            points[4] = end_point + (R @ np.array([2 * self.let_array.b, self.let_array.h])).T
            self.axes.plot(points[:, 0], points[:, 1], color=self.let_array.cmap(i % 10), alpha=0.5)
            self.axes.fill(points[:, 0], points[:, 1], color=self.let_array.cmap(i % 10), alpha=0.5)
            end_point = self.let_array.transforms[i + 1, 0:2, 2]
            R = self.let_array.transforms[i + 1, 0:2, 0:2]
            points = np.zeros((5, 2))
            points[0] = end_point + (R @ np.array([0, self.let_array.h])).T
            points[1] = end_point + (R @ np.array([-2 * self.let_array.b, self.let_array.h])).T
            points[2] = end_point + (R @ np.array([-2 * self.let_array.b, -self.let_array.h])).T
            points[3] = end_point + (R @ np.array([0, -self.let_array.h])).T
            points[4] = end_point + (R @ np.array([0, self.let_array.h])).T
            self.axes.plot(points[:, 0], points[:, 1], color=self.let_array.cmap(i % 10), alpha=0.5)
            self.axes.fill(points[:, 0], points[:, 1], color=self.let_array.cmap(i % 10), alpha=0.5)
        self.axes.plot(self.let_array.transforms[:, 0, 2], self.let_array.transforms[:, 1, 2], color=self.let_array.cmap(0))
        self.axes.plot(self.let_array.transforms[:, 0, 2], self.let_array.transforms[:, 1, 2], 'o', color=self.let_array.cmap(0))
        for i in range(self.let_array.series):
            end_point = self.let_array.transforms[i, 0:2, 2]
            F_mid = np.array([self.let_array.Fx_mid[i], self.let_array.Fy_mid[i]])
            F_norm = norm(F_mid)
            if F_norm != 0:
                self.axes.arrow(end_point[0], end_point[1], F_length * F_mid[0] / F_norm, F_length * F_mid[1] / F_norm, width=F_length / 50, head_width=F_length / 20)
        self.axes.plot(self.points_x, self.points_y, '-', color='black')
        self.axes.plot(self.points_x, self.points_y, 'o', color='black')
        self.axes.set_aspect(1)
        self.canvas.draw()

class TorsionBarStressVisualizer(QVBoxLayout):

    def __init__(self, array: LetArray, mesh_spacing, *args, **kwargs):
        super(QVBoxLayout, self).__init__(*args, **kwargs)

        self.array = array
        self.mesh_spacing = mesh_spacing
        self.gnomon_length = 2 * max(self.array.b, self.array.h)
        self.first_update = True
        self.generateMesh()

        self.top = QHBoxLayout()
        self.bottom = QVBoxLayout()

        self.calculate_button = QPushButton()
        self.calculate_button.setText('Show Stresses')
        self.calculate_button.setCheckable(True)
        self.calculate_button.clicked.connect(self.updatePlot)
        self.calculate_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        self.stress_selector = QComboBox()
        self.stress_labels = ['Bending', 'Torsion', 'Von Mises']
        self.stress_selector.addItems(self.stress_labels)
        self.stress_selector.setCurrentIndex(2)
        self.stress_selector.currentIndexChanged.connect(self.updatePlot)
        self.stress_selector.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        self.torsion_bar_selector = QComboBox()
        self.torsion_bar_labels = []
        for i in range(1, self.array.series + 1):
            self.torsion_bar_labels.append(f'Torsion Bar {i}')
        self.torsion_bar_selector.addItems(self.torsion_bar_labels)
        self.torsion_bar_selector.currentIndexChanged.connect(self.updatePlot)
        self.torsion_bar_selector.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        self.top.addWidget(self.calculate_button)
        self.top.addWidget(self.stress_selector)
        self.top.addWidget(self.torsion_bar_selector)
        self.top.addStretch(1)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.axes = self.figure.add_subplot(projection='3d')

        self.bottom.addWidget(self.torsion_bar_selector)
        self.bottom.addWidget(self.toolbar)
        self.bottom.addWidget(self.canvas)

        self.addLayout(self.top)
        self.addLayout(self.bottom)

    def generateMesh(self):

        x_cell_count = min(21, max(2, int(2 * self.array.b / self.mesh_spacing) + 1))
        y_cell_count = min(21, max(2, int(2 * self.array.h / self.mesh_spacing) + 1))
        z_cell_count = min(21, max(2, int(self.array.L / self.mesh_spacing) + 1))

        x = np.linspace(-self.array.b, self.array.b, x_cell_count)
        y = np.linspace(-self.array.h, self.array.h, y_cell_count)
        z = np.linspace(0, self.array.L, z_cell_count)

        self.ux, self.vx = np.meshgrid(y, z)
        self.uy, self.vy = np.meshgrid(x, z)
        self.uz, self.vz = np.meshgrid(x, y)

        # left face
        self.x_left = -self.array.b * np.ones(self.ux.shape)
        self.y_left = self.ux
        self.z_left = self.vx

        # right face
        self.x_right = self.array.b * np.ones(self.ux.shape)
        self.y_right = self.ux
        self.z_right = self.vx

        # front face
        self.x_front = self.uy
        self.y_front = -self.array.h ** np.ones(self.uy.shape)
        self.z_front = self.vy

        # back face
        self.x_back = self.uy
        self.y_back = self.array.h ** np.ones(self.uy.shape)
        self.z_back = self.vy

        # bottom face
        self.x_bottom = self.uz
        self.y_bottom = self.vz
        self.z_bottom = np.zeros(self.uz.shape)

        # top face
        self.x_top = self.uz
        self.y_top = self.vz
        self.z_top = self.array.L * np.ones(self.uz.shape)

    def regenerateMesh(self, mesh_spacing):

        self.mesh_spacing = mesh_spacing
        self.generateMesh()

    def deflection(self, x, y, z, index):

        x, y, z = self.deflectionTorsion(x, y, z, index)
        x, y, z = self.deflectionBending(x, y, z, index)

        return x, y, z

    def deflectionTorsion(self, x, y, z, index):

        gamma = self.array.gammas[index]

        if index % 2 == 0:
            gamma_z = gamma * (z / self.array.L)
        else:
            gamma_z = gamma * (1 - z / self.array.L)

        x_deflected = x * np.cos(gamma_z) - y * np.sin(gamma_z)
        y_deflected = x * np.sin(gamma_z) + y * np.cos(gamma_z)

        return x_deflected, y_deflected, z

    def deflectionBending(self, x, y, z, index):

        deflection_max = self.array.extensions[index]
        theta = self.array.thetas[index]

        deflection_z = -(6 * deflection_max / self.array.L ** 3) * ((z ** 3 / 3) - (self.array.L * z ** 2 / 2))
        if index % 2 == 0:
            deflection_z = deflection_z
            x_deflected = x + deflection_z * np.cos(theta)
            y_deflected = y + deflection_z * np.sin(theta)
        else:
            deflection_z = deflection_max - deflection_z
            x_deflected = x + deflection_z * np.cos(theta)
            y_deflected = y + deflection_z * np.sin(theta)

        return x_deflected, y_deflected, z

    def updatePlot(self):

        if not self.first_update:
            azimuth=self.axes.azim
            elevation=self.axes.elev
            xlim=self.axes.get_xlim3d()
            ylim=self.axes.get_ylim3d()
            zlim=self.axes.get_zlim3d()

        index = self.torsion_bar_selector.currentIndex()

        if self.calculate_button.isChecked():
            stress_type = self.stress_selector.currentIndex()
            if stress_type == 0:
                index_max, max_bending = self.array.getMaxBending()
                color_range_limit = max(self.array.Sy, max_bending)
            elif stress_type == 1:
                index_max, max_torsion = self.array.getMaxTorsion()
                color_range_limit = max(self.array.Sy / np.sqrt(3), max_torsion)
            elif stress_type == 2:
                index_max, max_von_mises = self.array.getMaxVonMises()
                color_range_limit = max(self.array.Sy, max_von_mises)
        else:
            color_range_limit = None
 
        self.axes.clear()

        # left face
        self.plotFace(self.x_left, self.y_left, self.z_left, index, color_range_limit)

        # right face
        self.plotFace(self.x_right, self.y_right, self.z_right, index, color_range_limit)

        # front face
        self.plotFace(self.x_front, self.y_front, self.z_front, index, color_range_limit)

        # back face
        self.plotFace(self.x_back, self.y_back, self.z_back, index, color_range_limit)

        # bottom face
        self.plotFace(self.x_bottom, self.y_bottom, self.z_bottom, index, color_range_limit)

        # top face
        self.plotFace(self.x_top, self.y_top, self.z_top, index, color_range_limit)

        self.axes.plot([0, self.gnomon_length], [0, 0], [0, 0], color='red')
        self.axes.plot([0, 0], [0, self.gnomon_length], [0, 0], color='green')
        self.axes.plot([0, 0], [0, 0], [0, self.gnomon_length], color='blue')
        self.axes.set_axis_off()
        self.axes.patch.set_edgecolor('black')
        self.axes.patch.set_linewidth(1)
        self.axes.set_aspect('equal')

        if not self.first_update:
            self.axes.view_init(elev=elevation, azim=azimuth)
            self.axes.set_xlim3d(xlim)
            self.axes.set_ylim3d(ylim)
            self.axes.set_zlim3d(zlim)
        else:
            self.first_update = False

        self.canvas.draw()

    def plotFace(self, x, y, z, index, color_range_limit=None):

        color = self.array.cmap(index % 10)

        if self.calculate_button.isChecked():

            stress_type = self.stress_selector.currentIndex()

            if color_range_limit is None:
                color_range_limit = self.array.Sy

            if stress_type == 0:
                bending_stresses = self.array.sigmaZZ(x, z, index)
                normalization = matplotlib.colors.Normalize(-color_range_limit, color_range_limit)
                stresses = bending_stresses
            elif stress_type == 1:
                torsion_stresses_x = self.array.sigmaZX(x, y, index)
                torsion_stresses_y = self.array.sigmaZY(x, y, index)
                torsion_stresses = norm([torsion_stresses_x, torsion_stresses_y], axis=0)
                normalization = matplotlib.colors.Normalize(0, color_range_limit)
                stresses = torsion_stresses
            elif stress_type == 2:
                bending_stresses = self.array.sigmaZZ(x, z, index)
                torsion_stresses_x = self.array.sigmaZX(x, y, index)
                torsion_stresses_y = self.array.sigmaZY(x, y, index)
                von_mises_stresses = self.array.vonMisesStress3D(np.zeros(bending_stresses.shape), np.zeros(bending_stresses.shape), bending_stresses, np.zeros(bending_stresses.shape), torsion_stresses_x, torsion_stresses_y)
                normalization = matplotlib.colors.Normalize(0, color_range_limit)
                stresses = von_mises_stresses

            m = plt.cm.ScalarMappable(norm=normalization, cmap='jet')
            facecolors = m.to_rgba(self.calculateFaceValues(stresses))
            x, y, z = self.deflection(x, y, z, index)
            self.axes.plot_surface(x, y, z, facecolors=facecolors)
        else:
            x, y, z = self.deflection(x, y, z, index)
            edgecolor = np.array(color) / 2
            self.axes.plot_surface(x, y, z, color=color, edgecolor=edgecolor)

    def calculateFaceValues(self, input_data):

        face_values = convolve2d(input_data, np.ones((2, 2)) / 4)[1:-1, 1:-1]

        return face_values

class ArrayPlayer:

    def __init__(self, array: LetArray, force_res=1, moment_res = 0.01, trans_res=0.00001, ang_res=0.001):

        '''
        array: LET_array object
        ang_res: resolution of the revolute joints, in degrees
        trans_res: resolution of the prismatic joint, in meters

        Description
        ===========
        A 2D graphics environment with a GUI for visualizing a Compact LET Array. The sliders (and associated buttons) allow the user to vary the loads on the array, while the spinboxes (and their associated buttons) allow manipulation of the end point in cartesian space using inverse kinematics.

        The arrow buttons allow for rapid movement of the robot, while the +/- buttons allow step input.
        '''

        # rename attributes from array
        self.array = array

        # profile points
        self.profile_points_x = []
        self.profile_points_y = []
        self.points = [self.profile_points_x, self.profile_points_y]

        # resolution parameters
        self.force_res = force_res
        self.moment_res = moment_res
        self.trans_res = trans_res
        self.ang_res = ang_res

        self.force_disp_res = getDecimals(force_res)
        self.moment_disp_res = getDecimals(moment_res)
        self.trans_disp_res = getDecimals(trans_res)
        self.ang_disp_res = getDecimals(ang_res)

        # OTHER PARAMETERS
        autoRepeatInterval = 100 # milliseconds

        # CONFIGURE APPLICATION WINDOW
        if QApplication.instance() is None:
            app = QApplication([])
        else:
            app = QApplication.instance()
        window = QMainWindow()
        window.setGeometry(50, 50, 1400, 800)
        window.setWindowTitle("Compact LET Array")

        main_layout = QHBoxLayout()

        torsion_bar_visualizer_container = QWidget()
        self.torsion_bar_visualizer = TorsionBarStressVisualizer(self.array, self.array.b / 5)
        torsion_bar_visualizer_container.setLayout(self.torsion_bar_visualizer)
        array_visualizer_container = QWidget()
        self.array_visualizer = ArrayDeflectionVisualizer(self.array, self.points)
        array_visualizer_container.setLayout(self.array_visualizer)
        controls = QVBoxLayout()
        top_controls = QHBoxLayout()
        bottom_controls = QHBoxLayout()
        left_controls = QVBoxLayout()
        right_controls = QVBoxLayout()
        top_controls.addLayout(left_controls)
        top_controls.addLayout(right_controls)

#         #####      #     ######
#        #     #    # #    #     #
#        #     #   #   #   #     #
#        #     #  #######  #     #
#        #     #  #     #  #     #
#        #     #  #     #  #     #
#######   #####   #     #  ######

 #####    #####   #     #  #######  ######    #####   #         #####
#     #  #     #  ##    #     #     #     #  #     #  #        #     #
#        #     #  # #   #     #     #     #  #     #  #        #
#        #     #  #  #  #     #     ######   #     #  #         #####
#        #     #  #   # #     #     #   #    #     #  #              #
#     #  #     #  #    ##     #     #    #   #     #  #        #     #
 #####    #####   #     #     #     #     #   #####   #######   #####

        # CONFIGURE LOAD CONTROLS
        load_controls_label = QLabel()
        load_controls_label.setText('End Load')
        load_controls_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        right_controls.addWidget(load_controls_label)

        self.load_controls_list = []
        left_load_controls_button_managers_list = []
        right_load_controls_button_managers_list = []
        left_load_controls_step_managers_list = []
        right_load_controls_step_managers_list = []
        lc_labels = [
            'Fx',
            'Fy',
            'T'
        ]
        for i in range(len(lc_labels)):

            # info display
            info_display = QHBoxLayout()

            # text
            text = QLabel()
            text.setText(lc_labels[i])
            text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
            text.setFixedWidth(40)

            # spinbox
            spinbox = CustomDoubleSpinBox()
            if i < 2:
                spinbox.setMaximum(1000000 * self.force_res)
                spinbox.setMinimum(-1000000 * self.force_res)
                spinbox.setSingleStep(self.force_res)
                spinbox.setDecimals(self.force_disp_res)
                spinbox.setSuffix(' N')
            else:
                spinbox.setMaximum(1000000 * self.moment_res)
                spinbox.setMinimum(-1000000 * self.moment_res)
                spinbox.setWrapping(False)
                spinbox.setSingleStep(self.moment_res)
                spinbox.setDecimals(self.moment_disp_res)
                spinbox.setSuffix(' N-m')
            spinbox.valueChanged.connect(self.updateStatics)
            spinbox.setKeyboardTracking(False)
            self.load_controls_list.append(spinbox)

            info_display.addWidget(text)
            info_display.addWidget(spinbox)

            # buttons
            buttons = QHBoxLayout()

            button_left = QToolButton()
            button_left.setAutoRepeat(True)
            button_left.setAutoRepeatInterval(autoRepeatInterval)
            button_left.setAutoRepeatDelay(autoRepeatInterval)
            button_left.setArrowType(Qt.ArrowType.LeftArrow)
            button_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            if i < 2:
                button_left_manager = ButtonManager(spinbox)
            else:
                button_left_manager = ButtonManager(spinbox)
            left_load_controls_button_managers_list.append(button_left_manager)
            button_left.pressed.connect(left_load_controls_button_managers_list[i].decrement)

            button_right = QToolButton()
            button_right.setAutoRepeat(True)
            button_right.setAutoRepeatInterval(autoRepeatInterval)
            button_right.setAutoRepeatDelay(autoRepeatInterval)
            button_right.setArrowType(Qt.ArrowType.RightArrow)
            button_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            if i < 2:
                button_right_manager = ButtonManager(spinbox)
            else:
                button_right_manager = ButtonManager(spinbox)
            right_load_controls_button_managers_list.append(button_right_manager)
            button_right.pressed.connect(right_load_controls_button_managers_list[i].increment)

            # step buttons

            # left step
            step_left = QToolButton()
            step_left.setText('-')
            step_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            if i < 2:
                step_left_manager = ButtonManager(spinbox)
            else:
                step_left_manager = ButtonManager(spinbox)
            left_load_controls_step_managers_list.append(step_left_manager)
            step_left.pressed.connect(left_load_controls_step_managers_list[i].decrement)

            # right step
            step_right = QToolButton()
            step_right.setText('+')
            step_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            if i < 2:
                step_right_manager = ButtonManager(spinbox)
            else:
                step_right_manager = ButtonManager(spinbox)
            right_load_controls_step_managers_list.append(step_right_manager)
            step_right.pressed.connect(right_load_controls_step_managers_list[i].increment)

            # add buttons to button layout
            buttons.addWidget(step_left)
            buttons.addWidget(button_left)
            buttons.addWidget(button_right)
            buttons.addWidget(step_right)
            # add button layout to right_controls
            right_controls.addLayout(info_display)
            right_controls.addLayout(buttons)

######    #####    #####   #######  #######  #######   #####   #     #
#     #  #     #  #     #     #        #        #     #     #  ##    #
#     #  #     #  #           #        #        #     #     #  # #   #
######   #     #   #####      #        #        #     #     #  #  #  #
#        #     #        #     #        #        #     #     #  #   # #
#        #     #  #     #     #        #        #     #     #  #    ##
#         #####    #####   #######     #     #######   #####   #     #

 #####    #####   #     #  #######  ######    #####   #         #####
#     #  #     #  ##    #     #     #     #  #     #  #        #     #
#        #     #  # #   #     #     #     #  #     #  #        #
#        #     #  #  #  #     #     ######   #     #  #         #####
#        #     #  #   # #     #     #   #    #     #  #              #
#     #  #     #  #    ##     #     #    #   #     #  #        #     #
 #####    #####   #     #     #     #     #   #####   #######   #####

        # CONFIGURE POSITION CONTROLS
        position_controls_label = QLabel()
        position_controls_label.setText('Position')
        position_controls_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        right_controls.addItem(QSpacerItem(1, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum))
        right_controls.addWidget(position_controls_label)

        self.position_controls_list = []
        position_controls_labels = [
            'X',
            'Y',
            'Theta'
        ]
        for i in range(len(position_controls_labels)):

            # info display
            info_display = QHBoxLayout()

            # text
            text = QLabel()
            text.setText(position_controls_labels[i])
            text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
            text.setFixedWidth(40)

            # spinbox
            spinbox = CustomDoubleSpinBox()
            spinbox.setReadOnly(True)
            if i < 2:
                spinbox.setMaximum(1000000 * self.trans_res)
                spinbox.setMinimum(-1000000 * self.trans_res)
                spinbox.setSingleStep(self.trans_res)
                spinbox.setDecimals(self.trans_disp_res)
                spinbox.setSuffix(' m')
            else:
                spinbox.setMaximum(36000)
                spinbox.setMinimum(-36000)
                spinbox.setWrapping(False)
                spinbox.setSingleStep(self.ang_res)
                spinbox.setDecimals(self.ang_disp_res)
                spinbox.setSuffix(' deg')
            spinbox.setKeyboardTracking(False)
            self.position_controls_list.append(spinbox)

            info_display.addWidget(text)
            info_display.addWidget(spinbox)

            # add info display layout to right_controls
            right_controls.addLayout(info_display)

#     #  #######  ######            #         #####      #     ######
##   ##     #     #     #           #        #     #    # #    #     #
# # # #     #     #     #           #        #     #   #   #   #     #
#  #  #     #     #     #           #        #     #  #######  #     #
#     #     #     #     #           #        #     #  #     #  #     #
#     #     #     #     #           #        #     #  #     #  #     #
#     #  #######  ######            #######   #####   #     #  ######

 #####    #####   #     #  #######  ######    #####   #         #####
#     #  #     #  ##    #     #     #     #  #     #  #        #     #
#        #     #  # #   #     #     #     #  #     #  #        #
#        #     #  #  #  #     #     ######   #     #  #         #####
#        #     #  #   # #     #     #   #    #     #  #              #
#     #  #     #  #    ##     #     #    #   #     #  #        #     #
 #####    #####   #     #     #     #     #   #####   #######   #####

        # CONFIGURE MID-LOAD CONTROLS
        mid_load_controls_label = QLabel()
        mid_load_controls_label.setText('Intermediate Loads')
        mid_load_controls_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        right_controls.addItem(QSpacerItem(1, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum))
        right_controls.addWidget(mid_load_controls_label)

        self.junction_selector = QComboBox()
        junction_labels = []
        for i in range(1, self.array.series):
            junction_labels.append(f'Junction {i}')
        self.junction_selector.addItems(junction_labels)
        self.junction_selector.currentIndexChanged.connect(self.updateJunctionSelection)
        self.junction_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        right_controls.addWidget(self.junction_selector)

        self.mlc_list = []
        left_mlc_button_managers_list = []
        right_mlc_button_managers_list = []
        left_mlc_step_managers_list = []
        right_mlc_step_managers_list = []
        mlc_labels = [
            'Fx',
            'Fy',
            'T'
        ]
        for i in range(len(mlc_labels)):

            # info display
            info_display = QHBoxLayout()

            # text
            text = QLabel()
            text.setText(mlc_labels[i])
            text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
            text.setFixedWidth(40)

            # spinbox
            spinbox = CustomDoubleSpinBox()
            if i < 2:
                spinbox.setMaximum(1000000 * self.force_res)
                spinbox.setMinimum(-1000000 * self.force_res)
                spinbox.setSingleStep(self.force_res)
                spinbox.setDecimals(self.force_disp_res)
                spinbox.setSuffix(' N')
            else:
                spinbox.setMaximum(1000000 * self.moment_res)
                spinbox.setMinimum(-1000000 * self.moment_res)
                spinbox.setWrapping(False)
                spinbox.setSingleStep(self.moment_res)
                spinbox.setDecimals(self.moment_disp_res)
                spinbox.setSuffix(' N-m')
            spinbox.valueChanged.connect(self.updateMidLoads)
            spinbox.setKeyboardTracking(False)
            self.mlc_list.append(spinbox)

            info_display.addWidget(text)
            info_display.addWidget(spinbox)

            # buttons
            buttons = QHBoxLayout()

            button_left = QToolButton()
            button_left.setAutoRepeat(True)
            button_left.setAutoRepeatInterval(autoRepeatInterval)
            button_left.setAutoRepeatDelay(autoRepeatInterval)
            button_left.setArrowType(Qt.ArrowType.LeftArrow)
            button_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            if i < 2:
                button_left_manager = ButtonManager(spinbox)
            else:
                button_left_manager = ButtonManager(spinbox)
            left_mlc_button_managers_list.append(button_left_manager)
            button_left.pressed.connect(left_mlc_button_managers_list[i].decrement)

            button_right = QToolButton()
            button_right.setAutoRepeat(True)
            button_right.setAutoRepeatInterval(autoRepeatInterval)
            button_right.setAutoRepeatDelay(autoRepeatInterval)
            button_right.setArrowType(Qt.ArrowType.RightArrow)
            button_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            if i < 2:
                button_right_manager = ButtonManager(spinbox)
            else:
                button_right_manager = ButtonManager(spinbox)
            right_mlc_button_managers_list.append(button_right_manager)
            button_right.pressed.connect(right_mlc_button_managers_list[i].increment)

            # step buttons

            # left step
            step_left = QToolButton()
            step_left.setText('-')
            step_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            if i < 2:
                step_left_manager = ButtonManager(spinbox)
            else:
                step_left_manager = ButtonManager(spinbox)
            left_mlc_step_managers_list.append(step_left_manager)
            step_left.pressed.connect(left_mlc_step_managers_list[i].decrement)

            # right step
            step_right = QToolButton()
            step_right.setText('+')
            step_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            if i < 2:
                step_right_manager = ButtonManager(spinbox)
            else:
                step_right_manager = ButtonManager(spinbox)
            right_mlc_step_managers_list.append(step_right_manager)
            step_right.pressed.connect(right_mlc_step_managers_list[i].increment)

            # add buttons to button layout
            buttons.addWidget(step_left)
            buttons.addWidget(button_left)
            buttons.addWidget(button_right)
            buttons.addWidget(step_right)
            # add button layout to right_controls
            right_controls.addLayout(info_display)
            right_controls.addLayout(buttons)

#######   #####    #####    #####   #        #######  
   #     #     #  #     #  #     #  #        #        
   #     #     #  #        #        #        #        
   #     #     #  #        #        #        ####     
   #     #     #  #   ###  #   ###  #        #        
   #     #     #  #     #  #     #  #        #        
   #      #####    #####    #####   #######  #######  

 #####   #     #  #######  #######   #####   #     #  
#     #  #     #     #        #     #     #  #     #  
#        #     #     #        #     #        #     #  
 #####   #  #  #     #        #     #        #######  
      #  # # # #     #        #     #        #     #  
#     #  ##   ##     #        #     #     #  #     #  
 #####   #     #  #######     #      #####   #     #  

        # CONFIGURE "STRESS/DEFLECTION" TOGGLE SWITCH
        self.visualization_toggle = QPushButton()
        self.visualization_toggle.setText('Torsion Bar Visualization')
        self.visualization_toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.visualization_toggle.pressed.connect(self.toggleVisualization)
        left_controls.addWidget(self.visualization_toggle)

#######  #     #  #######   #####   
   #     ##    #  #        #     #  
   #     # #   #  #        #     #  
   #     #  #  #  ####     #     #  
   #     #   # #  #        #     #  
   #     #    ##  #        #     #  
#######  #     #  #         #####   

######   #######   #####   ######   #           #     #     #   #####   
#     #     #     #     #  #     #  #          # #     #   #   #     #  
#     #     #     #        #     #  #         #   #     # #    #        
#     #     #      #####   ######   #        #######     #      #####   
#     #     #           #  #        #        #     #     #           #  
#     #     #     #     #  #        #        #     #     #     #     #  
######   #######   #####   #        #######  #     #     #      #####   

        # CREATE REACTION DISPLAYS
        reaction_display_title = QLabel()
        reaction_display_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        reaction_display_title.setText('Reaction Forces/Moments:')
        self.Rx_display = QLabel()
        self.Rx_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.Ry_display = QLabel()
        self.Ry_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.M_display = QLabel()
        self.M_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        left_controls.addWidget(reaction_display_title)
        left_controls.addWidget(self.Rx_display)
        left_controls.addWidget(self.Ry_display)
        left_controls.addWidget(self.M_display)

        # CREATE STRESS INFO DISPLAY
        stress_info_display_width = 280
        max_stress_display_title = QLabel()
        max_stress_display_title.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        max_stress_display_title.setFixedWidth(stress_info_display_width)
        max_stress_display_title.setText('\nMaximum Stress Data:')

        max_stress_display_categories = QHBoxLayout()

        stress_type_label = QLabel()
        stress_type_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        stress_type_label.setFixedWidth(int(stress_info_display_width / 4))
        stress_type_label.setText('Type')
        max_stress_display_categories.addWidget(stress_type_label)

        torsion_bar_index_label = QLabel()
        torsion_bar_index_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        torsion_bar_index_label.setFixedWidth(int(2 * stress_info_display_width / 10))
        torsion_bar_index_label.setText('Index')
        max_stress_display_categories.addWidget(torsion_bar_index_label)

        stress_value_label = QLabel()
        stress_value_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        stress_value_label.setFixedWidth(int(3 * stress_info_display_width / 10))
        stress_value_label.setText('Value')
        max_stress_display_categories.addWidget(stress_value_label)

        safety_factor_label = QLabel()
        safety_factor_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        safety_factor_label.setFixedWidth(int(stress_info_display_width / 4))
        safety_factor_label.setText('Safety Factor')
        max_stress_display_categories.addWidget(safety_factor_label)

        max_bending_display = QHBoxLayout()

        bending_label = QLabel()
        bending_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        bending_label.setFixedWidth(int(stress_info_display_width / 4))
        bending_label.setText('Bending')
        max_bending_display.addWidget(bending_label)

        self.bending_index = QLabel()
        self.bending_index.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.bending_index.setFixedWidth(int(2 * stress_info_display_width / 10))
        max_bending_display.addWidget(self.bending_index)

        self.bending_value = QLabel()
        self.bending_value.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.bending_value.setFixedWidth(int(3 * stress_info_display_width / 10))
        max_bending_display.addWidget(self.bending_value)

        self.bending_safety_factor = QLabel()
        self.bending_safety_factor.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.bending_safety_factor.setFixedWidth(int(stress_info_display_width / 4))
        max_bending_display.addWidget(self.bending_safety_factor)

        max_torsion_display = QHBoxLayout()

        torsion_label = QLabel()
        torsion_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        torsion_label.setFixedWidth(int(stress_info_display_width / 4))
        torsion_label.setText('Torsion')
        max_torsion_display.addWidget(torsion_label)

        self.torsion_index = QLabel()
        self.torsion_index.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.torsion_index.setFixedWidth(int(2 * stress_info_display_width / 10))
        max_torsion_display.addWidget(self.torsion_index)

        self.torsion_value = QLabel()
        self.torsion_value.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.torsion_value.setFixedWidth(int(3 * stress_info_display_width / 10))
        max_torsion_display.addWidget(self.torsion_value)

        self.torsion_safety_factor = QLabel()
        self.torsion_safety_factor.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.torsion_safety_factor.setFixedWidth(int(stress_info_display_width / 4))
        max_torsion_display.addWidget(self.torsion_safety_factor)

        max_von_mises_display = QHBoxLayout()

        von_mises_label = QLabel()
        von_mises_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        von_mises_label.setFixedWidth(int(stress_info_display_width / 4))
        von_mises_label.setText('Von Mises')
        max_von_mises_display.addWidget(von_mises_label)

        self.von_mises_index = QLabel()
        self.von_mises_index.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.von_mises_index.setFixedWidth(int(2 * stress_info_display_width / 10))
        max_von_mises_display.addWidget(self.von_mises_index)

        self.von_mises_value = QLabel()
        self.von_mises_value.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.von_mises_value.setFixedWidth(int(3 * stress_info_display_width / 10))
        max_von_mises_display.addWidget(self.von_mises_value)

        self.von_mises_safety_factor = QLabel()
        self.von_mises_safety_factor.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.von_mises_safety_factor.setFixedWidth(int(stress_info_display_width / 4))
        max_von_mises_display.addWidget(self.von_mises_safety_factor)

        left_controls.addWidget(max_stress_display_title)
        left_controls.addLayout(max_stress_display_categories)
        left_controls.addLayout(max_bending_display)
        left_controls.addLayout(max_torsion_display)
        left_controls.addLayout(max_von_mises_display)

######   ######    #####   #######  #######  #        #######   #####   
#     #  #     #  #     #  #           #     #        #        #     #  
#     #  #     #  #     #  #           #     #        #        #        
######   ######   #     #  ####        #     #        ####      #####   
#        #   #    #     #  #           #     #        #              #  
#        #    #   #     #  #           #     #        #        #     #  
#        #     #   #####   #        #######  #######  #######   #####   

        # CONFIGURE "PROFILE EDITOR" SECTION
        heading = QLabel()
        heading.setText('\nProfiles')
        heading.setAlignment(Qt.AlignmentFlag.AlignLeft)
        heading.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # add layout for obstacle index and add/remove buttons
        profile_top_controls = QHBoxLayout()
         # add spinbox for obstacle index
        index_text = QLabel()
        index_text.setText('ID:')
        index_text.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.index_box = QSpinBox()
        self.index_box.setMaximum(0)
        self.index_box.setMinimum(0)
        # self.index_box.setWrapping(True)
        self.index_box.setSingleStep(1)
        self.index_box.valueChanged.connect(self.updateCoordinateFields)
        self.index_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        # add push button for adding points
        add_point_button = QPushButton()
        add_point_button.setText("Add")
        add_point_button.pressed.connect(self.addProfilePoint)
        add_point_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        # add push button for removing points
        remove_point_button = QPushButton()
        remove_point_button.setText("Remove")
        remove_point_button.pressed.connect(self.removeProfilePoint)
        remove_point_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        # add everything to profile_top_controls
        profile_top_controls.addWidget(index_text)
        profile_top_controls.addWidget(self.index_box)
        profile_top_controls.addWidget(add_point_button)
        profile_top_controls.addWidget(remove_point_button)

        # add layout for point x position
        point_x_controls = QHBoxLayout()
        # X
        x_text = QLabel()
        x_text.setText('X:')
        x_text.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.x_position = QDoubleSpinBox()
        self.x_position.setMaximum(1000000 * self.trans_res)
        self.x_position.setMinimum(-1000000 * self.trans_res)
        self.x_position.setWrapping(False)
        self.x_position.setSingleStep(self.trans_res)
        self.x_position.setDecimals(getDecimals(self.trans_res))
        self.x_position.setSuffix(' m')
        self.x_position.setKeyboardTracking(False)
        self.x_position.valueChanged.connect(self.updateProfilePointsX)
        self.x_position.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        point_x_controls.addWidget(x_text)
        point_x_controls.addWidget(self.x_position)
        # rapid buttons
        x_button_left = QToolButton()
        x_button_left.setAutoRepeat(True)
        x_button_left.setAutoRepeatInterval(autoRepeatInterval)
        x_button_left.setAutoRepeatDelay(autoRepeatInterval)
        x_button_left.setArrowType(Qt.ArrowType.LeftArrow)
        x_button_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        x_button_left_manager = ButtonManager(self.x_position, False)
        x_button_left.pressed.connect(x_button_left_manager.decrement)
        x_button_right = QToolButton()
        x_button_right.setAutoRepeat(True)
        x_button_right.setAutoRepeatInterval(autoRepeatInterval)
        x_button_right.setAutoRepeatDelay(autoRepeatInterval)
        x_button_right.setArrowType(Qt.ArrowType.RightArrow)
        x_button_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        x_button_right_manager = ButtonManager(self.x_position, False)
        x_button_right.pressed.connect(x_button_right_manager.increment)
        # add everything to obstacle_x_controls
        point_x_controls.addWidget(x_button_left)
        point_x_controls.addWidget(x_button_right)
        # add layout for point y position
        point_y_controls = QHBoxLayout()
        # Y
        y_text = QLabel()
        y_text.setText('Y:')
        y_text.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.y_position = QDoubleSpinBox()
        self.y_position.setMaximum(1000000 * self.trans_res)
        self.y_position.setMinimum(-1000000 * self.trans_res)
        self.y_position.setWrapping(False)
        self.y_position.setSingleStep(self.trans_res)
        self.y_position.setDecimals(getDecimals(self.trans_res))
        self.y_position.setSuffix(' m')
        self.y_position.setKeyboardTracking(False)
        self.y_position.valueChanged.connect(self.updateProfilePointsY)
        self.y_position.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        point_y_controls.addWidget(y_text)
        point_y_controls.addWidget(self.y_position)
        # rapid buttons
        y_button_left = QToolButton()
        y_button_left.setAutoRepeat(True)
        y_button_left.setAutoRepeatInterval(autoRepeatInterval)
        y_button_left.setAutoRepeatDelay(autoRepeatInterval)
        y_button_left.setArrowType(Qt.ArrowType.LeftArrow)
        y_button_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        y_button_left_manager = ButtonManager(self.y_position, False)
        y_button_left.pressed.connect(y_button_left_manager.decrement)
        y_button_right = QToolButton()
        y_button_right.setAutoRepeat(True)
        y_button_right.setAutoRepeatInterval(autoRepeatInterval)
        y_button_right.setAutoRepeatDelay(autoRepeatInterval)
        y_button_right.setArrowType(Qt.ArrowType.RightArrow)
        y_button_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        y_button_right_manager = ButtonManager(self.y_position, False)
        y_button_right.pressed.connect(y_button_right_manager.increment)
        point_y_controls.addWidget(y_button_left)
        point_y_controls.addWidget(y_button_right)

        # add "remove all points" button
        remove_all_button = QPushButton()
        remove_all_button.setText("Remove All Points")
        remove_all_button.pressed.connect(self.removeAllProfilePoints)
        remove_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # add everything to left_controls
        left_controls.addWidget(heading)
        left_controls.addLayout(profile_top_controls)
        left_controls.addLayout(point_x_controls)
        left_controls.addLayout(point_y_controls)
        left_controls.addWidget(remove_all_button)
        left_controls.addStretch(1)

 #####   #######  #     #  #######  ######   
#     #     #     #     #  #        #     #  
#     #     #     #     #  #        #     #  
#     #     #     #######  ####     ######   
#     #     #     #     #  #        #   #    
#     #     #     #     #  #        #    #   
 #####      #     #     #  #######  #     #  

        # CONFIGURE "PRINT CONFIGURATION" BUTTON
        printer_button = QPushButton()
        printer_button.setText("Print Current Configuration")
        printer_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        printer_button.pressed.connect(self.printConfig)
        bottom_controls.addWidget(printer_button)

        # CONFIGURE "RESET" BUTTON
        reset_button = QPushButton()
        reset_button.setText("Reset LET Array")
        reset_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        reset_button.pressed.connect(self.resetPressed)
        bottom_controls.addWidget(reset_button)

        left_controls.addStretch(1)
        right_controls.addStretch(1)

        # CONFIGURE GRAPHICS LAYOUT
        self.graphics = QStackedLayout()
        self.graphics.addWidget(array_visualizer_container)
        self.graphics.addWidget(torsion_bar_visualizer_container)
        self.graphics.setCurrentIndex(0)

        # COMBINE GRAPHICS LAYOUT AND CONTROLS
        controls.addLayout(top_controls)
        controls.addLayout(bottom_controls)
        main_layout.addLayout(self.graphics, stretch=2)
        main_layout.addLayout(controls, stretch=1)

        self.updateDataFields()
        self.updateStatics()

        # WRAPPING UP
        w = QWidget()
        w.setLayout(main_layout)
        window.setCentralWidget(w)
        window.show()
        window.raise_()

        app.processEvents()
        app.exec()

    def toggleVisualization(self):

        if self.visualization_toggle.text() == 'LET Array Visualization':
            self.visualization_toggle.setText('Torsion Bar Visualization')
            self.array_visualizer.updatePlot()
            self.graphics.setCurrentIndex(0)
        elif self.visualization_toggle.text() == 'Torsion Bar Visualization':
            self.visualization_toggle.setText('LET Array Visualization')
            self.torsion_bar_visualizer.updatePlot()
            self.graphics.setCurrentIndex(1)

    def updateJunctionSelection(self):

        junction_index = self.junction_selector.currentIndex() + 1

        Fx_mid = self.array.Fx_mid[junction_index]
        Fy_mid = self.array.Fy_mid[junction_index]
        T_mid = self.array.T_mid[junction_index]

        self.mlc_list[0].blockSignals(True)
        self.mlc_list[0].setValue(Fx_mid)
        self.mlc_list[0].blockSignals(False)

        self.mlc_list[1].blockSignals(True)
        self.mlc_list[1].setValue(Fy_mid)
        self.mlc_list[1].blockSignals(False)

        self.mlc_list[2].blockSignals(True)
        self.mlc_list[2].setValue(T_mid)
        self.mlc_list[2].blockSignals(False)

    def updateMidLoads(self):

        junction_index = self.junction_selector.currentIndex() + 1

        self.array.Fx_mid[junction_index] = self.mlc_list[0].value()
        self.array.Fy_mid[junction_index] = self.mlc_list[1].value()
        self.array.T_mid[junction_index] = self.mlc_list[2].value()

        self.updateStatics()

    def updateDataFields(self):

        Rx, Ry, M = self.array.getReactions()

        self.Rx_display.setText(f'Rx: {Rx}')
        self.Ry_display.setText(f'Ry: {Ry}')
        self.M_display.setText(f'M: {M}')

        bending_index, max_bending = self.array.getMaxBending()
        if max_bending != 0:
            bending_safety_factor = str(np.round(self.array.Sy / max_bending, 2))
        else:
            bending_safety_factor = 'Inf'
        torsion_index, max_torsion = self.array.getMaxTorsion()
        if max_torsion != 0:
            torsion_safety_factor = str(np.round(self.array.Sy / (np.sqrt(3) * max_torsion), 2))
        else:
            torsion_safety_factor = 'Inf'
        von_mises_index, max_von_mises = self.array.getMaxVonMises()
        if max_von_mises != 0:
            von_mises_safety_factor = str(np.round(self.array.Sy / max_von_mises, 2))
        else:
            von_mises_safety_factor = 'Inf'

        self.bending_index.setText(str(bending_index + 1))
        self.bending_value.setText(str(int(np.rint(max_bending))))
        self.bending_safety_factor.setText(bending_safety_factor)

        self.torsion_index.setText(str(torsion_index + 1))
        self.torsion_value.setText(str(int(np.rint(max_torsion))))
        self.torsion_safety_factor.setText(torsion_safety_factor)

        self.von_mises_index.setText(str(von_mises_index + 1))
        self.von_mises_value.setText(str(int(np.rint(max_von_mises))))
        self.von_mises_safety_factor.setText(von_mises_safety_factor)

    def updateStatics(self):

        Rx, Ry, M = self.array.getReactions()

        Fx = self.load_controls_list[0].value()
        Fy = self.load_controls_list[1].value()
        T = self.load_controls_list[2].value()

        self.array.calculateStaticsInverse(Fx, Fy, T, guess=M)

        x, y = self.array.getEndPosition()
        theta = self.array.getEndRotation()

        self.position_controls_list[0].blockSignals(True)
        self.position_controls_list[0].setValue(x)
        self.position_controls_list[0].blockSignals(False)

        self.position_controls_list[1].blockSignals(True)
        self.position_controls_list[1].setValue(y)
        self.position_controls_list[1].blockSignals(False)

        self.position_controls_list[2].blockSignals(True)
        self.position_controls_list[2].setValue(np.degrees(theta))
        self.position_controls_list[2].blockSignals(False)

        current_visualization = self.graphics.currentIndex()
        if current_visualization == 0:
            self.array_visualizer.updatePlot()
        elif current_visualization == 1:
            self.torsion_bar_visualizer.updatePlot()
        self.updateDataFields()

    def printConfig(self):

        Rx, Ry, M = self.array.getReactions()
        Fx, Fy, T = self.array.getEndLoad()
        x, y = self.array.getEndPosition()
        theta = self.array.getEndRotation()

        print('LET Array Configuration ================')
        print(f'Rx = {Rx} N')
        print(f'Ry = {Ry} N')
        print(f'M = {M} N-m')
        print(f'Fx = {Fx} N')
        print(f'Fy = {Fy} N')
        print(f'T = {T} N-m')
        print(f'x = {x} m')
        print(f'y = {y} m')
        print(f'theta = {np.degrees(theta)} deg\n')

    def resetPressed(self):

        '''
        resets all deflections to zero
        '''

        for i, spinbox in enumerate(self.load_controls_list):
            spinbox.blockSignals(True)
            spinbox.setValue(0)
            spinbox.blockSignals(False)

        self.array.Fx_mid[:] = 0
        self.array.Fy_mid[:] = 0
        self.array.T_mid[:] = 0
        for i, spinbox in enumerate(self.mlc_list):
            spinbox.blockSignals(True)
            spinbox.setValue(0)
            spinbox.blockSignals(False)

        self.load_controls_list[0].setSingleStep(self.force_res)
        self.load_controls_list[0].setDecimals(getDecimals(self.force_res))
        self.load_controls_list[1].setSingleStep(self.force_res)
        self.load_controls_list[1].setDecimals(getDecimals(self.force_res))
        self.load_controls_list[2].setSingleStep(self.moment_res)
        self.load_controls_list[2].setDecimals(getDecimals(self.moment_res))

        self.position_controls_list[0].setSingleStep(self.trans_res)
        self.position_controls_list[0].setDecimals(getDecimals(self.trans_res))
        self.position_controls_list[1].setSingleStep(self.trans_res)
        self.position_controls_list[1].setDecimals(getDecimals(self.trans_res))
        self.position_controls_list[2].setSingleStep(self.ang_res)
        self.position_controls_list[2].setDecimals(getDecimals(self.ang_res))

        self.mlc_list[0].setSingleStep(self.force_res)
        self.mlc_list[0].setDecimals(getDecimals(self.force_res))
        self.mlc_list[1].setSingleStep(self.force_res)
        self.mlc_list[1].setDecimals(getDecimals(self.force_res))
        self.mlc_list[2].setSingleStep(self.moment_res)
        self.mlc_list[2].setDecimals(getDecimals(self.moment_res))

        self.updateStatics()

    def addProfilePoint(self):

        current_index = self.index_box.value()

        # adding first point
        if len(self.profile_points_x) == 0:
            self.profile_points_x.append(0)
            self.profile_points_y.append(0)
            self.index_box.setMaximum(len(self.profile_points_x) - 1)
            self.index_box.setValue(0)
        else:
            # adding to end
            if current_index == len(self.profile_points_x) - 1:
                self.profile_points_x.append(self.profile_points_x[-1] + self.array.b)
                self.profile_points_y.append(self.profile_points_y[-1] + self.array.b)
                self.index_box.setMaximum(len(self.profile_points_x) - 1)
                self.index_box.setValue(current_index + 1)
            # adding after current index (not at end)
            else:
                self.profile_points_x.insert(current_index + 1, (self.profile_points_x[current_index] + self.profile_points_x[current_index + 1]) / 2)
                self.profile_points_y.insert(current_index + 1, (self.profile_points_y[current_index] + self.profile_points_y[current_index + 1]) / 2)
                self.index_box.setMaximum(len(self.profile_points_x) - 1)
                self.index_box.setValue(current_index + 1)

        self.updateCoordinateFields()
        self.array_visualizer.updatePlot()

    def removeProfilePoint(self):

        if len(self.profile_points_x) != 0:
            index = self.index_box.value()
            del self.profile_points_x[index]
            del self.profile_points_y[index]

        self.index_box.setMaximum(len(self.profile_points_x) - 1)
        self.updateCoordinateFields()
        self.array_visualizer.updatePlot()

    def removeAllProfilePoints(self):

        self.profile_points_x.clear()
        self.profile_points_y.clear()
        self.index_box.setValue(0)
        self.index_box.setMaximum(0)

        self.updateCoordinateFields()
        self.array_visualizer.updatePlot()

    def updateProfilePointsX(self):

        current_index = self.index_box.value()
        new_value = self.x_position.value()
        self.profile_points_x[current_index] = new_value

        self.updateCoordinateFields()
        self.array_visualizer.updatePlot()

    def updateProfilePointsY(self):

        current_index = self.index_box.value()
        new_value = self.y_position.value()
        self.profile_points_y[current_index] = new_value

        self.updateCoordinateFields()
        self.array_visualizer.updatePlot()

    def updateCoordinateFields(self):

        current_index = self.index_box.value()
        if len(self.profile_points_x) != 0:
            self.x_position.setValue(self.profile_points_x[current_index])
            self.y_position.setValue(self.profile_points_y[current_index])
        else:
            self.x_position.blockSignals(True)
            self.x_position.setValue(0)
            self.x_position.blockSignals(False)
            self.y_position.blockSignals(True)
            self.y_position.setValue(0)
            self.y_position.blockSignals(False)

class ButtonManager:
    def __init__(self, widget, use_degrees=False):
        self.widget = widget
        self.use_degrees = use_degrees

    def decrement(self):
        inc = self.widget.singleStep()
        new_value = roundToResolution(self.widget.value() - inc, inc)
        self.widget.setValue(new_value)

    def increment(self):
        inc = self.widget.singleStep()
        new_value = roundToResolution(self.widget.value() + inc, inc)
        self.widget.setValue(new_value)

class CustomDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super(CustomDoubleSpinBox, self).__init__(*args, **kwargs)

        self.customContextMenu = QMenu(self)
        step_change_request = self.customContextMenu.addAction('Change Step Size')
        step_change_request.triggered.connect(self.changeStep)

    def contextMenuEvent(self, event):
        self.customContextMenu.exec(event.globalPos())

    def changeStep(self):

        current_step_size = self.singleStep()

        step_size_dialog = QInputDialog(self)
        step_size_dialog.setWindowTitle('Step Size Dialog')
        step_size_dialog.setLabelText('Enter a New Step Size:')
        step_size_dialog.setDoubleDecimals(12)
        step_size_dialog.setDoubleMaximum(1e12)
        step_size_dialog.setDoubleMinimum(-1e12)
        step_size_dialog.setDoubleValue(current_step_size)
        step_size_dialog.setDoubleStep(current_step_size)
        step_size_dialog.setOkButtonText('Enter')
        step_size_dialog.setCancelButtonText('Cancel')
        step_size_dialog.setInputMode(QInputDialog.InputMode.DoubleInput)
        okay_clicked = step_size_dialog.exec()

        if okay_clicked:
            new_step_size = step_size_dialog.doubleValue()
            self.setSingleStep(new_step_size)
            self.setDecimals(getDecimals(new_step_size))