import os
import sys
import cv2

import numpy as np
import qdarkstyle
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QSpacerItem, QSizePolicy)
from PyQt5.uic.properties import QtCore
from matplotlib import pyplot as plt
from qtpy.uic import loadUi

from canvas.canvas import Canvas
from networkx.network import forward_prop, get_predictions


def debug_trace():
    from pdb import set_trace
    QtCore.pyqtRemoveInputHook()
    set_trace()
    # QtCore.pyqtRestoreInputHook()


class DigitRecognizer(QMainWindow):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        super(DigitRecognizer, self).__init__()
        ui_path = os.path.join(self.ROOT_DIR, 'design/design_clone.ui')
        loadUi(ui_path, self)
        self.setFixedSize(800, 600)

        # Load network data
        self.W1 = np.fromfile(os.path.join(self.ROOT_DIR, 'network_Data/W1.dat')).reshape(16, 784)
        self.W2 = np.fromfile(os.path.join(self.ROOT_DIR, 'network_Data/W2.dat')).reshape(10, 16)
        self.b1 = np.fromfile(os.path.join(self.ROOT_DIR, 'network_Data/b1.dat')).reshape(16, 1)
        self.b2 = np.fromfile(os.path.join(self.ROOT_DIR, 'network_Data/b2.dat')).reshape(10, 1)
        print("Status: Success load network weights!")

        # Create an instance of the Canvas widget
        self.__canvas = Canvas()

        # Create a layout to hold both Canvas and the loaded UI content
        main_layout = QVBoxLayout()
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        main_layout.addWidget(self.__canvas, alignment=Qt.AlignCenter)
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        print("Status: Success load canvas!")

        # Create a central widget and set the main layout
        self.centralwidget.setLayout(main_layout)
        self.setCentralWidget(self.centralwidget)

        # Add buttons events
        self.guessButton.clicked.connect(self.__generate_matrix)
        self.resetButton.clicked.connect(self.__reset_canvas)
        print("Status: Success load buttons events!")

    def __reset_canvas(self):
        self.__canvas.clearCanvas()

    def __pixmap_to_bytes(self, pixmap):
        # Fetch pixel values from the Canvas and construct a matrix
        canvas_width, canvas_height, canvas_channels = self.__canvas.width(), self.__canvas.height(), 4

        image = pixmap.toImage()
        b = image.bits()
        b.setsize(canvas_height * canvas_width * canvas_channels)
        return b

    def __zero_border(self, matrix):
        # Setting the first and last rows to 0
        matrix[0, :] = 0
        matrix[-1, :] = 0

        # Setting the first and last columns to 0
        matrix[:, 0] = 0
        matrix[:, -1] = 0

        # Create a 28x28 matrix filled with zeros
        bordered_matrix = np.zeros((28, 28))

        # Calculate the position to place the original matrix
        start_row = (bordered_matrix.shape[0] - matrix.shape[0]) // 2
        start_col = (bordered_matrix.shape[1] - matrix.shape[1]) // 2

        # Place the original matrix at the center of the bordered matrix
        bordered_matrix[start_row:start_row + matrix.shape[0], start_col:start_col + matrix.shape[1]] = matrix

        return bordered_matrix

    def __generate_matrix(self):
        # Fetch pixel values from the Canvas and construct a matrix
        canvas_width, canvas_height, canvas_channels = self.__canvas.width(), self.__canvas.height(), 4

        b = self.__pixmap_to_bytes(self.__canvas.grab())
        arr = np.frombuffer(b, np.uint8).reshape((canvas_height, canvas_width, canvas_channels))
        gray_arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        resized_gray_arr = 255 - cv2.resize(gray_arr, (20, 20), interpolation=cv2.INTER_AREA)
        resized_gray_arr = self.__zero_border(resized_gray_arr)
        resized_gray_arr = (resized_gray_arr - np.mean(resized_gray_arr)) / np.std(resized_gray_arr)

        plt.gray()
        plt.imshow(resized_gray_arr, interpolation='nearest')
        plt.show()

        resized_gray_arr = resized_gray_arr.reshape(-1, 1)

        # Give prediction
        _, _, _, A2 = forward_prop(self.W1, self.b1, self.W2, self.b2, resized_gray_arr)
        pred = get_predictions(A2)
        self.guessLabel.setText(f"Number recognized is: {pred}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setStyleSheet(stylesheet)
    window = DigitRecognizer()
    window.show()
    sys.exit(app.exec_())
