from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.lastPoint = None
        self.lines = None
        self.initUI()

    def initUI(self):
        self.setGeometry(120, 200, 321, 321)
        self.setFixedSize(321, 321)
        self.setWindowTitle('Drawing on Canvas')

        self.lines = []
        self.lastPoint = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        painter.setPen(QPen(Qt.black, 30, Qt.SolidLine))

        for line in self.lines:
            if len(line) > 1:
                startPoint = line[0]
                for nextPoint in line[1:]:
                    painter.drawLine(startPoint, nextPoint)
                    startPoint = nextPoint

        # Draw border
        painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        painter.drawRect(self.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.lines.append([self.lastPoint])

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.lastPoint:
            newPoint = event.pos()
            self.lines[-1].append(newPoint)
            self.lastPoint = newPoint
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.lastPoint:
            self.lastPoint = None

    def clearCanvas(self):
        self.lines = []  # Clear the drawn lines
        self.update()  # Update the canvas to trigger repaint
