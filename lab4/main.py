from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLineEdit, QTableWidgetItem, \
    QHeaderView
import sys
import modeller

from math import sqrt

import numpy as np

QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


def calculate_params(la1, dla1, la2, dla2, mu1, dmu1, mu2, dmu2):
    mT11 = 1 / la1
    dT11 = (1 / (la1 - dla1) - 1 / (la1 + dla1)) / 2

    mT12 = 1 / la2
    dT12 = (1 / (la2 - dla2) - 1 / (la2 + dla2)) / 2

    mT21 = 1 / mu1
    dT21 = (1 / (mu1 - dmu1) - 1 / (mu1 + dmu1)) / 2

    mT22 = 1 / mu2
    dT22 = (1 / (mu2 - dmu2) - 1 / (mu2 + dmu2)) / 2

    return mT11, dT11, mT12, dT12, mT21, dT21, mT22, dT22


def process_matrixes(initialMatrix):
    levelMatrix = [[0.0 for j in range(len(initialMatrix[0]))] for i in range(len(initialMatrix))]

    for i in range(len(levelMatrix)):
        for j in range(len(levelMatrix[0])):
            try:
                levelMatrix[i][j] = float(initialMatrix[i][j])
            except:
                levelMatrix[i][j] = 0.0

    # print(levelMatrix)

    planningMatrix = list(map(lambda row: row[:256 + 8], levelMatrix.copy()[:-1]))
    checkVector = np.array(levelMatrix.copy()[-1][:256 + 8])

    # print(planningMatrix)

    # transposedPlanningMatrix = planningMatrix.transpose()

    # print(transposedPlanningMatrix)

    return planningMatrix, checkVector


def convert_value_to_factor(min, max, value):
    return (value - (max + min) / 2.0) / ((max - min) / 2.0)

def convert_factor_to_value(min, max, factor):
    return factor * ((max - min) / 2.0) + (max + min) / 2.0


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("lab4.ui", self)

        self.la1 = 0
        self.dla1 = 1
        self.la2 = 0
        self.dla2 = 1
        self.mu1 = 0
        self.dmu1 = 1
        self.mu2 = 0
        self.dmu2 = 1
        self.tmax = 300

        self.S = 0
        self.a = 1

        self.read_params()

        self.init_table()

        self.set_free_point()

        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.bTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    @pyqtSlot(name='on_calculateButton_clicked')
    def on_process(self):
        print('process')

        layout = self.ui.externalLayout.itemAt(0)
        print(layout)

        self.read_params()

        la1 = self.la1
        dla1 = self.dla1
        la2 = self.la2
        dla2 = self.dla2
        mu1 = self.mu1
        dmu1 = self.dmu1
        mu2 = self.mu2
        dmu2 = self.dmu2
        tmax = self.tmax

        mT11, dT11, mT12, dT12, mT21, dT21, mT22, dT22 = calculate_params(la1, dla1, la2, dla2, mu1, dmu1, mu2, dmu2)

        model = modeller.Model([mT11, mT12], [dT11, dT12], [mT21, mT22], [dT21, dT22], 2, 1, 0)

        print('start')

        ro = (la1 + la2) / ((mu1 + mu2) / 2)
        avg_queue_size, avg_queue_time, processed_requests = model.time_based_modellingg(tmax, 0.001)

        result = f'Расчетная загрузка системы: {ro}\n' \
                 f'Среднее количество заявок в системе: {avg_queue_size}\n' \
                 f'Среднее время ожидания: {avg_queue_time}\n' \
                 f'Обработано заявок: {processed_requests}'

        QMessageBox.information(self, 'Result', result)

        self.set_free_point()

    @pyqtSlot(name='on_calculateModelButton_clicked')
    def on_calculate_model(self):
        self.calculate_occe()

    def calculate_occe(self):
        layout = self.ui.externalLayout.itemAt(1)
        tableWidget = self.ui.tableWidget

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()
        print(Xmin, Xmax)

        planningTable = [[tableWidget.item(i, j).text() for j in range(cols)] for i in range(rows)]

        planningMatrix, checkVector = process_matrixes(planningTable)

        factorMatrix = np.matrix(list(map(lambda row: row[1:9], planningTable.copy())))

        Y = [0 for i in range(257 + 16 + 1)]

        for i in range(len(factorMatrix.tolist())):
            la1 = convert_factor_to_value(Xmin[0], Xmax[0], float(factorMatrix.item((i, 0))))
            dla1 = convert_factor_to_value(Xmin[1], Xmax[1], float(factorMatrix.item((i, 1))))
            la2 = convert_factor_to_value(Xmin[2], Xmax[2], float(factorMatrix.item((i, 2))))
            dla2 = convert_factor_to_value(Xmin[3], Xmax[3], float(factorMatrix.item((i, 3))))
            mu1 = convert_factor_to_value(Xmin[4], Xmax[4], float(factorMatrix.item((i, 4))))
            dmu1 = convert_factor_to_value(Xmin[5], Xmax[5], float(factorMatrix.item((i, 5))))
            mu2 = convert_factor_to_value(Xmin[6], Xmax[6], float(factorMatrix.item((i, 6))))
            dmu2 = convert_factor_to_value(Xmin[7], Xmax[7], float(factorMatrix.item((i, 7))))

            # print(la, dla, mu, dmu)
            mT11, dT11, mT12, dT12, mT21, dT21, mT22, dT22 = calculate_params(la1, dla1, la2, dla2, mu1, dmu1, mu2, dmu2)

            # print(i, mT11, dT11, mT12, dT12, mT2, dT2)

            model = modeller.Model([mT11, mT12], [dT11, dT12], [mT21, mT22], [dT21, dT22], 2, 1, 0)

            avg_queue_size, avg_queue_time, processed_requests = model.time_based_modellingg(100, 0.001)

            # print(avg_queue_time)
            Y[i] = avg_queue_time
            tableWidget.setItem(i, 256 + 8, QTableWidgetItem(str(round(avg_queue_time, 4))))

        Yt = [Y[-1]]
        Y = np.array(Y[:-1])
        print("calculated occe")

        transPlanningMatrix = np.transpose(planningMatrix.copy())
        B = [(transPlanningMatrix[i] @ Y) / self.calc_b_divider(planningMatrix, i) for i in range(256 + 8)]
        print(B[0])

        self.set_b_table(B, self.ui.bTableWidget, 0)

        B[0] = B[0] + (B[-8] * self.S + B[-7] * self.S + B[-6] * self.S + B[-5] * self.S \
               + B[-4] * self.S + B[-3] * self.S + B[-2] * self.S + B[-1] * self.S)

        # print(B[:5])
        # Yl = np.array(list(map(lambda row: row[:7], planningMatrix.tolist() + [checkVector.tolist()]))) @ np.array(
        #     B[:7])
        Yn = np.array(planningMatrix + [checkVector.tolist()]) @ np.array(B)
        resYList = Y.tolist() + Yt
        for i in range(len(resYList)):
            # tableWidget.setItem(i, 65 + 6, QTableWidgetItem(str(round(Yl.tolist()[i], 4))))
            tableWidget.setItem(i, 257 + 8, QTableWidgetItem(str(round(Yn.tolist()[i], 4))))
            # tableWidget.setItem(i, 67 + 6, QTableWidgetItem(
            #     str(abs(round(round(resYList[i], 6) - round(Yl.tolist()[i], 6), 6)))))
            tableWidget.setItem(i, 258 + 8, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Yn.tolist()[i], 6), 6)))))

    def calc_b_divider(self, matrix, i):
        res = 0

        for j in range(256 + 16 + 1):
            res += (matrix[j][i]) ** 2

        return res

    @pyqtSlot(name="on_zeroLevelButton_clicked")
    def set_zero_level(self):
        layout = self.ui.externalLayout.itemAt(0)

        Xmin, Xmax = self.read_model_params()

        for i in range(layout.rowCount()):
            for j in range(layout.columnCount()):
                if layout.itemAtPosition(i, j):
                    widget = layout.itemAtPosition(i, j).widget()
                    objName = widget.objectName()

                    if not isinstance(widget, QLineEdit):
                        continue

                    if objName == 'arriveIntensity1':
                        widget.setText(str(round((Xmin[0] + Xmax[0]) / 2, 4)))
                        self.la1 = round((Xmin[0] + Xmax[0]) / 2, 4)
                    elif objName == 'arriveIntensity2':
                        widget.setText(str(round((Xmin[2] + Xmax[2]) / 2, 4)))
                        self.la2 = round((Xmin[2] + Xmax[2]) / 2, 4)
                    elif objName == 'processIntensity1':
                        widget.setText(str(round((Xmin[4] + Xmax[4]) / 2, 4)))
                        self.mu1 = round((Xmin[4] + Xmax[4]) / 2, 4)
                    elif objName == 'processIntensity2':
                        widget.setText(str(round((Xmin[6] + Xmax[6]) / 2, 4)))
                        self.mu2 = round((Xmin[6] + Xmax[6]) / 2, 4)
                    elif objName == 'arriveIntensityDispersion1':
                        widget.setText(str(round((Xmin[1] + Xmax[1]) / 2, 4)))
                        self.dla1 = round((Xmin[1] + Xmax[1]) / 2, 4)
                    elif objName == 'arriveIntensityDispersion2':
                        widget.setText(str(round((Xmin[3] + Xmax[3]) / 2, 4)))
                        self.dla2 = round((Xmin[3] + Xmax[3]) / 2, 4)
                    elif objName == 'processIntensityDispersion1':
                        widget.setText(str(round((Xmin[5] + Xmax[5]) / 2, 4)))
                        self.dmu1 = round((Xmin[5] + Xmax[5]) / 2, 4)
                    elif objName == 'processIntensityDispersion2':
                        widget.setText(str(round((Xmin[7] + Xmax[7]) / 2, 4)))
                        self.dmu2 = round((Xmin[7] + Xmax[7]) / 2, 4)

        self.set_free_point()

    def read_params(self):
        layout = self.ui.externalLayout.itemAt(0)
        # print(layout)

        la1 = 0
        dla1 = 1
        la2 = 0
        dla2 = 1
        mu1 = 0
        dmu1 = 1
        mu2 = 0
        dmu2 = 1

        tmax = 300

        for i in range(layout.rowCount()):
            for j in range(layout.columnCount()):
                if layout.itemAtPosition(i, j):
                    widget = layout.itemAtPosition(i, j).widget()
                    objName = widget.objectName()

                    if not isinstance(widget, QLineEdit):
                        continue

                    try:
                        if objName == 'arriveIntensity1':
                            print('arrive 1')
                            la1 = float(widget.text())
                        elif objName == 'arriveIntensity2':
                            print('arrive 2')
                            la2 = float(widget.text())
                        elif objName == 'processIntensity1':
                            print('process 1')
                            mu1 = float(widget.text())
                        elif objName == 'processIntensity2':
                            print('process 2')
                            mu2 = float(widget.text())
                        elif objName == 'arriveIntensityDispersion1':
                            print('arrive disp 1')
                            dla1 = float(widget.text())
                        elif objName == 'arriveIntensityDispersion2':
                            print('arrive disp 2')
                            dla2 = float(widget.text())
                        elif objName == 'processIntensityDispersion1':
                            print('process disp 1')
                            dmu1 = float(widget.text())
                        elif objName == 'processIntensityDispersion2':
                            print('process disp 2')
                            dmu2 = float(widget.text())
                        elif objName == 'modellingTime':
                            print('time')
                            tmax = float(widget.text())
                    except ValueError:
                        QMessageBox.warning(self, 'Error', 'Ошибка ввода')
                        return

        if (la1 <= 0 or dla1 >= la1) or (la2 <= 0 or dla2 >= la2) or (mu1 <= 0 or dmu1 >= mu1) or (mu2 <= 0 or dmu2 >= mu2):
            print(la1, dla1, la2, dla2, mu1, dmu1, mu2, dmu2)
            QMessageBox.warning(self, 'Error', 'Интенсивности должны быть больше 0')
            return

        self.la1 = la1
        self.la2 = la2
        self.mu1 = mu1
        self.mu2 = mu2
        self.dla1 = dla1
        self.dla2 = dla2
        self.dmu1 = dmu1
        self.dmu2 = dmu2
        self.tmax = tmax

        # return la, dla, mu, dmu, tmax

    def read_model_params(self):
        layout = self.ui.externalLayout.itemAt(1)

        Xmin = [0, 0, 0, 0, 0, 0, 0, 0]
        Xmax = [0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(layout.rowCount()):
            for j in range(layout.columnCount()):
                if layout.itemAtPosition(i, j):
                    widget = layout.itemAtPosition(i, j).widget()
                    objName = widget.objectName()

                    if not isinstance(widget, QLineEdit):
                        continue

                    try:
                        if objName == 'arriveIntensityMin':
                            Xmin[0] = Xmin[2] = float(widget.text())
                        elif objName == 'arriveIntensityMax':
                            Xmax[0] = Xmax[2] = float(widget.text())
                        elif objName == 'arriveIntensityDispersionMin':
                            Xmin[1] = Xmin[3] = float(widget.text())
                        elif objName == 'arriveIntensityDispersionMax':
                            Xmax[1] = Xmax[3] = float(widget.text())
                        elif objName == 'processIntensityMin':
                            Xmin[4] = Xmin[6] = float(widget.text())
                        elif objName == 'processIntensityMax':
                            Xmax[4] = Xmax[6] = float(widget.text())
                        elif objName == 'processIntensityDispersionMin':
                            Xmin[5] = Xmin[7] = float(widget.text())
                        elif objName == 'processIntensityDispersionMax':
                            Xmax[5] = Xmax[7] = float(widget.text())
                    except ValueError:
                        QMessageBox.warning(self, 'Error', 'Ошибка ввода')
                        return

        return Xmin, Xmax

    def set_free_point(self):
        tableWidget = self.ui.tableWidget

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()
        x1 = convert_value_to_factor(Xmin[0], Xmax[0], self.la1)
        x2 = convert_value_to_factor(Xmin[1], Xmax[1], self.dla1)
        x3 = convert_value_to_factor(Xmin[2], Xmax[2], self.la2)
        x4 = convert_value_to_factor(Xmin[3], Xmax[3], self.dla2)
        x5 = convert_value_to_factor(Xmin[4], Xmax[4], self.mu1)
        x6 = convert_value_to_factor(Xmin[5], Xmax[5], self.dmu1)
        x7 = convert_value_to_factor(Xmin[6], Xmax[6], self.mu2)
        x8 = convert_value_to_factor(Xmin[7], Xmax[7], self.dmu2)
        # print(convert_value_to_factor(Xmin[0], Xmax[0], self.la),
        #       convert_factor_to_value(Xmin[0], Xmax[0], convert_value_to_factor(Xmin[0], Xmax[0], self.la)))

        x = self.get_factor_array(x1, x2, x3, x4, x5, x6, x7, x8, self.S)

        for i in range(256 + 8):
            tableWidget.setItem(256 + 16 + 1, i, QTableWidgetItem(str(round(x[i], 6))))

    def set_b_table(self, B, table, row):
        for i in range(len(B)):
            table.setItem(row, i, QTableWidgetItem(str(round(B[i], 7))))

    def init_table(self):
        table = self.ui.tableWidget

        for i in range(256):
            table.setItem(i, 1, QTableWidgetItem(str(1 if i % 2 == 0 else -1)))
            table.setItem(i, 2, QTableWidgetItem(str(1 if i % 4 <= 1 else -1)))
            table.setItem(i, 3, QTableWidgetItem(str(1 if i % 8 <= 3 else -1)))
            table.setItem(i, 4, QTableWidgetItem(str(1 if i % 16 <= 7 else -1)))
            table.setItem(i, 5, QTableWidgetItem(str(1 if i % 32 <= 15 else -1)))
            table.setItem(i, 6, QTableWidgetItem(str(1 if i % 64 <= 31 else -1)))
            table.setItem(i, 7, QTableWidgetItem(str(1 if i % 128 <= 63 else -1)))
            table.setItem(i, 8, QTableWidgetItem(str(1 if i % 256 <= 127 else -1)))

        for i in range(256 + 8):
            self.ui.bTableWidget.setItem(0, i, QTableWidgetItem('-'))

        N0 = 256
        n0 = 2 * 8
        N = N0 + n0 + 1

        self.S = sqrt(N0 / N)
        self.a = sqrt((self.S * N - N0) / 2)

        print('S, a: ', self.S, self.a)

        for i in range(N0):
            x1 = int(table.item(i, 1).text())
            x2 = int(table.item(i, 2).text())
            x3 = int(table.item(i, 3).text())
            x4 = int(table.item(i, 4).text())
            x5 = int(table.item(i, 5).text())
            x6 = int(table.item(i, 6).text())
            x7 = int(table.item(i, 7).text())
            x8 = int(table.item(i, 8).text())

            x = self.get_factor_array(x1, x2, x3, x4, x5, x6, x7, x8, self.S)

            for k in range(9, N0 + 8):
                table.setItem(i, k, QTableWidgetItem(str(round(x[k], 6))))

        xi = [
            [self.a, 0, 0, 0, 0, 0, 0, 0],
            [-self.a, 0, 0, 0, 0, 0, 0, 0],
            [0, self.a, 0, 0, 0, 0, 0, 0],
            [0, -self.a, 0, 0, 0, 0, 0, 0],
            [0, 0, self.a, 0, 0, 0, 0, 0],
            [0, 0, -self.a, 0, 0, 0, 0, 0],
            [0, 0, 0, self.a, 0, 0, 0, 0],
            [0, 0, 0, -self.a, 0, 0, 0, 0],
            [0, 0, 0, 0, self.a, 0, 0, 0],
            [0, 0, 0, 0, -self.a, 0, 0, 0],
            [0, 0, 0, 0, 0, self.a, 0, 0],
            [0, 0, 0, 0, 0, -self.a, 0, 0],
            [0, 0, 0, 0, 0, 0, self.a, 0],
            [0, 0, 0, 0, 0, 0, -self.a, 0],
            [0, 0, 0, 0, 0, 0, 0, self.a],
            [0, 0, 0, 0, 0, 0, 0, -self.a],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]

        for i in range(n0 + 1):
            x = self.get_factor_array(xi[i][0], xi[i][1], xi[i][2], xi[i][3], xi[i][4], xi[i][5], xi[i][6], xi[i][7], self.S)

            for j in range(8):
                table.setItem(256 + i, j + 1, QTableWidgetItem(str(round(xi[i][j], 6))))

            for k in range(9, N0 + 8):
                table.setItem(256 + i, k, QTableWidgetItem(str(round(x[k], 6))))

    def get_factor_array(self, x1, x2, x3, x4, x5, x6, x7, x8, S):
        return [
            1,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x1 * x5,
            x1 * x6,
            x1 * x7,
            x1 * x8,
            x2 * x3,
            x2 * x4,
            x2 * x5,
            x2 * x6,
            x2 * x7,
            x2 * x8,
            x3 * x4,
            x3 * x5,
            x3 * x6,
            x3 * x7,
            x3 * x8,
            x4 * x5,
            x4 * x6,
            x4 * x7,
            x4 * x8,
            x5 * x6,
            x5 * x7,
            x5 * x8,
            x6 * x7,
            x6 * x8,
            x7 * x8,
            x1 * x2 * x3,
            x1 * x2 * x4,
            x1 * x2 * x5,
            x1 * x2 * x6,
            x1 * x2 * x7,
            x1 * x2 * x8,
            x1 * x3 * x4,
            x1 * x3 * x5,
            x1 * x3 * x6,
            x1 * x3 * x7,
            x1 * x3 * x8,
            x1 * x4 * x5,
            x1 * x4 * x6,
            x1 * x4 * x7,
            x1 * x4 * x8,
            x1 * x5 * x6,
            x1 * x5 * x7,
            x1 * x5 * x8,
            x1 * x6 * x7,
            x1 * x6 * x8,
            x1 * x7 * x8,
            x2 * x3 * x4,
            x2 * x3 * x5,
            x2 * x3 * x6,
            x2 * x3 * x7,
            x2 * x3 * x8,
            x2 * x4 * x5,
            x2 * x4 * x6,
            x2 * x4 * x7,
            x2 * x4 * x8,
            x2 * x5 * x6,
            x2 * x5 * x7,
            x2 * x5 * x8,
            x2 * x6 * x7,
            x2 * x6 * x8,
            x2 * x7 * x8,
            x3 * x4 * x5,
            x3 * x4 * x6,
            x3 * x4 * x7,
            x3 * x4 * x8,
            x3 * x5 * x6,
            x3 * x5 * x7,
            x3 * x5 * x8,
            x3 * x6 * x7,
            x3 * x6 * x8,
            x3 * x7 * x8,
            x4 * x5 * x6,
            x4 * x5 * x7,
            x4 * x5 * x8,
            x4 * x6 * x7,
            x4 * x6 * x8,
            x4 * x7 * x8,
            x5 * x6 * x7,
            x5 * x6 * x8,
            x5 * x7 * x8,
            x6 * x7 * x8,
            x1 * x2 * x3 * x4,
            x1 * x2 * x3 * x5,
            x1 * x2 * x3 * x6,
            x1 * x2 * x3 * x7,
            x1 * x2 * x3 * x8,
            x1 * x2 * x4 * x5,
            x1 * x2 * x4 * x6,
            x1 * x2 * x4 * x7,
            x1 * x2 * x4 * x8,
            x1 * x2 * x5 * x6,
            x1 * x2 * x5 * x7,
            x1 * x2 * x5 * x8,
            x1 * x2 * x6 * x7,
            x1 * x2 * x6 * x8,
            x1 * x2 * x7 * x8,
            x1 * x3 * x4 * x5,
            x1 * x3 * x4 * x6,
            x1 * x3 * x4 * x7,
            x1 * x3 * x4 * x8,
            x1 * x3 * x5 * x6,
            x1 * x3 * x5 * x7,
            x1 * x3 * x5 * x8,
            x1 * x3 * x6 * x7,
            x1 * x3 * x6 * x8,
            x1 * x3 * x7 * x8,
            x1 * x4 * x5 * x6,
            x1 * x4 * x5 * x7,
            x1 * x4 * x5 * x8,
            x1 * x4 * x6 * x7,
            x1 * x4 * x6 * x8,
            x1 * x4 * x7 * x8,
            x1 * x5 * x6 * x7,
            x1 * x5 * x6 * x8,
            x1 * x5 * x7 * x8,
            x1 * x6 * x7 * x8,
            x2 * x3 * x4 * x5,
            x2 * x3 * x4 * x6,
            x2 * x3 * x4 * x7,
            x2 * x3 * x4 * x8,
            x2 * x3 * x5 * x6,
            x2 * x3 * x5 * x7,
            x2 * x3 * x5 * x8,
            x2 * x3 * x6 * x7,
            x2 * x3 * x6 * x8,
            x2 * x3 * x7 * x8,
            x2 * x4 * x5 * x6,
            x2 * x4 * x5 * x7,
            x2 * x4 * x5 * x8,
            x2 * x4 * x6 * x7,
            x2 * x4 * x6 * x8,
            x2 * x4 * x7 * x8,
            x2 * x5 * x6 * x7,
            x2 * x5 * x6 * x8,
            x2 * x5 * x7 * x8,
            x2 * x6 * x7 * x8,
            x3 * x4 * x5 * x6,
            x3 * x4 * x5 * x7,
            x3 * x4 * x5 * x8,
            x3 * x4 * x6 * x7,
            x3 * x4 * x6 * x8,
            x3 * x4 * x7 * x8,
            x3 * x5 * x6 * x7,
            x3 * x5 * x6 * x8,
            x3 * x5 * x7 * x8,
            x3 * x6 * x7 * x8,
            x4 * x5 * x6 * x7,
            x4 * x5 * x6 * x8,
            x4 * x5 * x7 * x8,
            x4 * x6 * x7 * x8,
            x5 * x6 * x7 * x8,
            x1 * x2 * x3 * x4 * x5,
            x1 * x2 * x3 * x4 * x6,
            x1 * x2 * x3 * x4 * x7,
            x1 * x2 * x3 * x4 * x8,
            x1 * x2 * x3 * x5 * x6,
            x1 * x2 * x3 * x5 * x7,
            x1 * x2 * x3 * x5 * x8,
            x1 * x2 * x3 * x6 * x7,
            x1 * x2 * x3 * x6 * x8,
            x1 * x2 * x3 * x7 * x8,
            x1 * x2 * x4 * x5 * x6,
            x1 * x2 * x4 * x5 * x7,
            x1 * x2 * x4 * x5 * x8,
            x1 * x2 * x4 * x6 * x7,
            x1 * x2 * x4 * x6 * x8,
            x1 * x2 * x4 * x7 * x8,
            x1 * x2 * x5 * x6 * x7,
            x1 * x2 * x5 * x6 * x8,
            x1 * x2 * x5 * x7 * x8,
            x1 * x2 * x6 * x7 * x8,
            x1 * x3 * x4 * x5 * x6,
            x1 * x3 * x4 * x5 * x7,
            x1 * x3 * x4 * x5 * x8,
            x1 * x3 * x4 * x6 * x7,
            x1 * x3 * x4 * x6 * x8,
            x1 * x3 * x4 * x7 * x8,
            x1 * x3 * x5 * x6 * x7,
            x1 * x3 * x5 * x6 * x8,
            x1 * x3 * x5 * x7 * x8,
            x1 * x3 * x6 * x7 * x8,
            x1 * x4 * x5 * x6 * x7,
            x1 * x4 * x5 * x6 * x8,
            x1 * x4 * x5 * x7 * x8,
            x1 * x4 * x6 * x7 * x8,
            x1 * x5 * x6 * x7 * x8,
            x2 * x3 * x4 * x5 * x6,
            x2 * x3 * x4 * x5 * x7,
            x2 * x3 * x4 * x5 * x8,
            x2 * x3 * x4 * x6 * x7,
            x2 * x3 * x4 * x6 * x8,
            x2 * x3 * x4 * x7 * x8,
            x2 * x3 * x5 * x6 * x7,
            x2 * x3 * x5 * x6 * x8,
            x2 * x3 * x5 * x7 * x8,
            x2 * x3 * x6 * x7 * x8,
            x2 * x4 * x5 * x6 * x7,
            x2 * x4 * x5 * x6 * x8,
            x2 * x4 * x5 * x7 * x8,
            x2 * x4 * x6 * x7 * x8,
            x2 * x5 * x6 * x7 * x8,
            x3 * x4 * x5 * x6 * x7,
            x3 * x4 * x5 * x6 * x8,
            x3 * x4 * x5 * x7 * x8,
            x3 * x4 * x6 * x7 * x8,
            x3 * x5 * x6 * x7 * x8,
            x4 * x5 * x6 * x7 * x8,
            x1 * x2 * x3 * x4 * x5 * x6,
            x1 * x2 * x3 * x4 * x5 * x7,
            x1 * x2 * x3 * x4 * x5 * x8,
            x1 * x2 * x3 * x4 * x6 * x7,
            x1 * x2 * x3 * x4 * x6 * x8,
            x1 * x2 * x3 * x4 * x7 * x8,
            x1 * x2 * x3 * x5 * x6 * x7,
            x1 * x2 * x3 * x5 * x6 * x8,
            x1 * x2 * x3 * x5 * x7 * x8,
            x1 * x2 * x3 * x6 * x7 * x8,
            x1 * x2 * x4 * x5 * x6 * x7,
            x1 * x2 * x4 * x5 * x6 * x8,
            x1 * x2 * x4 * x5 * x7 * x8,
            x1 * x2 * x4 * x6 * x7 * x8,
            x1 * x2 * x5 * x6 * x7 * x8,
            x1 * x3 * x4 * x5 * x6 * x7,
            x1 * x3 * x4 * x5 * x6 * x8,
            x1 * x3 * x4 * x5 * x7 * x8,
            x1 * x3 * x4 * x6 * x7 * x8,
            x1 * x3 * x5 * x6 * x7 * x8,
            x1 * x4 * x5 * x6 * x7 * x8,
            x2 * x3 * x4 * x5 * x6 * x7,
            x2 * x3 * x4 * x5 * x6 * x8,
            x2 * x3 * x4 * x5 * x7 * x8,
            x2 * x3 * x4 * x6 * x7 * x8,
            x2 * x3 * x5 * x6 * x7 * x8,
            x2 * x4 * x5 * x6 * x7 * x8,
            x3 * x4 * x5 * x6 * x7 * x8,
            x1 * x2 * x3 * x4 * x5 * x6 * x7,
            x1 * x2 * x3 * x4 * x5 * x6 * x8,
            x1 * x2 * x3 * x4 * x5 * x7 * x8,
            x1 * x2 * x3 * x4 * x6 * x7 * x8,
            x1 * x2 * x3 * x5 * x6 * x7 * x8,
            x1 * x2 * x4 * x5 * x6 * x7 * x8,
            x1 * x3 * x4 * x5 * x6 * x7 * x8,
            x2 * x3 * x4 * x5 * x6 * x7 * x8,
            x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8,
            x1 * x1 - S,
            x2 * x2 - S,
            x3 * x3 - S,
            x4 * x4 - S,
            x5 * x5 - S,
            x6 * x6 - S,
            x7 * x7 - S,
            x8 * x8 - S,
        ]


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sys.exit(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
