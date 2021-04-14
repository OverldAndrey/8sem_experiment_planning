from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLineEdit, QTableWidgetItem, \
    QHeaderView
import sys
import modeller

import numpy as np

QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


def calculate_params(la, dla, mu, dmu):
    mT1 = 1 / la
    dT1 = (1 / (la - dla) - 1 / (la + dla)) / 2

    mT2 = 1 / mu
    dT2 = (1 / (mu - dmu) - 1 / (mu + dmu)) / 2

    return mT1, dT1, mT2, dT2


def process_matrixes(initialMatrix):
    levelMatrix = [[0.0 for j in range(len(initialMatrix[0]))] for i in range(len(initialMatrix))]

    for i in range(len(levelMatrix)):
        for j in range(len(levelMatrix[0])):
            try:
                levelMatrix[i][j] = float(initialMatrix[i][j])
            except:
                levelMatrix[i][j] = 0.0

    # print(levelMatrix)

    planningMatrix = np.matrix(list(map(lambda row: row[:16], levelMatrix.copy()[:-1])))
    checkVector = np.array(levelMatrix.copy()[-1][:16])

    # print(planningMatrix)

    transposedPlanningMatrix = planningMatrix.transpose()

    # print(transposedPlanningMatrix)

    return np.linalg.inv(transposedPlanningMatrix * planningMatrix) * transposedPlanningMatrix, planningMatrix, checkVector


def convert_value_to_factor(min, max, value):
    return (value - (max + min) / 2.0) / ((max - min) / 2.0)

def convert_factor_to_value(min, max, factor):
    return factor * ((max - min) / 2.0) + (max + min) / 2.0


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("lab3.ui", self)

        self.la = 0
        self.dla = 1
        self.mu = 0
        self.dmu = 1
        self.tmax = 300

        self.read_params()

        self.set_free_point()

        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.bTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.tableWidget2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.bTableWidget2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.init_table2()

    @pyqtSlot(name='on_calculateButton_clicked')
    def on_process(self):
        print('process')

        layout = self.ui.externalLayout.itemAt(0)
        print(layout)

        self.read_params()

        la = self.la
        dla = self.dla
        mu = self.mu
        dmu = self.dmu
        tmax = self.tmax

        mT1, dT1, mT2, dT2 = calculate_params(la, dla, mu, dmu)

        model = modeller.Model(mT1, dT1, mT2, dT2, 2, 1, 0)

        print('start')

        ro = 2 * la / mu
        avg_queue_size, avg_queue_time, processed_requests = model.time_based_modellingg(tmax, 0.001)

        result = f'Расчетная загрузка системы: {ro}\n' \
                 f'Среднее количество заявок в системе: {avg_queue_size}\n' \
                 f'Среднее время ожидания: {avg_queue_time}\n' \
                 f'Обработано заявок: {processed_requests}'

        QMessageBox.information(self, 'Result', result)

        self.set_free_point()

    @pyqtSlot(name='on_calculateModelButton_clicked')
    def on_calculate_model(self):
        self.calculate_ffe()
        self.calculate_pfe()

    def calculate_ffe(self):
        layout = self.ui.externalLayout.itemAt(1)
        tableWidget = self.ui.tableWidget

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()

        planningTable = [[tableWidget.item(i, j).text() for j in range(cols)] for i in range(rows)]

        coefMatrix, planningMatrix, checkVector = process_matrixes(planningTable)

        factorMatrix = np.matrix(list(map(lambda row: row[1:5], planningTable.copy())))

        Y = [0 for i in range(17)]

        for i in range(len(factorMatrix.tolist())):
            la = convert_factor_to_value(Xmin[0], Xmax[0], float(factorMatrix.item((i, 0))))
            dla = convert_factor_to_value(Xmin[1], Xmax[1], float(factorMatrix.item((i, 1))))
            mu = convert_factor_to_value(Xmin[2], Xmax[2], float(factorMatrix.item((i, 2))))
            dmu = convert_factor_to_value(Xmin[3], Xmax[3], float(factorMatrix.item((i, 3))))

            # print(la, dla, mu, dmu)
            mT1, dT1, mT2, dT2 = calculate_params(la, dla, mu, dmu)

            model = modeller.Model(mT1, dT1, mT2, dT2, 2, 1, 0)

            avg_queue_size, avg_queue_time, processed_requests = model.time_based_modellingg(100, 0.001)

            # print(avg_queue_time)
            Y[i] = avg_queue_time
            tableWidget.setItem(i, 16, QTableWidgetItem(str(round(avg_queue_time, 4))))

        Yt = [Y[-1]]
        Y = np.array(Y[:-1])
        print("calculated ffe")

        B = (coefMatrix @ Y).tolist()[0]
        self.set_b_table(B, self.ui.bTableWidget)

        # print(B[:5])
        Yl = np.array(list(map(lambda row: row[:5], planningMatrix.tolist() + [checkVector.tolist()]))) @ np.array(
            B[:5])
        Ypn = np.array(planningMatrix.tolist() + [checkVector.tolist()]) @ np.array(B)
        resYList = Y.tolist() + Yt
        for i in range(len(resYList)):
            tableWidget.setItem(i, 17, QTableWidgetItem(str(round(Yl.tolist()[i], 4))))
            tableWidget.setItem(i, 18, QTableWidgetItem(str(round(Ypn.tolist()[i], 4))))
            tableWidget.setItem(i, 19, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Yl.tolist()[i], 6), 6)))))
            tableWidget.setItem(i, 20, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Ypn.tolist()[i], 6), 6)))))

    def calculate_pfe(self):
        tableWidget = self.ui.tableWidget2

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()

        planningTable = [[float(tableWidget.item(i, j).text()) for j in range(16)] for i in range(rows)]
        factorMatrix = np.matrix(list(map(lambda row: row[1:5], planningTable.copy())))
        checkVector = np.array(planningTable.copy()[-1][:16])

        Y = [0 for i in range(9)]

        for i in range(len(factorMatrix.tolist())):
            la = convert_factor_to_value(Xmin[0], Xmax[0], float(factorMatrix.item((i, 0))))
            dla = convert_factor_to_value(Xmin[1], Xmax[1], float(factorMatrix.item((i, 1))))
            mu = convert_factor_to_value(Xmin[2], Xmax[2], float(factorMatrix.item((i, 2))))
            dmu = convert_factor_to_value(Xmin[3], Xmax[3], float(factorMatrix.item((i, 3))))

            # print(la, dla, mu, dmu)
            mT1, dT1, mT2, dT2 = calculate_params(la, dla, mu, dmu)

            model = modeller.Model(mT1, dT1, mT2, dT2, 2, 1, 0)

            avg_queue_size, avg_queue_time, processed_requests = model.time_based_modellingg(100, 0.001)

            # print(avg_queue_time)
            Y[i] = avg_queue_time
            tableWidget.setItem(i, 16, QTableWidgetItem(str(round(avg_queue_time, 4))))

        Yt = [Y[-1]]
        Y = np.array(Y[:-1])
        print("calculated pfe")

        B = [np.array([float(planningTable[i][k]) / len(Y) for i in range(len(Y))]) @ Y for k in range(16)]

        for i in range(0, 16):
            B[i] = B[i] / 2

        self.set_b_table(B, self.ui.bTableWidget2)

        Yl = np.array(list(map(lambda row: row[:5], planningTable + [checkVector.tolist()]))) @ np.array(
            B[:5])
        Ypn = np.array(planningTable + [checkVector.tolist()]) @ np.array(B)
        resYList = Y.tolist() + Yt
        for i in range(len(resYList)):
            tableWidget.setItem(i, 17, QTableWidgetItem(str(round(Yl.tolist()[i], 4))))
            tableWidget.setItem(i, 18, QTableWidgetItem(str(round(Ypn.tolist()[i], 4))))
            tableWidget.setItem(i, 19, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Yl.tolist()[i], 6), 6)))))
            tableWidget.setItem(i, 20, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Ypn.tolist()[i], 6), 6)))))

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

                    if objName == 'arriveIntensity':
                        widget.setText(str(round((Xmin[0] + Xmax[0]) / 2, 4)))
                        self.la = round((Xmin[0] + Xmax[0]) / 2, 4)
                    elif objName == 'processIntensity':
                        widget.setText(str(round((Xmin[2] + Xmax[2]) / 2, 4)))
                        self.mu = round((Xmin[2] + Xmax[2]) / 2, 4)
                    elif objName == 'arriveIntensityDispersion':
                        widget.setText(str(round((Xmin[1] + Xmax[1]) / 2, 4)))
                        self.dla = round((Xmin[1] + Xmax[1]) / 2, 4)
                    elif objName == 'processIntensityDispersion':
                        widget.setText(str(round((Xmin[3] + Xmax[3]) / 2, 4)))
                        self.dmu = round((Xmin[3] + Xmax[3]) / 2, 4)

        self.set_free_point()

    def read_params(self):
        layout = self.ui.externalLayout.itemAt(0)
        # print(layout)

        la = 0
        dla = 1
        mu = 0
        dmu = 1

        tmax = 300

        for i in range(layout.rowCount()):
            for j in range(layout.columnCount()):
                if layout.itemAtPosition(i, j):
                    widget = layout.itemAtPosition(i, j).widget()
                    objName = widget.objectName()

                    if not isinstance(widget, QLineEdit):
                        continue

                    try:
                        if objName == 'arriveIntensity':
                            print('arrive')
                            la = float(widget.text())
                        elif objName == 'processIntensity':
                            print('process')
                            mu = float(widget.text())
                        elif objName == 'arriveIntensityDispersion':
                            print('arrive disp')
                            dla = float(widget.text())
                        elif objName == 'processIntensityDispersion':
                            print('process disp')
                            dmu = float(widget.text())
                        elif objName == 'modellingTime':
                            print('time')
                            tmax = float(widget.text())
                    except ValueError:
                        QMessageBox.warning(self, 'Error', 'Ошибка ввода')
                        return

        if (la <= 0 or dla >= la) or (mu <= 0 or dmu >= mu):
            QMessageBox.warning(self, 'Error', 'Интенсивности должны быть больше 0')
            return

        self.la = la
        self.mu = mu
        self.dla = dla
        self.dmu = dmu
        self.tmax = tmax

        return la, dla, mu, dmu, tmax

    def read_model_params(self):
        layout = self.ui.externalLayout.itemAt(1)

        Xmin = [0, 0, 0, 0]
        Xmax = [0, 0, 0, 0]

        for i in range(layout.rowCount()):
            for j in range(layout.columnCount()):
                if layout.itemAtPosition(i, j):
                    widget = layout.itemAtPosition(i, j).widget()
                    objName = widget.objectName()

                    if not isinstance(widget, QLineEdit):
                        continue

                    try:
                        if objName == 'arriveIntensityMin':
                            Xmin[0] = float(widget.text())
                        elif objName == 'arriveIntensityMax':
                            Xmax[0] = float(widget.text())
                        elif objName == 'arriveIntensityDispersionMin':
                            Xmin[1] = float(widget.text())
                        elif objName == 'arriveIntensityDispersionMax':
                            Xmax[1] = float(widget.text())
                        elif objName == 'processIntensityMin':
                            Xmin[2] = float(widget.text())
                        elif objName == 'processIntensityMax':
                            Xmax[2] = float(widget.text())
                        elif objName == 'processIntensityDispersionMin':
                            Xmin[3] = float(widget.text())
                        elif objName == 'processIntensityDispersionMax':
                            Xmax[3] = float(widget.text())
                    except ValueError:
                        QMessageBox.warning(self, 'Error', 'Ошибка ввода')
                        return

        return Xmin, Xmax

    def set_free_point(self):
        tableWidget = self.ui.tableWidget
        tableWidget2 = self.ui.tableWidget2

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()
        x1 = convert_value_to_factor(Xmin[0], Xmax[0], self.la)
        x2 = convert_value_to_factor(Xmin[1], Xmax[1], self.dla)
        x3 = convert_value_to_factor(Xmin[2], Xmax[2], self.mu)
        x4 = convert_value_to_factor(Xmin[3], Xmax[3], self.dmu)
        # print(convert_value_to_factor(Xmin[0], Xmax[0], self.la),
        #       convert_factor_to_value(Xmin[0], Xmax[0], convert_value_to_factor(Xmin[0], Xmax[0], self.la)))

        tableWidget.setItem(16, 0, QTableWidgetItem(str(round(1, 4))))
        tableWidget.setItem(16, 1, QTableWidgetItem(str(round(x1, 4))))
        tableWidget.setItem(16, 2, QTableWidgetItem(str(round(x2, 4))))
        tableWidget.setItem(16, 3, QTableWidgetItem(str(round(x3, 4))))
        tableWidget.setItem(16, 4, QTableWidgetItem(str(round(x4, 4))))
        tableWidget.setItem(16, 5, QTableWidgetItem(str(round(x1 * x2, 4))))
        tableWidget.setItem(16, 6, QTableWidgetItem(str(round(x1 * x3, 4))))
        tableWidget.setItem(16, 7, QTableWidgetItem(str(round(x1 * x4, 4))))
        tableWidget.setItem(16, 8, QTableWidgetItem(str(round(x2 * x3, 4))))
        tableWidget.setItem(16, 9, QTableWidgetItem(str(round(x2 * x4, 4))))
        tableWidget.setItem(16, 10, QTableWidgetItem(str(round(x3 * x4, 4))))
        tableWidget.setItem(16, 11, QTableWidgetItem(str(round(x1 * x2 * x3, 4))))
        tableWidget.setItem(16, 12, QTableWidgetItem(str(round(x1 * x2 * x4, 4))))
        tableWidget.setItem(16, 13, QTableWidgetItem(str(round(x1 * x3 * x4, 4))))
        tableWidget.setItem(16, 14, QTableWidgetItem(str(round(x2 * x3 * x4, 4))))
        tableWidget.setItem(16, 15, QTableWidgetItem(str(round(x1 * x2 * x3 * x4, 4))))

        tableWidget2.setItem(8, 0, QTableWidgetItem(str(round(1, 4))))
        tableWidget2.setItem(8, 1, QTableWidgetItem(str(round(x1, 4))))
        tableWidget2.setItem(8, 2, QTableWidgetItem(str(round(x2, 4))))
        tableWidget2.setItem(8, 3, QTableWidgetItem(str(round(x3, 4))))
        tableWidget2.setItem(8, 4, QTableWidgetItem(str(round(x1 * x2 * x3, 4))))
        tableWidget2.setItem(8, 5, QTableWidgetItem(str(round(x1 * x2, 4))))
        tableWidget2.setItem(8, 6, QTableWidgetItem(str(round(x1 * x3, 4))))
        tableWidget2.setItem(8, 7, QTableWidgetItem(str(round(x1 * x1 * x2 * x3, 4))))
        tableWidget2.setItem(8, 8, QTableWidgetItem(str(round(x2 * x3, 4))))
        tableWidget2.setItem(8, 9, QTableWidgetItem(str(round(x2 * x1 * x2 * x3, 4))))
        tableWidget2.setItem(8, 10, QTableWidgetItem(str(round(x3 * x1 * x2 * x3, 4))))
        tableWidget2.setItem(8, 11, QTableWidgetItem(str(round(x1 * x2 * x3, 4))))
        tableWidget2.setItem(8, 12, QTableWidgetItem(str(round(x1 * x2 * x1 * x2 * x3, 4))))
        tableWidget2.setItem(8, 13, QTableWidgetItem(str(round(x1 * x3 * x1 * x2 * x3, 4))))
        tableWidget2.setItem(8, 14, QTableWidgetItem(str(round(x2 * x3 * x1 * x2 * x3, 4))))
        tableWidget2.setItem(8, 15, QTableWidgetItem(str(round(x1 * x2 * x3 * x1 * x2 * x3, 4))))

    def set_b_table(self, B, table):
        for i in range(len(B)):
            table.setItem(0, i, QTableWidgetItem(str(round(B[i], 7))))

    def init_table2(self):
        table = self.ui.tableWidget2

        for i in range(table.rowCount() - 1):
            x1 = int(table.item(i, 1).text())
            x2 = int(table.item(i, 2).text())
            x3 = int(table.item(i, 3).text())
            x4 = x1 * x2 * x3

            table.setItem(i, 4, QTableWidgetItem(str(x4)))

            table.setItem(i, 5, QTableWidgetItem(str(x1 * x2)))
            table.setItem(i, 6, QTableWidgetItem(str(x1 * x3)))
            table.setItem(i, 7, QTableWidgetItem(str(x1 * x4)))
            table.setItem(i, 8, QTableWidgetItem(str(x2 * x3)))
            table.setItem(i, 9, QTableWidgetItem(str(x2 * x4)))
            table.setItem(i, 10, QTableWidgetItem(str(x3 * x4)))
            table.setItem(i, 11, QTableWidgetItem(str(x1 * x2 * x3)))
            table.setItem(i, 12, QTableWidgetItem(str(x1 * x2 * x4)))
            table.setItem(i, 13, QTableWidgetItem(str(x1 * x3 * x4)))
            table.setItem(i, 14, QTableWidgetItem(str(x2 * x3 * x4)))
            table.setItem(i, 15, QTableWidgetItem(str(x1 * x2 * x3 * x4)))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sys.exit(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
