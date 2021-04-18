from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLineEdit, QTableWidgetItem, \
    QHeaderView
import sys
import modeller

import numpy as np

QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


def calculate_params(la1, dla1, la2, dla2, mu, dmu):
    mT11 = 1 / la1
    dT11 = (1 / (la1 - dla1) - 1 / (la1 + dla1)) / 2

    mT12 = 1 / la2
    dT12 = (1 / (la2 - dla2) - 1 / (la2 + dla2)) / 2

    mT2 = 1 / mu
    dT2 = (1 / (mu - dmu) - 1 / (mu + dmu)) / 2

    return mT11, dT11, mT12, dT12, mT2, dT2


def process_matrixes(initialMatrix):
    levelMatrix = [[0.0 for j in range(len(initialMatrix[0]))] for i in range(len(initialMatrix))]

    for i in range(len(levelMatrix)):
        for j in range(len(levelMatrix[0])):
            try:
                levelMatrix[i][j] = float(initialMatrix[i][j])
            except:
                levelMatrix[i][j] = 0.0

    # print(levelMatrix)

    planningMatrix = np.matrix(list(map(lambda row: row[:64], levelMatrix.copy()[:-1])))
    checkVector = np.array(levelMatrix.copy()[-1][:64])

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

        self.la1 = 0
        self.dla1 = 1
        self.la2 = 0
        self.dla2 = 1
        self.mu = 0
        self.dmu = 1
        self.tmax = 300

        self.read_params()

        self.init_table()
        self.init_table2()

        self.set_free_point()

        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.bTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.tableWidget2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.bTableWidget2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

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
        mu = self.mu
        dmu = self.dmu
        tmax = self.tmax

        mT11, dT11, mT12, dT12, mT2, dT2 = calculate_params(la1, dla1, la2, dla2, mu, dmu)

        model = modeller.Model([mT11, mT12], [dT11, dT12], mT2, dT2, 2, 1, 0)

        print('start')

        ro = (la1 + la2) / mu
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
        print(Xmin, Xmax)

        planningTable = [[tableWidget.item(i, j).text() for j in range(cols)] for i in range(rows)]

        coefMatrix, planningMatrix, checkVector = process_matrixes(planningTable)

        factorMatrix = np.matrix(list(map(lambda row: row[1:7], planningTable.copy())))

        Y = [0 for i in range(65)]

        for i in range(len(factorMatrix.tolist())):
            la1 = convert_factor_to_value(Xmin[0], Xmax[0], float(factorMatrix.item((i, 0))))
            dla1 = convert_factor_to_value(Xmin[1], Xmax[1], float(factorMatrix.item((i, 1))))
            la2 = convert_factor_to_value(Xmin[2], Xmax[2], float(factorMatrix.item((i, 2))))
            dla2 = convert_factor_to_value(Xmin[3], Xmax[3], float(factorMatrix.item((i, 3))))
            mu = convert_factor_to_value(Xmin[4], Xmax[4], float(factorMatrix.item((i, 4))))
            dmu = convert_factor_to_value(Xmin[5], Xmax[5], float(factorMatrix.item((i, 5))))

            # print(la, dla, mu, dmu)
            mT11, dT11, mT12, dT12, mT2, dT2 = calculate_params(la1, dla1, la2, dla2, mu, dmu)

            model = modeller.Model([mT11, mT12], [dT11, dT12], mT2, dT2, 2, 1, 0)

            avg_queue_size, avg_queue_time, processed_requests = model.time_based_modellingg(100, 0.001)

            # print(avg_queue_time)
            Y[i] = avg_queue_time
            tableWidget.setItem(i, 64, QTableWidgetItem(str(round(avg_queue_time, 4))))

        Yt = [Y[-1]]
        Y = np.array(Y[:-1])
        print("calculated ffe")

        B = (coefMatrix @ Y).tolist()[0]
        self.set_b_table(B, self.ui.bTableWidget)

        # print(B[:5])
        Yl = np.array(list(map(lambda row: row[:7], planningMatrix.tolist() + [checkVector.tolist()]))) @ np.array(
            B[:7])
        Ypn = np.array(planningMatrix.tolist() + [checkVector.tolist()]) @ np.array(B)
        resYList = Y.tolist() + Yt
        for i in range(len(resYList)):
            tableWidget.setItem(i, 65, QTableWidgetItem(str(round(Yl.tolist()[i], 4))))
            tableWidget.setItem(i, 66, QTableWidgetItem(str(round(Ypn.tolist()[i], 4))))
            tableWidget.setItem(i, 67, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Yl.tolist()[i], 6), 6)))))
            tableWidget.setItem(i, 68, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Ypn.tolist()[i], 6), 6)))))

    def calculate_pfe(self):
        tableWidget = self.ui.tableWidget2

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()

        planningTable = [[float(tableWidget.item(i, j).text()) for j in range(64)] for i in range(rows)]
        factorMatrix = np.matrix(list(map(lambda row: row[1:7], planningTable.copy())))
        checkVector = np.array(planningTable.copy()[-1][:64])

        Y = [0 for i in range(9)]

        for i in range(len(factorMatrix.tolist())):
            la1 = convert_factor_to_value(Xmin[0], Xmax[0], float(factorMatrix.item((i, 0))))
            dla1 = convert_factor_to_value(Xmin[1], Xmax[1], float(factorMatrix.item((i, 1))))
            la2 = convert_factor_to_value(Xmin[2], Xmax[2], float(factorMatrix.item((i, 2))))
            dla2 = convert_factor_to_value(Xmin[3], Xmax[3], float(factorMatrix.item((i, 3))))
            mu = convert_factor_to_value(Xmin[4], Xmax[4], float(factorMatrix.item((i, 4))))
            dmu = convert_factor_to_value(Xmin[5], Xmax[5], float(factorMatrix.item((i, 5))))

            # print(la, dla, mu, dmu)
            mT11, dT11, mT12, dT12, mT2, dT2 = calculate_params(la1, dla1, la2, dla2, mu, dmu)

            model = modeller.Model([mT11, mT12], [dT11, dT12], mT2, dT2, 2, 1, 0)

            avg_queue_size, avg_queue_time, processed_requests = model.time_based_modellingg(100, 0.001)

            # print(avg_queue_time)
            Y[i] = avg_queue_time
            tableWidget.setItem(i, 64, QTableWidgetItem(str(round(avg_queue_time, 4))))

        Yt = [Y[-1]]
        Y = np.array(Y[:-1])
        print("calculated pfe")

        B = [np.array([float(planningTable[i][k]) / len(Y) for i in range(len(Y))]) @ Y for k in range(64)]

        for i in range(0, 64):
            B[i] = B[i] / self.count_eq_rows(planningTable, i)

        self.set_b_table(B, self.ui.bTableWidget2)

        Yl = np.array(list(map(lambda row: row[:7], planningTable + [checkVector.tolist()]))) @ np.array(
            B[:7])
        Ypn = np.array(planningTable + [checkVector.tolist()]) @ np.array(B)
        resYList = Y.tolist() + Yt
        for i in range(len(resYList)):
            tableWidget.setItem(i, 65, QTableWidgetItem(str(round(Yl.tolist()[i], 4))))
            tableWidget.setItem(i, 66, QTableWidgetItem(str(round(Ypn.tolist()[i], 4))))
            tableWidget.setItem(i, 67, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Yl.tolist()[i], 6), 6)))))
            tableWidget.setItem(i, 68, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Ypn.tolist()[i], 6), 6)))))

    def count_eq_rows(self, plTable, i):
        count = 0

        for j in range(len(plTable[0])):
            eq = True
            for k in range(len(plTable)):
                eq = eq and (plTable[k][j] == plTable[k][i])
            if eq:
                count += 1

        return count

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
                    elif objName == 'processIntensity':
                        widget.setText(str(round((Xmin[4] + Xmax[4]) / 2, 4)))
                        self.mu = round((Xmin[4] + Xmax[4]) / 2, 4)
                    elif objName == 'arriveIntensityDispersion1':
                        widget.setText(str(round((Xmin[1] + Xmax[1]) / 2, 4)))
                        self.dla1 = round((Xmin[1] + Xmax[1]) / 2, 4)
                    elif objName == 'arriveIntensityDispersion2':
                        widget.setText(str(round((Xmin[3] + Xmax[3]) / 2, 4)))
                        self.dla2 = round((Xmin[3] + Xmax[3]) / 2, 4)
                    elif objName == 'processIntensityDispersion':
                        widget.setText(str(round((Xmin[5] + Xmax[5]) / 2, 4)))
                        self.dmu = round((Xmin[5] + Xmax[5]) / 2, 4)

        self.set_free_point()

    def read_params(self):
        layout = self.ui.externalLayout.itemAt(0)
        # print(layout)

        la1 = 0
        dla1 = 1
        la2 = 0
        dla2 = 1
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
                        if objName == 'arriveIntensity1':
                            print('arrive 1')
                            la1 = float(widget.text())
                        elif objName == 'arriveIntensity2':
                            print('arrive 2')
                            la2 = float(widget.text())
                        elif objName == 'processIntensity':
                            print('process')
                            mu = float(widget.text())
                        elif objName == 'arriveIntensityDispersion1':
                            print('arrive disp 1')
                            dla1 = float(widget.text())
                        elif objName == 'arriveIntensityDispersion2':
                            print('arrive disp 2')
                            dla2 = float(widget.text())
                        elif objName == 'processIntensityDispersion':
                            print('process disp')
                            dmu = float(widget.text())
                        elif objName == 'modellingTime':
                            print('time')
                            tmax = float(widget.text())
                    except ValueError:
                        QMessageBox.warning(self, 'Error', 'Ошибка ввода')
                        return

        if (la1 <= 0 or dla1 >= la1) or (la2 <= 0 or dla2 >= la2) or (mu <= 0 or dmu >= mu):
            QMessageBox.warning(self, 'Error', 'Интенсивности должны быть больше 0')
            return

        self.la1 = la1
        self.la2 = la2
        self.mu = mu
        self.dla1 = dla1
        self.dla2 = dla2
        self.dmu = dmu
        self.tmax = tmax

        # return la, dla, mu, dmu, tmax

    def read_model_params(self):
        layout = self.ui.externalLayout.itemAt(1)

        Xmin = [0, 0, 0, 0, 0, 0]
        Xmax = [0, 0, 0, 0, 0, 0]

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
                            Xmin[4] = float(widget.text())
                        elif objName == 'processIntensityMax':
                            Xmax[4] = float(widget.text())
                        elif objName == 'processIntensityDispersionMin':
                            Xmin[5] = float(widget.text())
                        elif objName == 'processIntensityDispersionMax':
                            Xmax[5] = float(widget.text())
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
        x1 = convert_value_to_factor(Xmin[0], Xmax[0], self.la1)
        x2 = convert_value_to_factor(Xmin[1], Xmax[1], self.dla1)
        x3 = convert_value_to_factor(Xmin[2], Xmax[2], self.la2)
        x4 = convert_value_to_factor(Xmin[3], Xmax[3], self.dla2)
        x5 = convert_value_to_factor(Xmin[4], Xmax[4], self.mu)
        x6 = convert_value_to_factor(Xmin[5], Xmax[5], self.dmu)
        # print(convert_value_to_factor(Xmin[0], Xmax[0], self.la),
        #       convert_factor_to_value(Xmin[0], Xmax[0], convert_value_to_factor(Xmin[0], Xmax[0], self.la)))

        x = self.get_factor_array(x1, x2, x3, x4, x5, x6)

        for i in range(64):
            tableWidget.setItem(64, i, QTableWidgetItem(str(round(x[i], 4))))

        x4 = x1 * x2
        x5 = x1 * x3
        x6 = x2 * x3

        x = self.get_factor_array(x1, x2, x3, x4, x5, x6)

        for i in range(64):
            tableWidget2.setItem(8, i, QTableWidgetItem(str(round(x[i], 4))))

    def set_b_table(self, B, table):
        for i in range(len(B)):
            table.setItem(0, i, QTableWidgetItem(str(round(B[i], 7))))

    def init_table(self):
        table = self.ui.tableWidget

        for i in range(table.rowCount() - 1):
            x1 = int(table.item(i, 1).text())
            x2 = int(table.item(i, 2).text())
            x3 = int(table.item(i, 3).text())
            x4 = int(table.item(i, 4).text())
            x5 = int(table.item(i, 5).text())
            x6 = int(table.item(i, 6).text())

            x = self.get_factor_array(x1, x2, x3, x4, x5, x6)

            for k in range(7, 64):
                table.setItem(i, k, QTableWidgetItem(str(x[k])))

    def init_table2(self):
        table = self.ui.tableWidget2

        for i in range(table.rowCount() - 1):
            x1 = int(table.item(i, 1).text())
            x2 = int(table.item(i, 2).text())
            x3 = int(table.item(i, 3).text())
            x4 = x1 * x2
            x5 = x1 * x3
            x6 = x2 * x3

            table.setItem(i, 4, QTableWidgetItem(str(x4)))
            table.setItem(i, 5, QTableWidgetItem(str(x5)))
            table.setItem(i, 6, QTableWidgetItem(str(x6)))

            x = self.get_factor_array(x1, x2, x3, x4, x5, x6)

            for k in range(7, 64):
                table.setItem(i, k, QTableWidgetItem(str(x[k])))

    def get_factor_array(self, x1, x2, x3, x4, x5, x6):
        return [
            1,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x1 * x5,
            x1 * x6,
            x2 * x3,
            x2 * x4,
            x2 * x5,
            x2 * x6,
            x3 * x4,
            x3 * x5,
            x3 * x6,
            x4 * x5,
            x4 * x6,
            x5 * x6,
            x1 * x2 * x3,
            x1 * x2 * x4,
            x1 * x2 * x5,
            x1 * x2 * x6,
            x1 * x3 * x4,
            x1 * x3 * x5,
            x1 * x3 * x6,
            x1 * x4 * x5,
            x1 * x4 * x6,
            x1 * x5 * x6,
            x2 * x3 * x4,
            x2 * x3 * x5,
            x2 * x3 * x6,
            x2 * x4 * x5,
            x2 * x4 * x6,
            x2 * x5 * x6,
            x3 * x4 * x5,
            x3 * x4 * x6,
            x3 * x5 * x6,
            x4 * x5 * x6,
            x1 * x2 * x3 * x4,
            x1 * x2 * x3 * x5,
            x1 * x2 * x3 * x6,
            x1 * x2 * x4 * x5,
            x1 * x2 * x4 * x6,
            x1 * x2 * x5 * x6,
            x1 * x3 * x4 * x5,
            x1 * x3 * x4 * x6,
            x1 * x3 * x5 * x6,
            x1 * x4 * x5 * x6,
            x2 * x3 * x4 * x5,
            x2 * x3 * x4 * x6,
            x2 * x3 * x5 * x6,
            x2 * x4 * x5 * x6,
            x3 * x4 * x5 * x6,
            x1 * x2 * x3 * x4 * x5,
            x1 * x2 * x3 * x4 * x6,
            x1 * x2 * x3 * x5 * x6,
            x1 * x2 * x4 * x5 * x6,
            x1 * x3 * x4 * x5 * x6,
            x2 * x3 * x4 * x5 * x6,
            x1 * x2 * x3 * x4 * x5 * x6,
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
