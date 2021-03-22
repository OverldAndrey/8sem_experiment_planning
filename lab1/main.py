from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QLineEdit
from matplotlib import pyplot
import sys
import modeller
import math

import numpy as np


def calculate_params(la, dla, mu, dmu):
    mT1 = 1 / la
    dT1 = (1 / (la - dla) - 1 / (la + dla)) / 2
    # print(mT1, dT1)

    mT2 = 1 / mu
    dT2 = (1 / (mu - dmu) - 1 / (mu + dmu)) / 2
    # print(mT2, dT2)

    return mT1, dT1, mT2, dT2


def calculate_model_for_graph(calc_params, dla, dmu):
    la = 1
    # dla = 0.05
    mu = 10
    # dmu = 1

    loads1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    loads2 = np.arange(0.9, 0.999, 0.001)

    times1 = []
    times2 = []

    tmax = 100

    for p in loads1:
        la = p * mu
        m1, d1, m2, d2 = calc_params(la, dla, mu, dmu)

        model = modeller.Model(m1, d1, m2, d2, 1, 1, 0)

        _, t = model.time_based_modelling(tmax, 0.01)
        # print(t)

        times1.append(t)

    for p in loads2:
        la = p * mu
        m1, d1, m2, d2 = calc_params(la, dla, mu, dmu)

        model = modeller.Model(m1, d1, m2, d2, 1, 1, 0)

        _, t = model.time_based_modelling(tmax, 0.001)
        # print(t)

        times2.append(t)

    return np.concatenate(([0], loads1, loads2)), np.concatenate(([times1[0]], times1, times2))


def show_plot(x, y):
    pyplot.title('Среднее время ожидания')
    pyplot.grid(True)
    # pyplot.plot(Xdata, Ydata_t)
    pyplot.plot(x, y)
    pyplot.axis([0, 1, 0, 0.3])
    pyplot.xlabel("Коэффикиент загрузки")
    pyplot.ylabel("Среднее время пребывания в очереди")
    pyplot.show()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("lab1.ui", self)

        # print(self.ui.layout().calculateButton)

    @pyqtSlot(name='on_calculateButton_clicked')
    def on_process(self):
        print('process')

        print(self.ui.layout.rowCount())

        la = 0
        dla = 1
        mu = 0
        dmu = 1

        tmax = 300

        for i in range(self.ui.layout.rowCount()):
            for j in range(self.ui.layout.columnCount()):
                if self.ui.layout.itemAtPosition(i, j):
                    widget = self.ui.layout.itemAtPosition(i, j).widget()
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
                        QMessageBox.warning(self, 'Error', 'ValueError')
                        return

        mT1, dT1, mT2, dT2 = calculate_params(la, dla, mu, dmu)

        model = modeller.Model(mT1, dT1, mT2, dT2, 1, 1, 0)

        print('start')
        QMessageBox.information(self, 'Result', str(model.time_based_modellingg(tmax, 0.001)))

        # x, y = calculate_model_for_graph(calculate_params, 0.05, 0.01)
        # show_plot(x, y)
        #
        # x, y = calculate_model_for_graph(calculate_params, 0.05, 0.1)
        # show_plot(x, y)
        #
        # x, y = calculate_model_for_graph(calculate_params, 0.05, 0.5)
        # show_plot(x, y)
        #
        # x, y = calculate_model_for_graph(calculate_params, 0.05, 1)
        # show_plot(x, y)

    # @pyqtSlot(name='on_pushButton_clicked')
    # def _parse_parameters(self):
    #     try:
    #         ui = self.ui
    #         uniform_a = float(ui.lineEdit_generator_a.text())
    #         uniform_b = float(ui.lineEdit_generator_b.text())
    #         expo_l = float(ui.lineEdit_servicemachine_lambda.text())
    #         req_count = int(ui.lineEdit_request_count.text())
    #         reenter = float(ui.lineEdit_reenter_probability.text())
    #         method = ui.comboBox_method.currentIndex()
    #
    #         model = modeller.Model(uniform_a, uniform_b, expo_l, reenter)
    #         if method == 0:
    #             self._show_results(model.event_based_modelling(req_count))
    #         else:
    #             delta_t = float(ui.lineEdit_deltat.text())
    #             self._show_results(model.time_based_modelling(req_count, delta_t))
    #     except ValueError:
    #         QMessageBox.warning(self, 'Ошибка', 'Ошибка в данных!')
    #     except Exception as e:
    #         QMessageBox.critical(self, 'Ошибка', e)
    #
    # def _show_results(self, results):
    #     ui = self.ui
    #     ui.lineEdit_res_request_count.setText(str(results[0]))
    #     ui.lineEdit_res_reentered_count.setText(str(results[1]))
    #     ui.lineEdit_res_max_queue_size.setText(str(results[2]))
    #     ui.lineEdit_res_time.setText('{:.2f}'.format(results[3]))
    #
    # @pyqtSlot(int)
    # def on_comboBox_method_currentIndexChanged(self, index):
    #     if index == 1:
    #         # Δt
    #         visibility = True
    #     else:
    #         visibility = False
    #         # events
    #     self.ui.lineEdit_deltat.setEnabled(visibility)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sys.exit(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
