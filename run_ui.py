import sys
from PyQt5 import QtWidgets
from ui import main
import os

os.environ['QT_MAC_WANTS_LAYER'] = '1'

if __name__ == '__main__':
    qtapp = QtWidgets.QApplication.instance()
    if not qtapp:
        qtapp = QtWidgets.QApplication(sys.argv)
    app = main.DataViewer()
    app.show()
    sys.exit(qtapp.exec_())