import sys
from PyQt5 import QtWidgets
import os
import numpy as np
import main

os.environ['QT_MAC_WANTS_LAYER'] = '1'
np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    qtapp = QtWidgets.QApplication.instance()
    if not qtapp:
        qtapp = QtWidgets.QApplication(sys.argv)
    app = main.MainViewer()
    app.show()
    sys.exit(qtapp.exec_())