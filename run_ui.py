import sys
from PyQt5 import QtWidgets
from ui import main
import os
import numpy as np
import qdarktheme

os.environ['QT_MAC_WANTS_LAYER'] = '1'
np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    qtapp = QtWidgets.QApplication.instance()
    if not qtapp:
        qtapp = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    app = main.DataViewer()
    app.show()
    sys.exit(qtapp.exec_())