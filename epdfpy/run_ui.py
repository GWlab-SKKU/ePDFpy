import sys
from PyQt5 import QtWidgets
from epdfpy.ui import main as ui_main
import os
import numpy as np
import qdarktheme

# os.environ['QT_MAC_WANTS_LAYER'] = '1'
# np.seterr(divide='ignore', invalid='ignore')

def main():
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    np.seterr(divide='ignore', invalid='ignore')

    qtapp = QtWidgets.QApplication.instance()
    if not qtapp:
        qtapp = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    app = ui_main.DataViewer()
    app.show()
    sys.exit(qtapp.exec_())

if __name__ == '__main__':
    main()
    # qtapp = QtWidgets.QApplication.instance()
    # if not qtapp:
    #     qtapp = QtWidgets.QApplication(sys.argv)
    # qdarktheme.setup_theme("auto")
    # app = main.DataViewer()
    # app.show()
    # sys.exit(qtapp.exec_())