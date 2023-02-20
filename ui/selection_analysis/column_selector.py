from PyQt5 import QtWidgets, Qt
import pyqtgraph as pg
from PyQt5.QtCore import Qt

pg.setConfigOptions(antialias=True)

class DataViewer(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        viewer = ColumnSelector(self, "area02.q.csv")
        self.setCentralWidget(viewer)

class ColumnSelector(QtWidgets.QWidget):
    def __init__(self, mainWindow, dcs, func_when_finished=None, profile_extraction=None):
        super().__init__()
        self.func_when_finished = func_when_finished

        ############ ui init ###########
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        ## x axis
        x_axis_grp = QtWidgets.QHBoxLayout()
        self.lbl_x_axis = QtWidgets.QLabel("X axis")
        self.combo_x_axis = QtWidgets.QComboBox()
        combo_lst = ["None"]
        combo_lst.extend(dcs[0].pd_data.columns)
        self.combo_x_axis.addItems(combo_lst)
        x_axis_grp.addWidget(self.lbl_x_axis)
        x_axis_grp.addWidget(self.combo_x_axis)

        ## table
        self.table = pg.TableWidget(editable=False, sortable=False)
        self.table_load(dcs[0])

        # self.show_column()
        self.btn_ok = QtWidgets.QPushButton("OK")

        self.layout.addLayout(x_axis_grp)
        self.layout.addWidget(self.table)
        self.layout.addWidget(self.btn_ok)

        # signal binding
        self.btn_ok.clicked.connect(self.click_ok)

    def add_row(self):
        # todo
        pass

    def click_ok(self):
        rs = []
        for idx, chkbox in enumerate(self.checkbox_lst):
            if chkbox.isChecked():
                rs.append(self.table.item(1,idx).text())
        print(self.df[rs])
        if self.func_when_finished is not None:
            self.func_when_finished(self.combo_x_axis.currentText(),rs)
        self.close()

    def table_load(self, dc):
        ### file load ###
        df = dc.pd_data.copy()
        self.df = df.copy()
        columns = df.columns.to_list()
        df.loc[-2] = [''] * len(columns)
        df.loc[-1] = columns
        df.index = df.index + 2  # shifting index
        df = df.sort_index()  # sorting by index
        self.table.setData(df.to_dict(orient='index'))
        # self.table.horizontalHeader().clicked.connect(self.__mycell_clicked)
        # self.table.verticalHeader().clicked.connect(self.__mycell_clicked)
        self.checkbox_lst = []
        for idx, column in enumerate(columns):
            widget = QtWidgets.QCheckBox()
            widget.toggled.connect(lambda state, widget=widget, idx=idx: self.checkbox_clicked(state,widget,idx))
            widget.setChecked(True)
            self.checkbox_lst.append(widget)
            self.table.setCellWidget(0,idx,widget)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self.table_clicked)

        for clmn in range(len(columns)):
            self.table.item(0, clmn).setTextAlignment(Qt.AlignCenter)
            self.table.item(1, clmn).setTextAlignment(Qt.AlignCenter)
        # center align


    def checkbox_clicked(self,state, widget, idx):
        print(state)
        print(widget)
        print(idx)
        
    def table_clicked(self, row, column):
        # cell click - chkbox
        if row==0:
            self.checkbox_lst[column].toggle()
        # cell click - columns
        if row==1:
            name, ok = QtWidgets.QInputDialog.getText(self, 'Change name', 'Enter name:', text=self.table.item(row, column).text())
            if name=="":
                return
            self.table.item(row, column).setText(name)

if __name__ == "__main__":
    qtapp = QtWidgets.QApplication([])
    # QtWidgets.QMainWindow().show()
    viewer = DataViewer()
    viewer.show()
    qtapp.exec()