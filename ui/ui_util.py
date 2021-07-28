import typing

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QWidget
import numpy as np


class binding():
    def __init__(self, widget, event):
        self.functions = []
        self.event = event
        self.widget = widget
        pass
    def add_event(self, func):
        self.functions.append(func)
        pass

    def change_value(self,value):
        self.event.disconnect()
        self.widget.setValue(value)
        self.event.connect()


class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.validator = QRegExpValidator(QRegExp(r"[0-9]+[.][0-9]+"))
        self.setDecimals(10)

    def validate(self, input: str, pos: int):
        return self.validator.validate(input, pos)

    def textFromValue(self, v: float) -> str:
        # print("value:",v)
        # print("text:",super().textFromValue(v))
        str_v = str(v)
        return str(np.round(float(str_v),self.decimals()))

    def valueFromText(self, text: str) -> float:
        # print("text:",text)
        # print("value:",super().valueFromText(text))
        print("input:",str)
        print("output:", np.round(float(text),self.decimals()))
        return np.round(float(text),self.decimals())




class DoubleLineEdit(QtWidgets.QLineEdit):
    def __init__(self):
        super().__init__()
        self.validator = QRegExpValidator(QRegExp(r"[0-9]+[.][0-9]+"))

    def validate(self, input: str, pos: int):
        return self.validator.validate(input, pos)


class IntLineEdit(QtWidgets.QLineEdit):
    def __init__(self):
        super().__init__()
        self.validator = QRegExpValidator(QRegExp(r"[0-9]+"))

    def validate(self, input: str, pos: int):
        return self.validator.validate(input, pos)