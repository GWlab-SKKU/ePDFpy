from PyQt5 import QtCore, QtWidgets, QtGui

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
