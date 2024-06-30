import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QFont

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set window title
        self.setWindowTitle("Enhanced PyQt5 GUI with Input")
        self.setGeometry(100, 100, 400, 200)  # Set the initial size and position
        
        # Create a font object
        font = QFont("Arial", 14)
        
        # Create a label
        self.label = QLabel("Enter text in the boxes and click the button", self)
        self.label.setFont(font)
        
        # Create input boxes
        self.input1 = QLineEdit(self)
        self.input1.setFont(font)
        self.input1.setPlaceholderText("Input 1")
        
        self.input2 = QLineEdit(self)
        self.input2.setFont(font)
        self.input2.setPlaceholderText("Input 2")
        
        # Create a button
        self.button = QPushButton("Click Me", self)
        self.button.setFont(font)
        self.button.clicked.connect(self.on_button_click)
        
        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input1)
        input_layout.addWidget(self.input2)
        
        # Add some space between the input fields and the button
        layout.addLayout(input_layout)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        layout.addWidget(self.button)
        
        # Add padding and margins
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        self.setLayout(layout)
        
    def on_button_click(self):
        # Get text from input boxes
        text1 = self.input1.text()
        text2 = self.input2.text()
        
        # Change the label text
        self.label.setText(f"Input 1: {text1}, Input 2: {text2}")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
