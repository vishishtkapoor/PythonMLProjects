import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('bmd.csv')

# Encode the 'fracture' column
le = LabelEncoder()
df['fracture'] = le.fit_transform(df['fracture'])

# Features and target variable
X = df[['age', 'weight_kg', 'height_cm', 'waiting_time', 'bmd']]
y = df['fracture']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

class FracturePredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Hip Fracture Prediction')
        self.setGeometry(100, 100, 400, 400)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        self.layout = QVBoxLayout()

        self.font = QFont('Arial', 12)

        self.age_label = QLabel('Age:')
        self.age_label.setFont(self.font)
        self.layout.addWidget(self.age_label)
        self.age_input = QLineEdit()
        self.age_input.setFont(self.font)
        self.age_input.setStyleSheet("background-color: #3c3f41; color: #ffffff; border: 1px solid #5c5c5c;")
        self.layout.addWidget(self.age_input)

        self.weight_label = QLabel('Weight (kg):')
        self.weight_label.setFont(self.font)
        self.layout.addWidget(self.weight_label)
        self.weight_input = QLineEdit()
        self.weight_input.setFont(self.font)
        self.weight_input.setStyleSheet("background-color: #3c3f41; color: #ffffff; border: 1px solid #5c5c5c;")
        self.layout.addWidget(self.weight_input)

        self.height_label = QLabel('Height (cm):')
        self.height_label.setFont(self.font)
        self.layout.addWidget(self.height_label)
        self.height_input = QLineEdit()
        self.height_input.setFont(self.font)
        self.height_input.setStyleSheet("background-color: #3c3f41; color: #ffffff; border: 1px solid #5c5c5c;")
        self.layout.addWidget(self.height_input)

        self.waiting_time_label = QLabel('Waiting Time (minutes):')
        self.waiting_time_label.setFont(self.font)
        self.layout.addWidget(self.waiting_time_label)
        self.waiting_time_input = QLineEdit()
        self.waiting_time_input.setFont(self.font)
        self.waiting_time_input.setStyleSheet("background-color: #3c3f41; color: #ffffff; border: 1px solid #5c5c5c;")
        self.layout.addWidget(self.waiting_time_input)

        self.bmd_label = QLabel('Bone Mineral Density:')
        self.bmd_label.setFont(self.font)
        self.layout.addWidget(self.bmd_label)
        self.bmd_input = QLineEdit()
        self.bmd_input.setFont(self.font)
        self.bmd_input.setStyleSheet("background-color: #3c3f41; color: #ffffff; border: 1px solid #5c5c5c;")
        self.layout.addWidget(self.bmd_input)

        self.predict_button = QPushButton('Predict')
        self.predict_button.setFont(QFont('Arial', 14, QFont.Bold))
        self.predict_button.setStyleSheet("background-color: #4a90e2; color: #ffffff; border: none; padding: 10px;")
        self.predict_button.clicked.connect(self.predict)
        self.layout.addWidget(self.predict_button)

        self.setLayout(self.layout)

    def predict(self):
        try:
            age = int(self.age_input.text())
            weight = float(self.weight_input.text())
            height = float(self.height_input.text())
            waiting_time = float(self.waiting_time_input.text())
            bmd = float(self.bmd_input.text())

            input_data = [[age, weight, height, waiting_time, bmd]]
            prediction = clf.predict(input_data)
            result = 'Fracture' if prediction[0] == 1 else 'No Fracture'

            QMessageBox.information(self, 'Prediction', f'The prediction is: {result}')
        except ValueError as e:
            QMessageBox.warning(self, 'Input Error', str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FracturePredictionApp()
    window.show()
    sys.exit(app.exec_())
