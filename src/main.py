from PySide6.QtWidgets import QApplication
import sys
from src.ui import ExerciseWindow

def main():
    app = QApplication(sys.argv)
    window = ExerciseWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    
    