from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox
from PySide6.QtCore import Qt

from exercise import Squat, Pushup
from video_processing import VideoProcessor

import threading
import sys

class ExerciseWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("AI Trainer")
        self.resize(400, 200)
        
        layout = QVBoxLayout()

        file_label = QLabel("Select a file:")
        file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(file_label)
        
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        file_button = QPushButton("Browse")
        file_button.clicked.connect(self.browse_file)
        
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(file_button)
        layout.addLayout(file_layout)
        
        self.use_camera_checkbox = QCheckBox("Use Camera")
        self.use_camera_checkbox.setChecked(False)
        layout.addWidget(self.use_camera_checkbox)
        
        exercise_label = QLabel("Select an exercise:")
        exercise_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(exercise_label)    
        
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.pushup_button = QPushButton("Pushup")
        self.squat_button = QPushButton("Squat")
        
        self.pushup_button.clicked.connect(self.run_pushup)
        self.squat_button.clicked.connect(self.run_squat)
        
        self.pushup_button.setFixedSize(200, 60)
        self.squat_button.setFixedSize(200, 60)
        
        button_layout.addWidget(self.pushup_button)
        button_layout.addWidget(self.squat_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    
    def browse_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.gif)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.file_input.setText(file_path)
    
    
    def get_video_source(self):
        if self.use_camera_checkbox.isChecked():
            return 0
        elif self.file_input.text().strip():
            return self.file_input.text()
        else:
            return None
    
    
    def run_pushup(self):
        video_source = self.get_video_source()
        if video_source or video_source == 0:
            pushup = Pushup(video_source)
            pushup.analyze_video()
        else:
            print("No file selected.")
        
    
    def run_squat(self):
        video_source = self.get_video_source()
        if video_source is not None:
            squat = Squat()
            processor = VideoProcessor(video_source, squat.process_frame, resize_dim=(800, 600))
            threading.Thread(target=self.run_processor, args=(processor,)).start()
        else:
            self.set_status_message("Please select a valid video source.")
        
        
    def run_processor(self, processor):
        try:
            processor.start()
        except Exception as e:
            print(f"Error during processing: {e}")


if __name__ == "__main__":
    app = QApplication([])
    window = ExerciseWindow()
    window.show()
    sys.exit(app.exec())

