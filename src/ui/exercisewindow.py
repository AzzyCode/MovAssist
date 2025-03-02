from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout,QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox, QMessageBox
from PySide6.QtCore import Qt

import threading
import os
from load_dotenv import load_dotenv

from src.core import Squat, Pushup
from src.core import VideoProcessor
from src.utils import TrainerChat

from .trainerchatwindow import TrainerChatWindow
from .configwindow import ConfigWindow        
        
load_dotenv()

class ExerciseWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("MovAssist")
        self.resize(400, 300)
        
        self.trainer = TrainerChat(os.getenv("API_KEY"))
        
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
        
        self.config_button = QPushButton("Config")
        self.config_button.setFixedSize(file_button.sizeHint())
        self.config_button.clicked.connect(self.open_config)
        layout.addWidget(self.config_button)
        
        self.chat_button = QPushButton("Chat with Trainer")
        self.chat_button.clicked.connect(self.open_chat_window)
        layout.addWidget(self.chat_button)
        
        # Checkboxs
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(20)
        checkbox_layout.setContentsMargins(0, 10, 0, 10)
        
        self.use_camera_checkbox = QCheckBox("Use Camera")
        self.use_camera_checkbox.setChecked(False)
        
        self.raport_checkbox = QCheckBox("Raport")
        self.raport_checkbox.setChecked(False)
        
        self.replay_checkbox = QCheckBox("Replay")
        self.replay_checkbox.setChecked(False)
        
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.use_camera_checkbox)
        checkbox_layout.addWidget(self.raport_checkbox)
        checkbox_layout.addWidget(self.replay_checkbox)
        checkbox_layout.addStretch()
        
        layout.addLayout(checkbox_layout)
        
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
        
     
    def open_chat_window(self):
        app_data = None
        
        self.chat_window = TrainerChatWindow(self.trainer, app_data)
        self.chat_window.show()
    
    def run_pushup(self):
        """Initialize and run the pushup exercise."""
        video_source = self.get_video_source()
        use_camera = self.use_camera_checkbox.isChecked()
        if video_source is not None:
            pushup = Pushup(use_camera=use_camera)
            pushup_instance = pushup if self.raport_checkbox.isChecked() else None
            processor = VideoProcessor(video_source, pushup.process_frame, resize_dim=(1024, 768), pushup_instance=pushup_instance, save_replay=self.replay_checkbox.isChecked())
            threading.Thread(target=self.run_processor, args=(processor,)).start()
        else:
            self.set_status_message("Please select a valid video source.")
        

    def run_squat(self):    
        """Initialize and run the squat exercise."""
        video_source = self.get_video_source()
        use_camera = self.use_camera_checkbox.isChecked()
        if video_source is not None:
            squat = Squat(use_camera=use_camera)
            squat_instance = squat if self.raport_checkbox.isChecked() else None
            processor = VideoProcessor(video_source, squat.process_frame, resize_dim=(1024, 768), squat_instance=squat_instance, save_replay=self.replay_checkbox.isChecked())
            threading.Thread(target=self.run_processor, args=(processor,)).start()
        else:
            self.set_status_message("Please select a valid video source.")
        
             
    def run_processor(self, processor):
        """Run video processor"""
        try:
            processor.start()
        except Exception as e:
            print(f"Error during processing: {e}")
            
    def browse_file(self):
        """Open file dialog for video selection."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.gif, *.png *.jpg)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.file_input.setText(file_path)
            
            
    def open_config(self):
        """Open the configuration window."""
        self.config_window = ConfigWindow()
        self.config_window.show()
    
    
    def get_video_source(self):
        """Determine video source based on user selection."""
        if self.use_camera_checkbox.isChecked():
            return 0
        elif self.file_input.text().strip():
            return self.file_input.text()
        else:
            return None
        
        
