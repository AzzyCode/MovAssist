from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QMessageBox, QTabWidget, QScrollArea, QGroupBox, QSpinBox, QFormLayout
from PySide6.QtCore import Qt
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent.parent / "config"/ "exercise_config.json"

DEFAULT_CONFIG = {
    "squat": {
        "STATE_THRESH": {
            "up": 150,      # Angle above which the person is considered in "up" position
            "mid": 120,     # Angle below which the person is considered in "mid" position
            "depth": 90     # Angle threshold for proper squat depth
        },
        "ANGLE_THRESHOLDS": {
            "depth": 90,    # Minimum angle for proper depth
            "min_depth": 40,  # Minimum safe angle (don't go deeper than this)
            "hip_min": 60,  # Minimum hip angle (prevent excessive forward lean)
            "hip_max": 90,  # Maximum hip angle (prevent excessive upright position)
            "ankle_min": 80 # Minimum ankle angle (prevent knees going too far forward)
        },
        "FEEDBACK_MESSAGES": {
            "depth": "Squat not deep enough",
            "min_depth": "Squat too deep",
            "hip_min": "Maintain more upright position",
            "hip_max": "Bend forward slightly more",
            "ankle_min": "Knees going too far forward"
        }
    },
    "pushup": {
        "STATE_THRESH": {
            "up": 150,      # Angle above which the person is considered in "up" position
            "mid": 95       # Angle below which the person is considered in "mid" position
        },
        "ANGLE_THRESHOLDS": {
            "hip_low": 190, # Maximum hip angle (prevent sagging)
            "hip_high": 160, # Minimum hip angle (prevent pike position)
            "elbow_min": 50  # Minimum elbow angle (prevent going too deep)
        },
        "FEEDBACK_MESSAGES": {
            "hip_low": "Hip too low",
            "hip_high": "Hip too high",
            "elbow_min": "Too deep",
            "depth": "Not deep enough"
        }
    }
}


def load_config():
    """Load configuration from a JSON file with error handling."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            validated_config = DEFAULT_CONFIG.copy()
            for exercise in validated_config:
                if exercise in config:
                    for section in validated_config[exercise]:
                        if section in config[exercise]:
                            for key in validated_config[exercise][section]:
                                if key in config[exercise][section]:
                                    validated_config[exercise][section][key] = config[exercise][section][key]
            return validated_config     
        else:
            return DEFAULT_CONFIG
    except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Error loading config: {e}. Using defaults.")
        return DEFAULT_CONFIG
    

def save_config(config):
    """Save configuration to JSON file."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        return True
    except (IOError, TypeError) as e:
        print(f"Error saving config: {e}")
        return False
    
    
class ConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exercise Config")
        self.resize(500, 600)
        
        self.config = load_config()
        self.setup_ui()
        
        
    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout()
        
        self.tab_widget = QTabWidget()
        self.create_exercise_tabs()
        
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save Configuration")
        save_button.setMinimumHeight(40)
        save_button.clicked.connect(self.save_settings)
        
        reset_button = QPushButton("Reset Configuration")
        reset_button.setMinimumHeight(40)
        reset_button.clicked.connect(self.reset_to_defaults)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(reset_button)
        
        main_layout.addWidget(self.tab_widget)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        
    def create_exercise_tabs(self):
        """Create tabs for each execise type."""
        self.input_widgets = {}
        
        for exercise in self.config:
            exercise_tab = QWidget()
            
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            
            state_group = QGroupBox("State Tresholds")
            state_layout = QFormLayout()
            
            for key, value in self.config[exercise]["STATE_THRESH"].items():
                spin_box = QSpinBox()
                spin_box.setRange(0, 360)
                spin_box.setValue(value)
                spin_box.setSuffix("°")
                self.input_widgets[f"{exercise}_STATE_THRESH_{key}"] = spin_box
                state_layout.addRow(f"{key.capitalize()}: Angle", spin_box)              
                
            state_group.setLayout(state_layout)
            scroll_layout.addWidget(state_group)
            
            angle_group = QGroupBox("Angle Tresholds")
            angle_layout = QFormLayout()
            
            for key, value in self.config[exercise]["ANGLE_THRESHOLDS"].items():
                spin_box = QSpinBox()
                spin_box.setRange(0, 360)
                spin_box.setValue(value)
                spin_box.setSuffix("°")
                self.input_widgets[f"{exercise}_ANGLE_THRESHOLDS_{key}"] = spin_box
                angle_layout.addRow(f"{key.replace('_', ' ').capitalize()}:", spin_box)
                
            angle_group.setLayout(angle_layout)
            scroll_layout.addWidget(angle_group)
                
            feedback_group = QGroupBox("Feedback Messages")
            feedback_layout = QFormLayout()
            
            for key, value in self.config[exercise]["FEEDBACK_MESSAGES"].items():
                text_input = QLineEdit(value)
                self.input_widgets[f"{exercise}_FEEDBACK_MESSAGES_{key}"] = text_input
                feedback_layout.addRow(f"{key.replace('_', ' ').capitalize()}:", text_input)
            
            feedback_group.setLayout(feedback_layout)
            scroll_layout.addWidget(feedback_group)
            
            scroll.setWidget(scroll_content)
            tab_layout = QVBoxLayout()  # Create a layout instance
            exercise_tab.setLayout(tab_layout)  # Set the layout
            tab_layout.addWidget(scroll)  # Add widget to the layout
            self.tab_widget.addTab(exercise_tab, exercise.capitalize())
                        
    
    def save_settings(self):
        """Save the configuration settings."""
        try:
            for widget_id, widget in self.input_widgets.items():
                try:
                    # Split more carefully to handle keys with underscores
                    parts = widget_id.split('_', 1)  # Split only on first underscore
                    exercise = parts[0]
                    
                    # Split the remaining part to get section and key
                    remaining = parts[1]
                    if 'STATE_THRESH_' in remaining:
                        section = 'STATE_THRESH'
                        key = remaining.replace('STATE_THRESH_', '')
                    elif 'ANGLE_THRESHOLDS_' in remaining:
                        section = 'ANGLE_THRESHOLDS'
                        key = remaining.replace('ANGLE_THRESHOLDS_', '')
                    elif 'FEEDBACK_MESSAGES_' in remaining:
                        section = 'FEEDBACK_MESSAGES'
                        key = remaining.replace('FEEDBACK_MESSAGES_', '')
                    else:
                        print(f"Could not parse widget ID: {widget_id}")
                        continue
                    
                    print(f"Saving {exercise}.{section}.{key}")  # For debugging
                    
                    if isinstance(widget, QSpinBox):
                        self.config[exercise][section][key] = widget.value()
                    elif isinstance(widget, QLineEdit):
                        self.config[exercise][section][key] = widget.text()
                except Exception as e:
                    print(f"Error processing widget {widget_id}: {str(e)}")
                    raise
                    
            if save_config(self.config):
                config_manager = ConfigManager()
                config_manager.reload_config()
                
                QMessageBox.information(self, "Success", "Configuration saved successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to save configuration.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
                
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        confirm = QMessageBox.question(
            self,
            "Confirm Reset",
            "Are you sure you want to reset the configuration to default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )     
                
        if confirm == QMessageBox.Yes:
            self.config = DEFAULT_CONFIG.copy()
            current_tab = self.tab_widget.currentIndex()
            
            # Remove all tabs
            while self.tab_widget.count() > 0:
                self.tab_widget.removeTab(0)
            
            # Recreate tabs with default values
            self.create_exercise_tabs()
            
            if current_tab < self.tab_widget.count():
                self.tab_widget.setCurrentIndex(current_tab)
            
            save_config(self.config)
            QMessageBox.information(self, "Success", "Configuration reset to default values.")
                                
            
class ConfigManager:
    """Class to manage configuration for exercises."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = load_config()
        return cls._instance
    
    def get_exercise_config(self, exercise_name):
        """Get configuration for a specific exercise."""
        if exercise_name.lower() in self.config:
            return self.config[exercise_name.lower()]
        return None
    
    def reload_config(self):
        """Reload configuration from file."""
        self.config = load_config()
    
    def get_state_thresholds(self, exercise_name):
        """Get state thresholds for a specific exercise."""
        config = self.get_exercise_config(exercise_name)
        if config and "STATE_THRESH" in config:
            return config["STATE_THRESH"]
        return {}
    
    def get_angle_thresholds(self, exercise_name):
        """Get angle thresholds for a specific exercise."""
        config = self.get_exercise_config(exercise_name)
        if config and "ANGLE_THRESHOLDS" in config:
            return config["ANGLE_THRESHOLDS"]
        return {}
    
    def get_feedback_message(self, exercise_name, error_key):
        """Get feedback message for a specific error."""
        config = self.get_exercise_config(exercise_name)
        if config and "FEEDBACK_MESSAGES" in config and error_key in config["FEEDBACK_MESSAGES"]:
            return config["FEEDBACK_MESSAGES"][error_key]
        return f"Issue with {error_key.replace('_', ' ')}"

    
    