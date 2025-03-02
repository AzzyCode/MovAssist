# __init__.py
from .configwindow import ConfigWindow, ConfigManager
from .trainerchatwindow import TrainerChatWindow 
from .exercisewindow import ExerciseWindow

# Export the classes you want available when importing the ui package
__all__ = ['ConfigWindow', 'ConfigManager','TrainerChatWindow', 'ExerciseWindow']