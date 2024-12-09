import numpy as np
import matplotlib as mp
import math
import cv2

from typing import Tuple, List, Optional, Union

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate the angle between three points."""
    
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def angle_of_singleline(a, b):
    x_difference = b[0] - a[0]
    y_difference = b[1] - a[1]
    
    return math.degrees(math.atan2(y_difference, x_difference))


def rescale_frame(frame, scale=None, target_dim=None):
    """Rescale a frame either by a scale factor or to target dimension."""
    if frame is None:
        return None
    
    try:
        if scale is not None:
            if scale <= 0:
                raise ValueError("Scale must be positive.")
            width = int(frame.shape[1] * scale)
            height = int(frame.shape[0] * scale)
            new_dim = (width, height)
        elif target_dim is not None:
            if any(dim <= 0 for dim in target_dim):
                raise ValueError("Target dimensions must be positive.")
            new_dim = target_dim
        else:
            return frame
        
        return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
    
    except Exception as e:
        print(f"Error in rescale_frame: {str(e)}")
        return None
    
    
def preprocess_frame(frame, target_height=480, target_width=640):
    """Preprocess video frame by resizing and cropping"""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_aspect = target_width / target_height
    
    if aspect_ratio > target_aspect:
        # Image is wider than target
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Crop width to target
        start_x = (new_width - target_width) // 2
        frame = frame[:, start_x:start_x+target_width]

    else:
        # Image is taller that target
        new_height = int(target_width / aspect_ratio)
        new_width = target_width
        frame = cv2.resize(frame, (new_width, new_height))
        
    return frame
             
def display_counter(frame, rep_counter, exercise) -> None:
    """Display exercise counter on frame"""
    text = f"{exercise} reps: {rep_counter}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale =1
    font_thickness = 2
    color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = 50
    
    cv2.putText(frame, text, (text_x + 2, text_y + 2), font, font_scale, shadow_color, font_thickness)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, font_thickness)
    

def display_feedback(image, issues):
    """Display feedback messages in a box in the top left corner of the frame"""
    if not issues:
        return
    
    # Configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (255, 255, 255)  # White text
    box_color = (0, 0, 0)  # Black box
    box_alpha = 0.7  # Box transparency
    line_spacing = 25
    padding = 10
    
    # Fixed position in top left corner
    start_x = 20
    start_y = 20
    
    # Calculate box dimensions
    max_text_width = 0
    total_height = padding * 2  # Initial padding
    
    # Get dimensions for all error messages
    text_dimensions = []
    for issue in issues:
        (text_width, text_height), _ = cv2.getTextSize(issue, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_width)
        text_dimensions.append((text_width, text_height))
        total_height += text_height + line_spacing
    
    # Create box coordinates
    box_width = max_text_width + (padding * 2)
    box_height = total_height - line_spacing  # Remove extra line spacing from last item
    
    # Create semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (start_x, start_y),
        (start_x + box_width, start_y + box_height),
        box_color,
        -1
    )
    
    # Apply transparency
    cv2.addWeighted(overlay, box_alpha, image, 1 - box_alpha, 0, image)
    
    # Add error messages
    current_y = start_y + padding + text_dimensions[0][1]  # Start after top padding
    for i, issue in enumerate(issues):
        cv2.putText(
            image,
            issue,
            (start_x + padding, current_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )
        current_y += text_dimensions[i][1] + line_spacing


