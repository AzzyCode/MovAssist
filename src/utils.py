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
    """Display feedback messages on frame"""
    if not issues:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)
    shadow_color = (50, 50, 50)   
    bg_color = (0, 0, 255, 150)
    alpha = 0.8         
    line_spacing = 15
    padding = 10
                    
    x = 20                          
    y = 40                         
    
    overlay = image.copy()
    
    for issue in enumerate(issues):
        issue = issue[1]
        
        (text_width, text_height), _ = cv2.getTextSize(issue, font, font_scale, font_thickness)
        
        bg_top_left = (x - padding, y - text_height - padding)
        bg_bottom_right = (x + text_width + padding, y + padding)
        
        cv2.rectangle(overlay, bg_top_left, bg_bottom_right, bg_color[:3], -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        shadow_offset = 2
        cv2.putText(image, issue, (x + shadow_offset, y + shadow_offset), font, font_scale, shadow_color, font_thickness)            
        cv2.putText(image, issue, (x, y), font, font_scale, text_color, font_thickness)
        
        y += text_height + line_spacing


