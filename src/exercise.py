import cv2
import mediapipe as mp
import numpy as np
import time
import logging

from utils import calculate_angle, display_counter, display_feedback
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseExercise:
    """Base class for exercise pose estimation"""
    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=min_detecting_confidentce, 
            min_tracking_confidence=min_tracking_confidence)
        
        self.state = "up"
        self.last_state = "up"
        self.is_going_up = True
        
        self.rep_completed = False
        self.rep_count = 0
        self.rep_errors = set()
        
        self.feedback_timestamp = 0
        self.feedback_duration = 3
        self.set_feedback = False
        
        self.body_direction = ""
      
      
    def get_coordinates(self, landmarks, mp_pose, side, joint):
        """Retrives x and y coordinates of a particular keypoint from the pose estimation model"""
        coord = getattr(mp_pose.PoseLandmark, side.upper() + "_" + joint.upper())
        x_coord_val = landmarks[coord].x
        y_coord_val = landmarks[coord].y
        
        return [x_coord_val, y_coord_val]
              
    def draw_landmarks(self, image, pose_landmarks):
        """Draw pose landmarks on the image"""
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=0, circle_radius=0), 
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
    def determine_body_direction(self, landmarks):
        """Determine body direction (Left or Right) based on z-coordinates"""
        try:
            left_shoulder_z = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z
            right_shoulder_z = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z
            
            if left_shoulder_z < right_shoulder_z:
                self.body_direction = "Left"
            else: 
                self.body_direction = "Right"
        except KeyError as e:
            logging.error(f"KeyError determinig body directions {e}")
            self.body_direction = "Left"
                    
            
class Squat(BaseExercise):
    """Class to process and analyze squats"""
    STATE_THRESH = {
        "up": 150,
        "mid": 120,
        "depth": 90,
    }
    
    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        super().__init__( min_detecting_confidentce, min_tracking_confidence)
        
        self.min_knee_angle = 180
        self.lowest_knee_angle = 180
        
    def process_frame(self, frame):
        """Process a video frame to analyze squat exercise."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            self.determine_body_direction(landmarks)
            
            hip_angle, knee_angle, ankle_angle = self.get_calculated_angles(landmarks)
            self.update_squat_state(knee_angle)
                            
            if self.state == "mid":
                has_issues = self.check_squat_form(hip_angle, knee_angle, ankle_angle)
                if has_issues:
                    self.set_feedback = True
                    self.feedback_timestamp = time.time()                   
                    
            if self.rep_completed and self.last_state == "mid":
                self.rep_count += 1
                self.rep_completed = False
                self.rep_errors.clear()
                
            self.last_state = self.state
            self.display_angles(image, hip_angle, knee_angle, ankle_angle, landmarks)
        
        self.draw_landmarks(image, result.pose_landmarks)       
        display_counter(image, self.rep_count, "Squat")
        
        if self.set_feedback and (time.time() - self.feedback_timestamp < self.feedback_duration):
            display_feedback(image, self.rep_errors)
        else:
            self.set_feedback = False
        
        return image

    
    def get_calculated_angles(self, landmarks):
        """Calculate hip, knee and ankle angles based on landmarks"""
        mp_pose = mp.solutions.pose
        side = self.body_direction
        
        shoulder = self.get_coordinates(landmarks, mp_pose, side, "shoulder")
        hip = self.get_coordinates(landmarks, mp_pose, side, "hip")
        knee = self.get_coordinates(landmarks, mp_pose, side, "knee")
        ankle = self.get_coordinates(landmarks, mp_pose, side, "ankle")
        feet = self.get_coordinates(landmarks, mp_pose, side, "foot_index")
        
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)
        ankle_angle = calculate_angle(knee, ankle, feet)
        
        
        return hip_angle, knee_angle, ankle_angle
    
    
    def check_squat_form(self, hip_angle, knee_angle, ankle_angle):
        """Check squat form for errors with improved tracking and feedback."""
        has_issues = False
            
        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle
            self.is_going_up = False
        elif knee_angle > self.min_knee_angle:
            self.is_going_up = True
            
        
        if self.state == "mid" and knee_angle < self.STATE_THRESH["depth"]:
            self.rep_errors.clear()
            
            # Check 1: Hip Position (back angle)
            if hip_angle < 45: # Leaning too far forward
                self.rep_errors.add("Maitain more upright position")
                has_issues = True
            elif hip_angle > 120: # Too upright
                self.rep_errors.add("Bend forward slightly more")
                has_issues = True
                
            # Check 2: Knee Position
            if ankle_angle < 60: # Knees too forward
                self.rep_errors.add("Knees going too forward")
                has_issues = True
                
        elif self.state == "mid" and self.is_going_up:
            if self.min_knee_angle > self.STATE_THRESH["depth"]:
                self.rep_errors.add("Squat not deep enough")
                has_issues = True
                
        return has_issues
                
            
    def update_squat_state(self, knee_angle):
        """Update squat state based on knee angle"""
        # Going down
        if self.state == "up" and knee_angle < self.STATE_THRESH['mid']:
            self.state = "mid"
        #Going up
        elif self.state == "mid" and knee_angle > self.STATE_THRESH['up']:
            self.state = "up"
            if self.min_knee_angle < self.STATE_THRESH["depth"]:
                self.rep_completed = True # Full squat has been completed
            self.min_knee_angle = 180
    
    def display_angles(self, image, hip_angle, knee_angle, ankle_angle, landmarks):
        """Display angles on the image."""
        pass
        side = self.body_direction.lower()
        mp_pose = mp.solutions.pose
        
        hip_coords = _normalized_to_pixel_coordinates(
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP")].x, 
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP")].y,
            image.shape[1], image.shape[0]
        ) 
        knee_coords = _normalized_to_pixel_coordinates(
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_KNEE")].x,
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_KNEE")].y,
            image.shape[1], image.shape[0]
        )  
        ankle_coords = _normalized_to_pixel_coordinates(
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_ANKLE")].x,
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_ANKLE")].y,
            image.shape[1], image.shape[0]
        )
        
        cv2.putText(image, str(int(hip_angle)), hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(knee_angle)), knee_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(ankle_angle)), ankle_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

       
class Pushup(BaseExercise):
    """Class to process and analyze pushup exercise."""
    STATE_THRESH = {
        "up": 150,
        "mid": 95,
    }
    
    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        super().__init__(min_detecting_confidentce, min_tracking_confidence)
        
        self.min_elbow_angle = 180
    
    
    def process_frame(self, frame):
        """Process a video frame to analyze pushup execise."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        frame_height, frame_width, _ = image.shape
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            self.determine_body_direction(landmarks)

            elbow_angle, hip_angle = self.get_calculated_angles(landmarks)
            self.update_pushup_state(elbow_angle)
                            
            if self.state == "mid":
                has_issues = self.check_pushup_form(elbow_angle, hip_angle)
                if has_issues:
                    self.set_feedback = True
                    self.feedback_timestamp = time.time()                   
                    
            if self.rep_completed and self.last_state == "mid":
                self.rep_count += 1
                self.rep_completed = False
                self.rep_errors.clear()
                
            self.last_state = self.state
            self.display_angles(image, hip_angle, elbow_angle, landmarks)
        
        self.draw_landmarks(image, result.pose_landmarks)       
        display_counter(image, self.rep_count, "Pushup")
        
        if self.set_feedback and (time.time() - self.feedback_timestamp < self.feedback_duration):
            display_feedback(image, self.rep_errors)
        else:
            self.set_feedback = False
        
        return image 

    def check_pushup_form(self, elbow_angle, hip_angle):
        """Check pushup form for erros"""
        has_issues = False
        
        if elbow_angle < self.min_elbow_angle:
            self.min_elbow_angle = elbow_angle
            self.is_going_up = False
        elif elbow_angle > self.min_elbow_angle:
            self.is_going_up = True

        
        if self.state == "mid" and elbow_angle < 90:    
            if elbow_angle < 60:
                self.rep_errors.add("Too deep")
                has_issues = True
            
            if hip_angle < 160:
                self.rep_errors.add("Hip too low")
                has_issues = True
            elif hip_angle > 200:
                self.rep_errors.add("Hip too high") 
                has_issues = True     
                
        elif self.state == "mid" and self.is_going_up:
            if self.min_elbow_angle > 90:
                self.rep_errors.add("Not deep enough")
                has_issues = True       
        
        return has_issues    
        

    def update_pushup_state(self, elbow_angle):
        """Update pushup state based on elbow angle."""
        # Going down
        if self.state == "up" and elbow_angle < self.STATE_THRESH["mid"]:
            self.state = "mid"
        # Going up
        elif self.state == "mid" and elbow_angle > self.STATE_THRESH["up"]:
            self.state = "up"
            self.min_elbow_angle = 180
            self.rep_completed = True 


    def get_calculated_angles(self, landmarks):
        """Calculate elbow and hip angles based on landmarks."""
        mp_pose = mp.solutions.pose
        side = self.body_direction
    
        shoulder = self.get_coordinates(landmarks, mp_pose, side, "shoulder")
        elbow = self.get_coordinates(landmarks, mp_pose, side, "elbow")
        wrist = self.get_coordinates(landmarks, mp_pose, side, "wrist")
        hip = self.get_coordinates(landmarks, mp_pose, side, "hip")
        knee = self.get_coordinates(landmarks, mp_pose, side, "knee")

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        hip_angle = calculate_angle(shoulder, hip, knee)
        return elbow_angle, hip_angle
    
    
    def display_angles(self, image, hip_angle, elbow_angle, landmarks):
        """Display angles on the image."""
        side = self.body_direction.lower()
        mp_pose = mp.solutions.pose
        
        elbow_coords = _normalized_to_pixel_coordinates(
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_ELBOW")].x,
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_ELBOW")].y,
            image.shape[1], image.shape[0]
        )    
        hip_coords = _normalized_to_pixel_coordinates(
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP")].x, 
            landmarks[getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP")].y,
            image.shape[1], image.shape[0]
        )
        
        cv2.putText(image, str(int(elbow_angle)), elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(hip_angle)), hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        
