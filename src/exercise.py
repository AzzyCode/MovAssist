import cv2
import mediapipe as mp
import numpy as np
import time
import logging

from utils import calculate_angle, display_counter, display_feedback
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseExercise:
    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
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
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=0, circle_radius=0), 
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
                    
            
class Squat(BaseExercise):
    STATE_THRESH = {
        "up": 150,
        "mid": 100,
    }
    
    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        super().__init__( min_detecting_confidentce, min_tracking_confidence)
        
        self.min_knee_angle = 180
         
        
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        result = self.pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #image = rescale_frame(image, 30)
        frame_height, frame_width, _ = image.shape
        
        if result.pose_landmarks:
        
            landmarks = result.pose_landmarks.landmark

            self.get_body_direction(landmarks)
            
            hip_angle, knee_angle, knee_foot = self.get_calculated_angles(landmarks)
            self.update_squat_state(knee_angle)
                            
            if self.state == "mid":
                has_issues = self.check_squat(hip_angle, knee_angle, knee_foot)
                if has_issues:
                    self.set_feedback = True
                    self.feedback_timestamp = time.time()                   
                    
            if self.rep_completed and self.last_state == "mid":
                self.rep_count += 1
                self.rep_completed = False
                self.rep_errors.clear()
                
            self.last_state = self.state
            
            self.display_angles(image, frame_width, frame_height, hip_angle, knee_angle, landmarks)
        
        self.draw_landmarks(image, result.pose_landmarks)       
        display_counter(image, self.rep_count, "Squat")
        
        if self.set_feedback and (time.time() - self.feedback_timestamp < self.feedback_duration):
            display_feedback(image, self.rep_errors)
        else:
            self.set_feedback = False
        
        return image

    
    def get_calculated_angles(self, landmarks):
        mp_pose = mp.solutions.pose
        
        if self.body_direction == "Left":
            left_shoulder = self.get_coordinates(landmarks, mp_pose, "left", "shoulder")
            left_hip =  self.get_coordinates(landmarks, mp_pose, "left", "hip")
            left_knee = self.get_coordinates(landmarks, mp_pose, "left", "knee")
            #left_foot = self.get_coordinates(landmarks, mp_pose, "left", "foot_index")
            left_ankle = self.get_coordinates(landmarks, mp_pose, "left", "ankle")
            
            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            
            knee_foot_x = 0
            #knee_foot_x = abs(left_knee[0] - left_foot[0])
            
        elif self.body_direction == "Right":    
            right_shoulder = self.get_coordinates(landmarks, mp_pose, "right", "shoulder")
            right_hip = self.get_coordinates(landmarks, mp_pose, "right", "hip")
            right_knee = self.get_coordinates(landmarks, mp_pose, "right", "knee")
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
            right_ankle = self.get_coordinates(landmarks, mp_pose, "right", "ankle")
        
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
            knee_foot_x = 0
            #knee_foot_x = abs(right_knee[0] - right_foot[0])
            
        #hip_angle = (left_hip_angle + right_hip_angle) / 2
        #knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        
        return hip_angle, knee_angle, knee_foot_x
    
    
    def check_squat(self, hip_angle, knee_angle, knee_foot):
        has_issues = False
        
        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle
            self.is_going_up = False
        elif knee_angle > self.min_knee_angle:
            self.is_going_up = True
        
        if self.state == "mid" and not self.is_going_up:
            if not (60 < hip_angle < 120 and knee_angle < 90 and knee_foot <= 0.1):
                has_issues = True
                
                # Condition 1 (Hips)
                if hip_angle < 60:
                    self.rep_errors.add("Hips are too low")
                elif hip_angle > 120:
                    self.rep_errors.add("Hips are too high")

                # Condition 2 (Knees)
                if knee_angle > 100:
                    self.rep_errors.add("Thighs should be parallel with the floor.")

                # Condition 3 (Feet)
                if knee_foot > 0.1:
                    self.rep_errors.add("Knee should not extend beyond toe tips.")
                    
            else:
                self.rep_errors.clear()
                
        return has_issues
    
    
    def update_squat_state(self, knee_angle):
        if self.state == "up" and knee_angle < self.STATE_THRESH['mid']:
            self.state = "mid"
        elif self.state == "mid" and knee_angle > self.STATE_THRESH['up']:
            self.state = "up"
            self.min_knee_angle = 180
            self.rep_completed = True # Full squat has been completed
            
    
    def display_angles(self, image, frame_width, frame_height, hip_angle, knee_angle, landmarks):
        if self.body_direction == "Left":
            left_hip_coordinates = _normalized_to_pixel_coordinates(
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, 
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y,
                frame_width, frame_height)
        
            left_knee_coordinates = _normalized_to_pixel_coordinates(
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x, 
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y,
                frame_width, frame_height)    
           
            cv2.putText(image, str(int(hip_angle)), left_hip_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(knee_angle)), left_knee_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
           
        elif self.body_direction == "Right":
            right_hip_coordinates = _normalized_to_pixel_coordinates(
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x, 
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y,
                frame_width, frame_height)
        
            right_knee_coordinates = _normalized_to_pixel_coordinates(
                    landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x, 
                    landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y,
                    frame_width, frame_height)
                
            cv2.putText(image, str(int(hip_angle)), right_hip_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(knee_angle)), right_knee_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
           

    def get_body_direction(self, landmarks):
        try:
            left_knee_z = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].z
            right_knee_z = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].z
            
            if left_knee_z < right_knee_z:
                self.body_direction = "Left"
            elif left_knee_z > right_knee_z:
                self.body_direction = "Right"
            else:
                logging.warning("Could not detect body direction. Defaulting to 'LEFT'.")
                self.body_direction = "Left"
        except KeyError as e:
            logging.error(f"KeyError: {e}. Unable to determine body direction.")
            self.body_direction = "Left"
            
    
    
class Pushup(BaseExercise):
    STATE_THRESH = {
        "up": 150,
        "mid": 95,
    }
    
    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        super().__init__(min_detecting_confidentce, min_tracking_confidence)
        
        self.min_elbow_angle = 180
    
    
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        result = self.pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        frame_height, frame_width, _ = image.shape
        
        if result.pose_landmarks:
            
            landmarks = result.pose_landmarks.landmark
            
            self.get_body_direction(landmarks)

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
            
            self.display_angles(image, frame_width, frame_height, hip_angle, elbow_angle, landmarks)
        
        self.draw_landmarks(image, result.pose_landmarks)       
        display_counter(image, self.rep_count, "Pushup")
        
        if self.set_feedback and (time.time() - self.feedback_timestamp < self.feedback_duration):
            display_feedback(image, self.rep_errors)
        else:
            self.set_feedback = False
        
        return image 

    def check_pushup_form(self, elbow_angle, hip_angle):
        has_issues = False
        
        if elbow_angle < self.min_elbow_angle:
            self.min_elbow_angle = elbow_angle
            self.is_going_up = False
        elif elbow_angle > self.min_elbow_angle:
            self.is_going_up = True

        if self.state == "mid" and not self.is_going_up:
            if not (60 < hip_angle < 120 and elbow_angle < 90):
                has_issues = True
                
                if elbow_angle < 60:
                    self.rep_errors.add("Too deep")
                elif elbow_angle > 100:
                    self.rep_errors.add("Too shallow")
                
                if hip_angle < 160:
                    self.rep_errors.add("Hip too low")
                elif hip_angle > 200:
                    self.rep_errors.add("Hip too high")             
            else:
                self.rep_errors.clear()
            
        return has_issues    
        

    def update_pushup_state(self, elbow_angle):
        if self.state == "up" and elbow_angle < self.STATE_THRESH["mid"]:
            self.state = "mid"
        elif self.state == "mid" and elbow_angle > self.STATE_THRESH["up"]:
            self.state = "up"
            self.min_elbow_angle = 180
            self.rep_completed = True 


    def get_calculated_angles(self, landmarks):
        mp_pose = mp.solutions.pose
        
        if self.body_direction == "Left":
            left_shoulder = self.get_coordinates(landmarks, mp_pose, "left", "shoulder")
            left_elbow = self.get_coordinates(landmarks, mp_pose, "left", "elbow")
            left_wrist = self.get_coordinates(landmarks, mp_pose, "left", "wrist")
            left_hip = self.get_coordinates(landmarks, mp_pose, "left", "hip")
            left_knee = self.get_coordinates(landmarks, mp_pose, "left", "knee")

            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        
        elif self.body_direction == "Right":    
            right_shoulder = self.get_coordinates(landmarks, mp_pose, "right", "shoulder")
            right_elbow = self.get_coordinates(landmarks, mp_pose, "right", "elbow")
            right_wrist = self.get_coordinates(landmarks, mp_pose, "right", "wrist")
            right_hip = self.get_coordinates(landmarks, mp_pose, "right", "hip")
            right_knee = self.get_coordinates(landmarks, mp_pose, "right", "knee")
            
            elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
        return elbow_angle, hip_angle
    
    
    def display_angles(self, image, frame_width, frame_height, hip_angle, elbow_angle, landmarks):
        if self.body_direction == "Left":
            left_elbow_coordinates = _normalized_to_pixel_coordinates(
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x, 
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y,
                frame_width, frame_height)
            
            left_hip_coordinates = _normalized_to_pixel_coordinates(
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, 
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y,
                frame_width, frame_height)
        
            cv2.putText(image, str(int(hip_angle)), left_hip_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(elbow_angle)), left_elbow_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
        
        elif self.body_direction == "Right":        
            right_elbow_coordinates = _normalized_to_pixel_coordinates(
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x, 
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y,
                frame_width, frame_height)
            
            right_hip_coordinates = _normalized_to_pixel_coordinates(
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x, 
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y,
                frame_width, frame_height)
            
            cv2.putText(image, str(int(hip_angle)), right_hip_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(elbow_angle)), right_elbow_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
        
    def get_body_direction(self, landmarks):
        left_shoulder_z = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z
        right_shoulder_z = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z
        
        if left_shoulder_z < right_shoulder_z:
            self.body_direction = "Left"
        elif left_shoulder_z > right_shoulder_z:
            self.body_direction = "Right"
        else:
            raise ValueError("Error: Unable to determine body direction.")