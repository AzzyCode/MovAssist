import cv2
import mediapipe as mp
import numpy as np
import time
import os
import imutils

from imutils.video import FileVideoStream, WebcamVideoStream
from imutils.video import FPS

from utils import calculate_angle, display_counter, display_feedback
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


class BaseExercise:
    def __init__(self, video_path, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Error: Video file {video_path} not found")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=min_detecting_confidentce, min_tracking_confidence=min_tracking_confidence)
        
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
        
        
    def analyze_video(self):
        fvs = FileVideoStream(self.video_path).start()
        time.sleep(1.0)
        fps = FPS().start()
        
        try:
            while fvs.more():
                frame = fvs.read()
                if frame is None:
                    break
                
                frame = imutils.resize(frame, width = 800)
                image = self.process_frame(frame)
                
                cv2.imshow("Frame", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
                fps.update()
        finally:
            fps.stop()
            fvs.stop()
            cv2.destroyAllWindows()
            
            
    def analyze_webcam(self, num_frames=60, display=True):
        stream = WebcamVideoStream(src=0).start()
        fps = FPS().start()
        
        try:
            while fps._numFrames < num_frames:
                frame = stream.read()
                if frame is None:
                    break
                
                frame = imutils.resize(frame, width=800)
                image = self.process_frame(frame)
                
                if display:
                    cv2.imshow("Frame", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    
                fps.update() 
        finally:
            fps.stop()
            stream.stop()
            cv2.destroyAllWindows()      
                
    
    def get_direction(self, landmarks):
        mp_pose = mp.solutions.pose
        
        nose = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
        
        if nose[0] > 0.5:
            self.body_direction = "Right"
        elif nose[0] < 0.5: 
            self.body_direction = "Left"
        else:
            print("Error: Unable to determine body direction.")
            os.exit(1)      
            
            
    def draw_landmarks(self, image, pose_landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
            
class Squat(BaseExercise):
    STATE_THRESH = {
        "up": 150,
        "mid": 100,
    }
    
    def __init__(self, video_path, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        super().__init__(video_path, min_detecting_confidentce, min_tracking_confidence)
        
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

            self.get_direction(landmarks)
            
            hip_angle, knee_angle, l_knee_foot, r_knee_foot = self.get_calculated_angles(landmarks)
            self.update_squat_state(knee_angle)
                            
            if self.state == "mid":
                has_issues, feedback = self.check_squat(hip_angle, knee_angle, l_knee_foot, r_knee_foot)
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
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        elif self.body_direction == "Right":    
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]  
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
        
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        #hip_angle = (left_hip_angle + right_hip_angle) / 2
        #knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        #l_knee_foot_x = left_knee[0] - left_foot[0]
        #r_knee_foot_y = right_knee[1] - right_foot[1]
        
        l_knee_foot_x = 0
        r_knee_foot_y = 0
        
        return hip_angle, knee_angle, l_knee_foot_x, r_knee_foot_y
    
    
    def check_squat(self, hip_angle, knee_angle, l_knee_foot, r_knee_foot):
        has_issues = False
        
        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle
            self.is_going_up = False
        elif knee_angle > self.min_knee_angle:
            self.is_going_up = True
        
        if self.state == "mid" and not self.is_going_up:
            if not (60 < hip_angle < 120 and knee_angle < 100 and l_knee_foot <= 0.1 and r_knee_foot <= 0.1):
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
                if l_knee_foot > 0.1 or r_knee_foot > 0.1:
                    self.rep_errors.add("Knee should not extend beyond toe tips.")
                    
            else:
                self.rep_errors.clear()
                
        return has_issues, "Incorrect" if has_issues else "Correct"
    
    
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
           

class Pushup(BaseExercise):
    STATE_THRESH = {
        "up": 150,
        "mid": 95,
    }
    
    def __init__(self, video_path, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        super().__init__(video_path, min_detecting_confidentce, min_tracking_confidence)
        
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
                    self.rep_errors.add("Too shwallow")
                
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
        
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]  
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        return elbow_angle, hip_angle
    
    
    def display_angles(self, image, frame_width, frame_height, hip_angle, elbow_angle, landmarks):
        right_hip_coordinates = _normalized_to_pixel_coordinates(
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x, 
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y,
            frame_width, frame_height)
        
        left_hip_coordinates = _normalized_to_pixel_coordinates(
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, 
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y,
            frame_width, frame_height)
        
        right_elbow_coordinates = _normalized_to_pixel_coordinates(
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x, 
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y,
            frame_width, frame_height)
        
        left_elbow_coordinates = _normalized_to_pixel_coordinates(
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x, 
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y,
            frame_width, frame_height)
        
        cv2.putText(image, str(int(hip_angle)), left_hip_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(elbow_angle)), left_elbow_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    