import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import json
import os
from datetime import datetime

from src.utils import calculate_angle, display_counter, display_feedback, display_status, format_duration
from src.ui import ConfigManager
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseExercise:
    """Base class for exercise pose estimation"""

    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5, use_camera=False):

        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=min_detecting_confidentce,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True,
            )
        
        
        self.state = "up"
        self.last_state = "up"
        self.is_going_up = True

        self.rep_count = 0
        self.rep_completed = False
        self.rep_errors = set()

        self.feedback_timestamp = 0
        self.feedback_duration = 1
        self.set_feedback = False

        self.body_direction = ""

        self.reps_correct = 0
        self.reps_incorrect = 0
                
        self.warmup_duration = 10
        self.warmup_start = None
        self.is_warmup = use_camera

        self.config_manager = ConfigManager()
        
        
    def process_frame(self, frame):
        """Process a video frame to analyze exercise."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.is_warmup:
            if self.warmup_start is None:
                self.warmup_start = time.time()
            
            elapsed_time = time.time() - self.warmup_start
            remaining_time = max(0, self.warmup_duration - elapsed_time)
            
            cv2.putText(
                image,
                f"Starting in: {int(remaining_time)}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            
            if remaining_time <= 0:
                self.is_warmup = False
                self.warmup_start = None
                
            return image

        if not result.pose_landmarks:
            display_status(image, "Please position yourself correctly")
            return image

        landmarks = result.pose_landmarks.landmark

        self.determine_body_direction(landmarks)
        required_landmarks = self.get_required_landmarks()

        if not self.validate_required_landmarks(landmarks, required_landmarks):
            return image
        
        self.process_exercise(landmarks, image)
        self.draw_landmarks(image, result.pose_landmarks)
        display_counter(image, self.rep_count, self.__class__.__name__)

        if self.set_feedback and (time.time() - self.feedback_timestamp < self.feedback_duration):
            display_feedback(image, self.rep_errors)
        else:
            self.set_feedback = False
        
        return image
        

    def get_coordinates(self, landmarks, mp_pose, side, joint):
        """Retrives x and y coordinates of a particular keypoint from the pose estimation model."""
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
        """Determine body direction (Left or Right) based on z-coordinates."""
        try:
            left_shoulder_z = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z
            right_shoulder_z = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z

            if left_shoulder_z < right_shoulder_z:
                self.body_direction = "Left"
            else:
                self.body_direction = "Right"
        except KeyError as e:
            logging.error(f"KeyError determinig body directions {e}")
            self.body_direction = None


    def validate_required_landmarks(self, landmarks, required_landmarks) -> bool:
        """Check if all required landmarks are visible."""
        if not required_landmarks:
            logging.error("No required landmarks specified")
            return False
        try:
            for landmark_idx in required_landmarks:
                landmark = landmarks[landmark_idx]
                if not hasattr(landmark, "visibility") or landmark.visibility < 0.5:
                    logging.debug(f"Landmark {landmark_idx} is not visible")
                    return False
            return True
        except Exception as e:
            logging.error(f"Error validating landmarks: {e}")
            return False
    
        
    def clear_rep_errors(self):
        """Clear errors for the next repetition, ensuring metrics are updated first."""
        if self.rep_completed:
            self.rep_errors.clear()
            
    
    def check_reps(self):
        if self.rep_errors:
            self.reps_incorrect += 1
        else:
            self.reps_correct += 1
                    
        
class Squat(BaseExercise):
    # """Class to process and analyze squats."""
    # STATE_THRESH = {
    #     "up": 150,
    #     "mid": 120,
    #     "depth": 90,
    # }
    
    # ANGLE_THRESHOLDS = {
    #     "depth": 90,
    #     "min_depth": 40,
    #     "hip_min": 60,
    #     "hip_max": 90,
    #     "ankle_min": 80
    # }

    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5, use_camera=False):
        super().__init__(min_detecting_confidentce, min_tracking_confidence, use_camera=use_camera)

        self.min_knee_angle = 180
        self.lowest_knee_angle = 180
        
        self.start_time = time.time()
        
        self.min_squat_angles = {
            "knee_angles": [],
            "hip_angles": [],
            "ankle_angles": []
        }
        
        self.STATE_THRESH = self.config_manager.get_state_thresholds(self.__class__.__name__)
        self.ANGLE_THRESHOLDS = self.config_manager.get_angle_thresholds(self.__class__.__name__)


    def process_exercise(self, landmarks, image):
        """Process squat exercise logic."""
        hip_angle, knee_angle, ankle_angle = self.get_calculated_angles(landmarks)
        
        
        #self.update_frame_buffer(landmarks)
        self.update_squat_state(knee_angle)
        
        if self.state == "mid":
            has_issues = self.check_squat_form(hip_angle, knee_angle, ankle_angle, landmarks)
            if has_issues:
                self.set_feedback = True
                self.feedback_timestamp = time.time()

        if self.rep_completed and self.last_state == "mid":
            self.rep_count += 1
            self.check_reps()
            self.rep_completed = False
        #     prediction_result = self.get_prediction()
        
        #     if prediction_result is not None:
        #         logging.info(f"Prediction result: {prediction_result}")
        #         if prediction_result == "Good":
        #             logging.info("Good form detected")
        #             self.rep_count += 1
        #         else:
        #             logging.info("Bad form detected")
        #         self.check_reps()
        #         self.rep_completed = False
        #         self.rep_frame_sequence = []

        self.last_state = self.state
        self.display_angles(image, hip_angle, knee_angle, ankle_angle, landmarks)
        self.clear_rep_errors()


    def get_calculated_angles(self, landmarks):
        """Calculate hip, knee and ankle angles based on landmarks."""
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


    def check_squat_form(self, hip_angle, knee_angle, ankle_angle, landmarks):
        """Check squat form for errors with improved tracking and feedback."""
        has_issues = False

        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle
            self.min_hip_angle = hip_angle
            self.min_ankle_angle = ankle_angle
            self.is_going_up = False
        elif knee_angle > self.min_knee_angle:
            self.is_going_up = True
            
        if self.state == "mid" and knee_angle < self.STATE_THRESH["depth"]:
            
            # Check 1: Hip Position (back angle)
            if hip_angle < self.ANGLE_THRESHOLDS["hip_min"]:  # Leaning too far forward
                self.rep_errors.add("Maitain more upright position")
                has_issues = True
            elif hip_angle > self.ANGLE_THRESHOLDS["hip_max"]:  # Too upright
                self.rep_errors.add("Bend forward slightly more")
                has_issues = True

            # Check 2: Knee Postion
            if ankle_angle < self.ANGLE_THRESHOLDS["ankle_min"]:  # Knees too forward
                self.rep_errors.add("Knees going too forward")
                has_issues = True

            # Check 3: Knee Depth
            if knee_angle < self.ANGLE_THRESHOLDS["min_depth"]:  # Too deep
                self.rep_errors.add("Squat too deep")
                has_issues = True

        elif self.state == "mid" and self.is_going_up:
            if self.min_knee_angle > self.STATE_THRESH["depth"]:
                self.rep_errors.add("Squat not deep enough")
                has_issues = True
        
        return has_issues


    def update_squat_state(self, knee_angle):
        """Update squat state based on knee angle."""
        # Going down
        if self.state == "up" and knee_angle < self.STATE_THRESH['mid']:
            self.state = "mid"
        # Going up
        elif self.state == "mid" and knee_angle > self.STATE_THRESH['up']:
            if self.min_knee_angle < self.STATE_THRESH["depth"]:
                self.rep_completed = True  # Full squat has been complete
                
            self.state = "up"
            self.min_squat_angles["knee_angles"].append(self.min_knee_angle)
            self.min_squat_angles["hip_angles"].append(self.min_hip_angle)
            self.min_squat_angles["ankle_angles"].append(self.min_ankle_angle)
            self.min_knee_angle = 180
            
            

    def get_required_landmarks(self) -> list:
        """Get required landmarks for squat exercise."""
        mp_pose = mp.solutions.pose

        if not self.body_direction:
            logging.warning("Body direction not determinted")
            return None

        side = self.body_direction
        return [
            mp_pose.PoseLandmark.NOSE,
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_SHOULDER"),
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP"),
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_KNEE"),
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_ANKLE"),
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_FOOT_INDEX"),
        ]


    def display_angles(self, image, hip_angle, knee_angle, ankle_angle, landmarks):
        """Display angles on the image."""
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

        cv2.putText(image, str(int(hip_angle)), hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, str(int(knee_angle)), knee_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, str(int(ankle_angle)), ankle_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        
        
    def generate_summary(self, filename = None) -> None:
        """Generate summary and save to JSON file."""
        
        if not os.path.exists("summary"):
            os.makedirs("summary")
        
        if filename is None:
            current_data = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join("summary", f"pushup_summary_{current_data}.json")
        
        duration = format_duration(time.time() - self.start_time)
        
        if self.min_squat_angles:
            stats = {
                "session_info": {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": duration,
                },
                "angles": {
                    "knee_angles": {
                        "average": sum(self.min_squat_angles["knee_angles"]) / len(self.min_squat_angles["knee_angles"]),
                        "min": min(self.min_squat_angles["knee_angles"]),
                        "max": max(self.min_squat_angles["knee_angles"]),
                    },
                    "hip_angles": {
                        "average": sum(self.min_squat_angles["hip_angles"]) / len(self.min_squat_angles["hip_angles"]),
                        "min": min(self.min_squat_angles["hip_angles"]),
                        "max": max(self.min_squat_angles["hip_angles"]),
                    },
                    "ankle_angles": {
                        "average": sum(self.min_squat_angles["ankle_angles"]) / len(self.min_squat_angles["ankle_angles"]),
                        "min": min(self.min_squat_angles["ankle_angles"]),
                        "max": max(self.min_squat_angles["ankle_angles"]),
                    },
                }
            }
            
            with open(filename, "w") as f:
                json.dump(stats, f, indent=4)
                
            print(f"Statistic saved to {filename}")


class Pushup(BaseExercise):
    """Class to process and analyze pushup exercise."""    
    # STATE_THRESH = {
    #     "up": 150,
    #     "mid": 95,
    # }

    # ANGLE_THRESHOLDS = {
    #     "hip_low": 190,
    #     "hip_high": 160,
    #     "elbow_min": 50,
    # }
    
    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5, use_camera=False):
        super().__init__(min_detecting_confidentce, min_tracking_confidence, use_camera=use_camera)

        self.min_elbow_angle = 180
        
        self.start_time = time.time()
        
        self.min_pushup_angles = {
            "elbow_angles": [],
            "hip_angles": [],
        }
        
        self.STATE_THRESH = self.config_manager.get_state_thresholds(self.__class__.__name__)
        self.ANGLE_THRESHOLDS = self.config_manager.get_angle_thresholds(self.__class__.__name__)
        
        
    def process_exercise(self, landmarks, image):
        """Process pushup exercise logic."""
        elbow_angle, hip_angle = self.get_calculated_angles(landmarks)
       
        self.update_pushup_state(elbow_angle)

        if self.state == "mid":
            has_issues = self.check_pushup_form(elbow_angle, hip_angle)
            if has_issues:
                self.set_feedback = True
                self.feedback_timestamp = time.time()

        if self.rep_completed and self.last_state == "mid":
            self.rep_count += 1
            self.check_reps()
            self.rep_completed = False

        self.last_state = self.state
        self.display_angles(image, elbow_angle, hip_angle, landmarks)
        self.clear_rep_errors()
        

    def check_pushup_form(self, elbow_angle, hip_angle):
        """Check pushup form for erros"""
        has_issues = False

        if elbow_angle < self.min_elbow_angle:
            self.min_elbow_angle = elbow_angle
            self.min_hip_angle = hip_angle
            self.is_going_up = False
        elif elbow_angle > self.min_elbow_angle:
            self.is_going_up = True

        if self.state == "mid":
            if elbow_angle < self.ANGLE_THRESHOLDS["elbow_min"]:
                self.rep_errors.add("Too deep")
                has_issues = True

            if hip_angle < self.ANGLE_THRESHOLDS["hip_high"]:
                print(hip_angle)
                print(self.ANGLE_THRESHOLDS["hip_high"])
                self.rep_errors.add("Hip too high")
                has_issues = True
            elif hip_angle > self.ANGLE_THRESHOLDS["hip_low"]:
                self.rep_errors.add("Hip too low")
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
            self.rep_completed = True
            
            self.min_pushup_angles["elbow_angles"].append(self.min_elbow_angle)
            self.min_pushup_angles["hip_angles"].append(self.min_hip_angle)
            self.min_elbow_angle = 180 
            

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


    def get_required_landmarks(self) -> list:
        """Get required landmarks for pushup exercise."""
        mp_pose = mp.solutions.pose

        if not self.body_direction:
            logging.warning("Body direction not determinted")
            return None

        side = self.body_direction
        return [
            mp_pose.PoseLandmark.NOSE,
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_SHOULDER"),
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP"),
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_KNEE"),
            getattr(mp_pose.PoseLandmark, f"{side.upper()}_ANKLE"),
        ]

    
    def display_angles(self, image, elbow_angle, hip_angle, landmarks):
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

        cv2.putText(image, str(int(elbow_angle)), elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, str(int(hip_angle)), hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)


    def generate_summary(self, filename = None) -> None:
        """Generate summary and save to JSON file."""
        
        try:
            if not os.path.exists("summary"):
                os.makedirs("summary")
        except Exception as e:
            print(f"Error creating 'summary' folder: {e}")
            return
            
        if filename is None:
            current_data = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join("summary", f"pushup_summary_{current_data}.json")
        
        duration = format_duration(time.time() - self.start_time)
        
        if self.min_pushup_angles:
            stats = {
                "session_info": {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": duration,
                },
                "reps": {
                    "completed reps": self.rep_count,
                    "correct reps": self.reps_correct,
                    "incorrect reps": self.reps_incorrect, 
                },
                "angles": {
                    "elbow_angles": {
                        "average": sum(self.min_pushup_angles["elbow_angles"]) / len(self.min_pushup_angles["elbow_angles"]),
                        "min": min(self.min_pushup_angles["elbow_angles"]),
                        "max": max(self.min_pushup_angles["elbow_angles"]),
                    },
                    "hip_angles": {
                        "average": sum(self.min_pushup_angles["hip_angles"]) / len(self.min_pushup_angles["hip_angles"]),
                        "min": min(self.min_pushup_angles["hip_angles"]),
                        "max": max(self.min_pushup_angles["hip_angles"]),
                    },
                }
            }
            try:
                with open(filename, "w") as f:
                    json.dump(stats, f, indent=4)
                print(f"Statistics saved to {filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
                
        else:
            print("Error: Invalid or missing angle data.")