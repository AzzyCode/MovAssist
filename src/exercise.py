import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

from utils import calculate_angle, display_counter, display_feedback, display_status
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

        self.rep_count = 0
        self.rep_completed = False
        self.rep_errors = set()

        self.feedback_timestamp = 0
        self.feedback_duration = 3
        self.set_feedback = False

        self.body_direction = ""

    def process_frame(self, frame):
        """Process a video frame to analyze exercise."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            self.determine_body_direction(landmarks)
            required_landmarks = self.get_required_landmarks()

            if not self.validate_required_landmarks(landmarks, required_landmarks):
                display_status(image, "Please position yourself correctly")
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
            

class Squat(BaseExercise):
    """Class to process and analyze squats."""
    STATE_THRESH = {
        "up": 150,
        "mid": 120,
        "depth": 90,
    }

    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        super().__init__(min_detecting_confidentce, min_tracking_confidence)

        self.min_knee_angle = 180
        self.lowest_knee_angle = 180

        self._last_hip_angle = 0
        self._last_knee_angle = 0
        self._last_ankle_angle = 0

    def process_exercise(self, landmarks, image):
        """Process squat exercise logic."""
        hip_angle, knee_angle, ankle_angle = self.get_calculated_angles(landmarks)
        self._last_hip_angle = hip_angle
        self._last_knee_angle = knee_angle
        self._last_ankle_angle = ankle_angle

        self.update_squat_state(knee_angle)

        if self.state == "mid":
            has_issues = self.check_squat_form(hip_angle, knee_angle, ankle_angle)
            if has_issues:
                self.set_feedback = True
                self.feedback_timestamp = time.time()

        if self.rep_completed and self.last_state == "mid":
            self.rep_count += 1
            self.rep_completed = False
            #self.rep_errors.clear()

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

    def check_squat_form(self, hip_angle, knee_angle, ankle_angle):
        """Check squat form for errors with improved tracking and feedback."""
        has_issues = False

        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle
            self.is_going_up = False
        elif knee_angle > self.min_knee_angle:
            self.is_going_up = True

        if self.state == "mid" and knee_angle < self.STATE_THRESH["depth"]:

            # Check 1: Hip Position (back angle)
            if hip_angle < 45:  # Leaning too far forward
                self.rep_errors.add("Maitain more upright position")
                has_issues = True
            elif hip_angle > 120:  # Too upright
                self.rep_errors.add("Bend forward slightly more")
                has_issues = True

            # Check 2: Knee Position
            if ankle_angle < 60:  # Knees too forward
                self.rep_errors.add("Knees going too forward")
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
            self.state = "up"
            if self.min_knee_angle < self.STATE_THRESH["depth"]:
                self.rep_completed = True  # Full squat has been completed
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


class Pushup(BaseExercise):
    """Class to process and analyze pushup exercise."""
    STATE_THRESH = {
        "up": 150,
        "mid": 95,
    }

    REQUIRED_LANDMARKS = [
        mp.solutions.pose.PoseLandmark.NOSE,
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        mp.solutions.pose.PoseLandmark.LEFT_HIP,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    ]

    def __init__(self, min_detecting_confidentce=0.5, min_tracking_confidence=0.5):
        super().__init__(min_detecting_confidentce, min_tracking_confidence)

        self.min_elbow_angle = 180

        self._last_elbow_angle = 0
        self._last_hip_angle = 0

    def process_exercise(self, landmarks, image):
        """Process pushup exercise logic."""
        elbow_angle, hip_angle = self.get_calculated_angles(landmarks)
        self._last_elbow_angle = elbow_angle
        self._last_hip_angle = hip_angle

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
            self.display_angles(image, elbow_angle, hip_angle, landmarks)

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

        cv2.putText(image, str(int(elbow_angle)), elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, str(int(hip_angle)), hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)


@dataclass
class ExerciseMetrics:
    """Data class to store exercise metrics."""
    exercise_type: str
    start_time: str
    end_time: str
    total_reps: int = 0
    correct_reps: int = 0
    incorrect_reps: int = 0
    issues_frequency: Dict[str, int] = None
    angle_metrics: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        self.issues_frequency = {}
        self.angle_metrics = {}


class ExerciseSummary:
    """Class to track and generate exercise summary statistics."""

    def __init__(self, exercise_instance):
        self.exercise = exercise_instance
        self.exercise_type = exercise_instance.__class__.__name__
        self.start_time = datetime.now()

        self.metrics = ExerciseMetrics(
            exercise_type=self.exercise_type,
            start_time=self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=""
        )

        self.last_rep_count = 0
        self.angle_values = {angle: [] for angle in self._get_tracked_angles()}

    
    def _get_tracked_angles(self):
        """Determine which angles to track based on exercise type."""
        angle_mapping = {
            "Squat": ["hip_angle", "knee_angle", "ankle_angle"],
            "Pushup": ["elbow_angle", "hip_angle"]
        }
        return angle_mapping.get(self.exercise_type, [])
    
    
    def update_metrics(self):
        """Update exercise metrics during exercise execution"""
        current_angles = {angle: getattr(self.exercise, f"_last_{angle}", None) for angle in self.angle_values}
        logging.debug(f"Current angles: {current_angles}")
        
        # Track angle values
        for angle_name, angle_value in current_angles.items():
            if angle_value is not None:
                self.angle_values[angle_name].append(float(angle_value))
                
        # Update repetition counts
        current_rep_count = self.exercise.rep_count
        if current_rep_count > self.last_rep_count:
            self.metrics.total_reps = current_rep_count
            
        if self.exercise.rep_errors:
            self.metrics.incorrect_reps += 1
            for issue in self.exercise.rep_errors:
                self.metrics.issues_frequency[issue] = self.metrics.issues_frequency.get(issue, 0) + 1
        else:
            self.metrics.correct_reps += 1
            
        self.last_rep_count = current_rep_count
        
        
    def calculate_angle_statistics(self):
        """Calculate statistical metrics for tracked angles."""
        for angle_name, values in self.angle_values.items():
            if values:
                self.metrics.angle_metrics[angle_name] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values))
                }


    def generate_summary(self) -> Dict:
        """Generate exercise summary."""
        end_time = datetime.now()
        self.metrics.end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        self.calculate_angle_statistics()

        return asdict(self.metrics)


    def save_summary(self, directory: str = "exercise_summaries"):
        """Save exercise summary to a text file with formatted output."""
        self.update_metrics()
        summary = self.generate_summary()

        os.makedirs(directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.exercise_type.lower()}_summary_{timestamp}.txt"
        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, "w") as f:
                # Header
                f.write("=" * 50 + "\n")
                f.write(f"Exercise Summary - {self.exercise_type}\n")
                f.write("=" * 50 + "\n\n")

                # Time
                f.write("Time information:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Start time: {summary['start_time']}\n")
                f.write(f"End Time: {summary['end_time']}\n\n")

                # Repetition statisics
                f.write("Repetition statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total repetitions: {summary['total_reps']}\n")
                f.write(f"Correct repetitions: {summary['correct_reps']}\n")
                f.write(f"Incorrect repetitions: {summary['incorrect_reps']}\n")

                if summary["total_reps"] > 0:
                    accuracy = (summary["correct_reps"] / summary["total_reps"]) * 100
                    f.write(f"Accuracy: {accuracy:.1f}%\n\n")

                # Issues
                if summary["issues_frequency"]:
                    f.write("Form Issues:\n")
                    f.write("-" * 20 + "\n")
                    for issue, count in summary["issues_frequency"].items():
                        f.write(f"{issue}: {count} time\n")
                    f.write("\n")

                # Angle statistics
                if summary["angle_metrics"]:
                    f.write("Angle Statistics:\n")
                    f.write("-" * 20 + "\n")
                    for angle_name, stats in summary["angle_metrics"].items():
                        f.write(f"{angle_name}:\n")
                        for stat_name, value in stats.items():
                            f.write(f" {stat_name.capitalize()}: {value:.2f}\n")

                # json_filepath = filepath.replace(".txt", ".json")
                # with open(json_filepath, "w") as json_file:
                #     json.dump(summary, json_file, indent=4)

            logging.info(f"Exercise summary saved to {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"Error saving exercise summary: {e}")
            return None
