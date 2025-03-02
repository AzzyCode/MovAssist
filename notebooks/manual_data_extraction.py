import cv2
import mediapipe as mp
import os
import numpy as np
import json
import glob
import csv 

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True,
)

def extract_landmarks(frame):
    """
    Extract pose landmarks from a frame using MediaPipe Pose.
    """
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        features = [coord for landmark in landmarks for coord in (landmark.x, landmark.y, landmark.z, landmark.visibility)]
        return features
    return None

def manually_mark_reps(video_path, resize_height=720):
    """
    Play the video and allow the user to mark the start and end of each rep using the Z and X keys.
    The video is resized for better visualization if its height exceeds the specified resize_height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return []

    reps = []  # Store start and end frames of each rep
    current_rep = None  # Track the current rep being marked

    print("Instructions: Press 'Z' to start a rep, 'X' to end it, and 'Q' to quit marking.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video.")
            break

        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Resize the frame if its height exceeds the resize_height
        original_height, original_width = frame.shape[:2]
        if original_height > resize_height:
            scale = resize_height / original_height
            frame = cv2.resize(frame, (int(original_width * scale), resize_height))

        cv2.imshow("Video", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('z'):  # Start a rep
            if current_rep is None:
                current_rep = [int(cap.get(cv2.CAP_PROP_POS_FRAMES))]
                print(f"Started rep at frame {current_rep[0]}")
            else:
                print("Rep already in progress. End it first with 'X'.")

        elif key == ord('x'):  # End a rep
            if current_rep is not None:
                current_rep.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                reps.append(current_rep)
                print(f"Ended rep at frame {current_rep[1]}. Saved as rep {len(reps)}.")
                current_rep = None
            else:
                print("No rep in progress. Start one first with 'Z'.")

        elif key == ord('q'):  # Quit marking
            print("Exiting manual marking.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return reps

def downsample_frames(frames, target_frames):
    """
    Downsample a list of frames to a target number of frames.
    """
    indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
    return [frames[i] for i in indices]

def process_video(video_path, label, exercise_name, output_folder, target_frames=30, resize_height=720):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reps = manually_mark_reps(video_path, resize_height)
    if not reps:
        print(f"No reps marked for {video_path}. Skipping video.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    csv_file = os.path.join(output_folder, "push_metadata.csv")
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not csv_exists:
            writer.writerow(["exercise_name", "file_name", "rep_number", "label", "npy_file", "json_file"])

        for i, (start_frame, end_frame) in enumerate(reps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            rep_frames = []
            frame_landmarks_data = []

            while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"Error reading frame in rep {i + 1}. Skipping.")
                    break

                landmarks = extract_landmarks(frame)
                if landmarks is not None:
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_landmarks_data.append({
                        "frame": frame_number,
                        "landmarks": landmarks
                    })
                    rep_frames.append(landmarks)

            if not rep_frames:
                print(f"No landmarks extracted for rep {i + 1}. Skipping.")
                continue

            if len(rep_frames) > target_frames:
                rep_frames = downsample_frames(rep_frames, target_frames)
            else:
                padding = [np.zeros_like(rep_frames[0]) for _ in range(target_frames - len(rep_frames))]
                rep_frames.extend(padding)

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            rep_file = os.path.join(output_folder, f"{video_name}_rep{i+1}.npy")
            np.save(rep_file, rep_frames)

            metadata = {
                "exercise_name": exercise_name,
                "label": label,
                "file_name": os.path.basename(video_path),
                "rep_number": i + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "total_frames": len(rep_frames),
                "num_features": len(rep_frames[0]) if rep_frames else 0,
                "landmarks": frame_landmarks_data
            }
            metadata_file = os.path.join(output_folder, f"{video_name}_rep{i+1}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)

            writer.writerow([exercise_name, os.path.basename(video_path), i + 1, label, rep_file, metadata_file])
            print(f"Saved rep {i + 1} landmarks to {rep_file}")
            print(f"Saved rep {i + 1} metadata to {metadata_file}")

    cap.release()
    cv2.destroyAllWindows()
    

def process_all_videos(folder_path, label, exercise_name, output_folder, target_frames=30, resize_height=720):
    """
    Process all videos in a specified folder to extract landmarks for each manually marked rep.
    """
    # Get a list of all video files in the folder (common formats: mp4, avi, mov)
    video_files = glob.glob(os.path.join(folder_path, "*.mp4")) + \
                  glob.glob(os.path.join(folder_path, "*.avi")) + \
                  glob.glob(os.path.join(folder_path, "*.mov"))
    
    if not video_files:
        print(f"No video files found in folder: {folder_path}")
        return

    print(f"Found {len(video_files)} video(s) in folder: {folder_path}")

    for video_path in video_files:
        print(f"\nProcessing video: {video_path}")
        try:
            process_video(video_path, label, exercise_name, output_folder, target_frames)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")


# Example usage
video_path = "input_data/pushup/bad"
output_folder = "output_data"
label = "Bad"
exercise_name = "pushup"

process_all_videos(video_path, label, exercise_name, output_folder)
