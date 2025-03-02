# MovAssist: Real-time Exercise Form Analysis

## Overview
MovAssist is a real-time exercise form analysis system that uses computer vision and pose estimation to help users perform selected exercises correctly. The system provides immediate feedback on form, counts repetitions, and generates exercise summaries.

## Features
- **Pose Tracking**: Tracks squats and push-ups via webcam or video using MediaPipe’s real-time pose estimation.
- **Form Feedback**: Delivers instant feedback on exercise form (e.g., "Knees too forward") with customizable angle thresholds.
- **Rep Counting**: Monitors repetition counts and distinguishes correct vs. incorrect reps.
- **Interactive GUI**: Featuring video replay, a detailed configuration editor, and exercise summaries.
- **AI Trainer Chat**: Offers real-time AI fitness chat in a separate window.

## Demos

### Squat
| Correct Form | Incorrect Form |
|--------------|----------------|
| ![Squat Correct](https://github.com/AzzyCode/MovAssist/blob/main/assets/squat_correct.gif?raw=true) | ![Squat Incorrect](https://github.com/AzzyCode/MovAssist/blob/main/assets/squat_incorrect.gif?raw=true) |

### Push-up
| ![Push-up Demo](https://github.com/AzzyCode/MovAssist/blob/main/assets/pushup.gif?raw=true) |


## Installation
1. Clone the repository: git clone git@github.com:AzzyCode/MovAssist.git
2. Install dependencies: pip install -r requirements.txt
3. Run the application: python -m src.main

Requirements: Python 3.8+, PySide6, OpenCV, MediaPipe, TensorFlow, and an OpenRouter API key (stored in `.env`).

## Usage
1. Launch MovAssist with `python -m src.main`.
2. Select "Squat" or "Pushup" from the main window.
3. Use a webcam or load a video file to start tracking.
4. Receive real-time feedback on your form and rep count.
5. Adjust exercise thresholds via the "Config" tab.
6. Chat with the AI trainer for tips (e.g., "How deep should I squat?").

**Tip: Check summary/ for post-workout stats and replays/ for saved videos.**

## Project Status
This is a work-in-progress project. Current focus: improving form detection accuracy and integrating ML models for evaluate general repetition form.

## Contributing
Contributions are welcome! Potential areas:

- Enhancing form detection with ML models.
- Adding new exercises (e.g., lunges).
- Improving UI/UX or performance.

## License
MovAssist is licensed under the MIT License. See LICENSE for details.
