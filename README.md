# MovAssist: Real-time Exercise Form Analysis

## Overview
MovAssist is a real-time exercise form analysis system that uses computer vision and pose estimation to help users perform selected exercises correctly. The system provides immediate feedback on form, counts repetitions, and generates exercise summaries.

## Features
- Tracks squats and push-ups via webcam or video with MediaPipe pose estimation
- Provides real-time form feedback
- Features GUI with video replay and configuration options
- Includes an AI trainer chat offering fitness advice in a separate window

## Squat
![Alt text](https://github.com/AzzyCode/MovAssist/blob/main/assets/squat_correct.mp40)
![Alt text](https://github.com/AzzyCode/MovAssist/blob/main/assets/squat_incorect.mp4)

## Installation
1. Clone the repository: git clone https://github.com/your-username/movassist.git
2. Install dependencies: pip install -r requirements.txt
3. Run the application: python main.py

## Usage
1. Launch the application and select an exercise (squat or push-up)
2. Perform the exercise in front of a webcam or use prepared video.
3. Receive real-time feedback on form and repetition count.
4. View exercise summaries and adjust settings as needed.

## Contributing
Contributions are welcome! If you'd like to contribute to MovAssist, please fork the repository and submit a pull request with your changes.

## License
MovAssist is licensed under the MIT License. See LICENSE for details.

## Acknowledgments
MovAssist uses the following open-source libraries:

- MediaPipe
- OpenCV
- TensorFlow
We appreciate the contributions of the open-source community to these libraries.