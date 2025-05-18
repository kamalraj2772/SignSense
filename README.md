# Real-Time Sign Language Recognition

This project implements a real-time American Sign Language (ASL) recognition system using a pretrained CNN model and MediaPipe hands detection. The system captures hand gestures through a webcam, predicts the corresponding ASL letter, and constructs sentences dynamically. It also includes speech synthesis to read the constructed sentence aloud.

---

## Features

- Real-time hand gesture detection using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- CNN-based ASL alphabet classification with a custom pretrained model (`models/asl_cnn.pth`)
- Sentence construction from sequential ASL letter predictions
- Speech synthesis to read the sentence aloud (using `gTTS` and `pygame`)
- Clear sentence functionality with a keypress
- GPU support for faster model inference if CUDA is available
- Logs predictions and sentences to a log file for debugging and training analysis

---

## Getting Started

### Prerequisites

- Python 3.8 or above
- GPU with CUDA support (optional but recommended for faster inference)
- Webcam

### Python Packages

Install required dependencies:

```bash
pip install -r requirements.txt
