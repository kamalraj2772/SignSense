import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from gtts import gTTS
import os
import time
from collections import deque
from datetime import datetime
import mediapipe as mp  


MODEL_PATH = 'models/asl_cnn.pth'
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']
LOG_FILE = 'logs/prediction_log.txt'


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# --- Speak Text and Save Audio ---
def speak(text, lang_code='en', filename='char.mp3'):
    if not text:
        return

    output_path = os.path.join('outputs', 'audio')
    os.makedirs(output_path, exist_ok=True)

    save_path = os.path.join(output_path, filename)

    tts = gTTS(text=text, lang=lang_code)
    tts.save(save_path)
    print(f"[Info] Audio saved at: {save_path}")

# --- Logging ---
def log_prediction(label, confidence):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"{datetime.now()} | Label: {label} | Confidence: {confidence:.2f}\n")

# --- Majority Voting Helper ---
def majority_vote(deque_preds):
    if not deque_preds:
        return None
    return max(set(deque_preds), key=deque_preds.count)

# --- Main Prediction Loop ---
def main():
    print("Starting real-time sign language prediction with MediaPipe hands detection...")
    print("Press [q] to quit, [space] to speak full sentence, [c] to clear sentence, [m] to save sentence as audio.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(device)

    # Setup MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    predicted_text = ""
    last_char = ''
    confidence_threshold = 0.8
    last_prediction_time = time.time()
    PREDICTION_BUFFER_SIZE = 5
    pred_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(max(min(x_coords) * w - 20, 0))
            xmax = int(min(max(x_coords) * w + 20, w))
            ymin = int(max(min(y_coords) * h - 20, 0))
            ymax = int(min(max(y_coords) * h + 20, h))

            roi = frame[ymin:ymax, xmin:xmax]

            if roi.size == 0:
                continue

            img = transform(roi).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted = torch.max(probs, 0)

                label = LABELS[predicted]
                confidence_val = confidence.item()

                if confidence_val > confidence_threshold:
                    pred_buffer.append(label)
                else:
                    pred_buffer.append('nothing')

                smooth_label = majority_vote(list(pred_buffer))

                if smooth_label and smooth_label != last_char and time.time() - last_prediction_time > 1.5:
                    if smooth_label == 'space':
                        predicted_text += ' '
                    elif smooth_label == 'del':
                        predicted_text = predicted_text[:-1]
                    elif smooth_label == 'nothing':
                        pass
                    else:
                        predicted_text += smooth_label

                    last_char = smooth_label
                    last_prediction_time = time.time()

                    print(f"Predicted: {smooth_label} | Confidence: {confidence_val:.2f}")
                    print(f"Current sentence: {predicted_text}")
                    log_prediction(smooth_label, confidence_val)

                    if smooth_label not in ['nothing', 'del']:
                        speak(smooth_label, filename='char.mp3')

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, f"Prediction: {smooth_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 2)
        else:
            pred_buffer.append('nothing')
            cv2.putText(frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)

        cv2.putText(frame, f"Sentence: {predicted_text}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        cv2.imshow("Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Speak sentence
            if predicted_text.strip():
                print(f"Speaking full sentence: {predicted_text}")
                speak(predicted_text, filename='char.mp3')
                predicted_text = ""
        elif key == ord('c'):  # Clear sentence
            print("Sentence cleared.")
            predicted_text = ""
        elif key == ord('m'):  # Save sentence to full_sentence.mp3
            if predicted_text.strip():
                speak(predicted_text, filename='full_sentence.mp3')
                print(f"Full sentence saved: '{predicted_text}'")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
