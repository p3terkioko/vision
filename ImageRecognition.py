import cv2
import mediapipe as mp
import numpy as np
import serial
from sampler import get_stable_gesture  # Import the sampling function
from KNN import CustomKNN
import pandas as pd

# Load the dataset
data = pd.read_csv('./dataset.csv')
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels
knn = CustomKNN(k=3)
knn.fit(X, y)

# Bluetooth connection setup
port = "COM7"  # Update with the correct port
baud_rate = 9600
bt = serial.Serial(port, baud_rate, timeout=1)

# Gesture to command mapping
gesture_to_command = {
    'stop': 'S',
    'forward': 'F',
    'backward': 'B',
    'left': 'L',
    'right': 'R'
}

# Function to send command via Bluetooth
def send_command(command):
    if bt.is_open:
        bt.write(command.encode('utf-8'))

# Function to calculate angles from hand landmarks
def get_angles(hand_landmarks, frame_center):
    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
    index_finger = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    middle_finger = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    wrist_pos = np.array([wrist.x, wrist.y])
    index_finger_pos = np.array([index_finger.x, index_finger.y])
    thumb_tip_pos = np.array([thumb_tip.x, thumb_tip.y])
    middle_finger_pos = np.array([middle_finger.x, middle_finger.y])
    ring_finger_pos = np.array([ring_finger.x, ring_finger.y])
    pinky_finger_pos = np.array([pinky_finger.x, pinky_finger.y])

    def calculate_angle(tip_pos):
        delta_tip = tip_pos - wrist_pos
        delta_center = frame_center - wrist_pos
        angle = np.degrees(np.arctan2(delta_tip[1], delta_tip[0]) - np.arctan2(delta_center[1], delta_center[0])) % 360
        return angle

    finger_angles = [
        calculate_angle(index_finger_pos),
        calculate_angle(thumb_tip_pos),
        calculate_angle(middle_finger_pos),
        calculate_angle(ring_finger_pos),
        calculate_angle(pinky_finger_pos),
    ]
    return finger_angles

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Track the last sent command
last_command = None

# Start video capture
cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        frame_center = np.array([frame_width / 2, frame_height / 2])
        
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            gesture_detected = False
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Get angles and predict gesture
                angles = get_angles(hand_landmarks, frame_center)
                gesture = knn.predict(angles)[0]
                gesture = get_stable_gesture(gesture)
                print(f"Stable Gesture: {gesture}")

                # Send command if the gesture changes
                if gesture in gesture_to_command:
                    gesture_detected = True
                    command = gesture_to_command[gesture]
                    if command != last_command:
                        send_command(command)
                        last_command = command

            # If no gesture was detected in this frame, send the stop command
            if not gesture_detected and last_command != 'S':
                print("No gesture detected. Sending Stop command.")
                send_command('S')
                last_command = 'S'
        else:
            # No hand detected, send stop command
            if last_command != 'S':
                print("No hand detected. Sending Stop command.")
                send_command('S')
                last_command = 'S'

        cv2.imshow('Hand Gesture Control', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
bt.close()
