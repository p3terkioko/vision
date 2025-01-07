import cv2
import mediapipe as mp
import numpy as np
from sampler import get_stable_gesture  # Import the sampling function
from sendCommand import send_command
from KNN import CustomKNN
import pandas as pd
import threading
import time
from queue import Queue

# Load the dataset
data = pd.read_csv('./dataset.csv') 
# Prepare features (X) and labels (y)
X = data.iloc[:, :-1].values  # All rows, all columns except the last one
y = data.iloc[:, -1].values   # All rows, only the last column (gesture labels)
# Create an instance of CustomKNN
knn = CustomKNN(k=3)  
# Fit the model
knn.fit(X, y)

# Initialize MediaPipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_angles(hand_landmarks, frame_center):
    # Extract necessary landmarks for checking positions
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Convert landmarks to pixel coordinates
    wrist_pos = np.array([wrist.x, wrist.y])
    index_finger_pos = np.array([index_finger.x, index_finger.y])
    thumb_tip_pos = np.array([thumb_tip.x, thumb_tip.y])
    middle_finger_pos = np.array([middle_finger.x, middle_finger.y])
    ring_finger_pos = np.array([ring_finger.x, ring_finger.y])
    pinky_finger_pos = np.array([pinky_finger.x, pinky_finger.y])
    
    # Function to calculate angle formed by the tip of the finger, wrist, and center of the frame
    def calculate_angle(tip_pos):
        delta_tip = tip_pos - wrist_pos
        delta_center = frame_center - wrist_pos
        angle = np.degrees(np.arctan2(delta_tip[1], delta_tip[0]) - np.arctan2(delta_center[1], delta_center[0])) % 360
        return angle

    # Get angles for each finger
    finger_angles = [
        calculate_angle(index_finger_pos),
        calculate_angle(thumb_tip_pos),
        calculate_angle(middle_finger_pos),
        calculate_angle(ring_finger_pos),
        calculate_angle(pinky_finger_pos),
    ]
    return finger_angles

def send_commands_at_interval(port, baud_rate, command_queue, interval=1):
    while True:
        if not command_queue.empty():
            command = command_queue.get()
            send_command(port, baud_rate, command)
        time.sleep(interval)

if __name__ == "__main__":
    port = "COM4"  # Replace with the correct outgoing port for HC-05
    baud_rate = 9600
    command_queue = Queue()

    # Start the thread for sending commands
    sending_thread = threading.Thread(target=send_commands_at_interval, args=(port, baud_rate, command_queue))
    sending_thread.daemon = True
    sending_thread.start()

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            # Flip and convert the frame to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get the frame dimensions and calculate the center
            frame_height, frame_width, _ = frame.shape
            frame_center = (frame_width / 2, frame_height / 2)
            
            # Process the frame for hand landmarks
            results = hands.process(rgb_frame)

            # Access the landmarks if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    angles = get_angles(hand_landmarks, frame_center)
                    # Reshape to 2D for model prediction
                    gesture = knn.predict(angles)[0]
                    gesture = get_stable_gesture(gesture)
                    #deque the command queue
                    while not command_queue.empty():
                        command_queue.get()
                    # Enqueue the stable gesture
                    command_queue.put(gesture)
                    # Display the stable gesture
                    #print(f"Stable Gesture: {gesture}")
                    #Print the command queue
                    #print(command_queue.queue)
            
            # Display the webcam feed with annotations
            cv2.imshow('Hand Gesture Control', frame)
            
            # Exit if 'Esc' is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()