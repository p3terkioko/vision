# Gesture Recognition Model
## Overview
There are many ways to control a robot. You could use a joystick to operate it remotely almost like how you control characters in video games. You could use voice commands , physical buttons or even simply allow it to 
manouver around using the information it gets from the environment i.e. using sensors.However, the scope of this project deals with a robot controlled using gestures. Specifically, the code that is used to recognise and classify the gestures as seen by a camera. 

This project has the following main parts:
1. Capturing the images
2. Processing the frames
3. Extracting hand position information
4. Recording data
5. Training a model to predict the hand gestures
6. Implementing the model to correctly identify hand gestures from a video stream
7. Testing

## Features
- Real-time hand gesture recognition.
- Uses MediaPipe for hand landmark detection.
- KNN model for gesture classification.
- Outputs recognized gestures to the console.

## Requirements
Make sure you have the following installed:
- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- Pandas

You can install the required packages using pip:
```bash
pip install opencv-python mediapipe numpy pandas
```

## Capturing the images
Capturing images is the first step in real-time hand gesture recognition. This involves accessing the webcam and continuously retrieving frames (images) from it. The captured frames serve as input for further processing to detect and analyze hand gestures.

To effectively capture images, the code establishes a video stream from the webcam. This stream consists of a sequence of images (frames) that are read and processed in real-time. The logic ensures that frames are captured as long as the video feed is active, allowing for continuous gesture recognition.

### Libraries/Functions Used:
`OpenCV (cv2):` This library is used for image and video processing. The specific function used to capture video is `cv2.VideoCapture()`.
#### Sample usage
```py
#Initializing the camera feed
cap = cv2.VideoCapture(0)
```

```py
#Loop to capture the frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
```
`cap.read()` captures a single frame from the webcam. It returns two values: `ret`, a boolean indicating if the frame was captured successfully, and `frame`, the actual image captured.
The loop continues as long as the camera feed is open, ensuring continuous image capturing until the user decides to stop it.

## Processing the Frames
Once images are captured from the webcam, they need to be processed to facilitate hand detection. This step includes flipping the frame and converting the color format from BGR (default in OpenCV) to RGB, which is required by MediaPipe for hand landmark detection.

The processing involves adjusting the captured image for optimal analysis. Flipping the image horizontally simulates a mirror view, making it intuitive for users to see their hands in the frame. The color conversion prepares the image for the MediaPipe library, which expects RGB input.
### Libraries/Functions Used:
`OpenCV (cv2)`: Used for image manipulation functions.
`MediaPipe (mp)`: Specifically, the mp.solutions.hands module is used for hand landmark detection.
### Sample usage
```py
#Flips the frame
frame = cv2.flip(frame, 1)
```
```py
#converts the color space from BGR to RGB using cv2.cvtColor(), as MediaPipe requires RGB format for its processing
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

## Extracting Hand Position Information
Extracting hand position information involves detecting hand landmarks from the processed frames. The landmark coordinates are used to calculate angles between fingers and the wrist, which serve as features for gesture recognition.

MediaPipe's hand detection models identify specific landmarks on the hand, such as the wrist, fingertips, and joints. By calculating the positions and angles formed by these landmarks, the system can recognize different hand gestures based on predefined criteria.
### Libraries/Functions Used
`MediaPipe (mp)`: The mp.solutions.hands module is used for detecting hand landmarks.
`Numpy (np)`: Used for mathematical operations, such as calculating angles.
### Sample usage
```py
results = hands.process(rgb_frame)
```
This line processes the RGB frame to detect hand landmarks. The `results` object contains information about detected landmarks.
```py
wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
# ... other fingers
```
Here, the landmark positions are extracted for each finger and the wrist. Each landmark is represented as a `Landmark` object with `x` and `y` properties that indicate the position in normalized coordinates.
```py
def calculate_angle(tip_pos):
    delta_tip = tip_pos - wrist_pos
    delta_center = frame_center - wrist_pos
    angle = np.degrees(np.arctan2(delta_tip[1], delta_tip[0]) - np.arctan2(delta_center[1], delta_center[0])) % 360
    return angle
```
This function calculates the angle between the tip of the finger and the wrist, relative to the center of the frame. The use of `np.arctan2()` helps to determine the angle in degrees based on the differences in coordinates.

## Recording data
We could simply implement some logic in order to predict the gesture every frame. However this approach has several drawbacks in practise. For example, it is difficult to predict the exact gesture using just the information from the finger positions
since one's hand isn't always in the same exact position every time. Even if we created a lower and upper bound for these positions, a fixed approach is not very flexible and will require re-programming every time.

In light of this, this project utilises a mashine learning algorithm to predict the hand geatures because it is trainable and easily customizable. Since our database is relatively small with few dimensions, KNN is a suitable
machine learning algorithm to use. Therefore, there is need to record training data that will be used to train the machine learning model.

This is implemented in `dataRecorder.py`. This file records the positions(angles) of each finger in a csv file to create a dataset for a particular gesture. By combining datasets for different hand gestures, we can produce a comprehesnive dataset to use with our KNN model.
To run it:
```bash
py dataRecorder.py
```
This will prompt you to enter the filename for that particular dataset and what gesture you are recording for. After that the openCV window will open, make sure your hand is in the correct position for the target gesture
to ensure data integrity. Move your hand around to ensure you capture as many possibilities as possible. Afterwards, terminate the program and it will create the file with the recorded data inside it.
This process was repeated for every hand gesture to create the `dataset.csv` file included in this project

## Training the model
For this project, KNN (K-nearest neighbours) was used to predict the gesture according to the angles of the fingers relative to the wrist and the centre of the frame. The angles are calculated by the `get_angles(hand_landmarks, frame_center)` function then fed into the KNN model to predict
### How KNN works
#### __Feature Comparison:__ 

KNN classifies data points based on the similarities between them. 
Each gesture in our dataset has specific features (like finger angles), and KNN compares these features with those in new, unseen data to determine the gesture.

#### __Distance Calculation:__ 

When predicting a gesture, KNN calculates the distance between the new data point (finger angles captured in real-time) and every point in the training dataset.

#### __Selecting Neighbors:__

Once the distances are calculated, KNN identifies the k closest points (neighbors). The value of k is a parameter you choose, often by trial and error, which influences the accuracy. In our code, k=3 means KNN will look at the three nearest gestures in the dataset.

#### __Voting Mechanism:__ 

For classification tasks like gesture recognition, KNN uses a majority vote among the k neighbors. The gesture label that appears most frequently among the neighbors is assigned as the predicted gesture for the new data point.

Sample usage
```py
from KNN import CustomKNN
import pandas as pd
# Load the dataset
data = pd.read_csv('./dataset.csv')
# Prepare features (X) and labels (y)
X = data.iloc[:, :-1].values  # All rows, all columns except the last one
y = data.iloc[:, -1].values   # All rows, only the last column (gesture labels)
# Create an instance of CustomKNN
knn = CustomKNN(k=3)  # You can experiment with different values of k
# Fit the model
knn.fit(X, y)
# Make predictions on a sample (for example, the first test sample)
sample_data = np.array([[338.27794587544304, 356.3608903714182, 326.48402123972147, 317.0225736078043, 300.0044154933058]])  # Replace with actual angles
prediction = knn.predict(sample_data)
print(f'Predicted Gesture: {prediction[0]}')
```

## Implementation
This is easier explained using a flow chart

![computer vision drawio](https://github.com/user-attachments/assets/bb73ae3a-83d6-476a-ac35-16e4ff086e00)

## Testing
To run the project, clone the project
Open the folder with this repo
```bash
py ImageRecognision.py
```

## Acknowledgements
[MediaPipe](https://pypi.org/project/mediapipe/) for hand tracking.
[OpenCV](https://opencv.org/) for image processing functionalities.
