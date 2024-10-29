from collections import Counter

# Parameters
gesture_buffer = []
buffer_size = 3  # You can adjust this based on testing

def get_stable_gesture(new_gesture):
    # Maintain a rolling window of the last few gestures
    gesture_buffer.append(new_gesture)
    if len(gesture_buffer) > buffer_size:
        gesture_buffer.pop(0)
    
    # Get the most common gesture in the buffer
    common_gesture = Counter(gesture_buffer).most_common(1)[0][0]
    return common_gesture
