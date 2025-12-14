import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    # Optimized: Removed redundant double-loops and checks
    data_aux = []
    if results.multi_hand_landmarks:
        # Support for up to 2 hands
        num_hands = len(results.multi_hand_landmarks)
        
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            # 1. Collect coordinates to find min (bounding box)
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            
            min_x, min_y = min(x_), min(y_)

            # 2. Normalize and append
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

        # If only 1 hand detected, pad with zeros for the second hand (42 features)
        if num_hands == 1:
            data_aux.extend([0.0] * 42)
            
    else:
        # No hands: 84 zeros
        data_aux = [0.0] * 84
        
    return data_aux

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())