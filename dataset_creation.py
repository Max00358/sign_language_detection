# This file converts gesture images into landmarks
import os
import pickle           # Serialize & save data for later use
import mediapipe as mp  # Used for hand landmark detection and visualization
import cv2              # Processes images, reads files, and converts color spaces

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3,
    max_num_hands=2,
    model_complexity=1
)

data_dir = './data'
data = []
labels = []
for one_dir in os.listdir(data_dir):  # loop through all hand gesture directories
    gesture_dir = os.path.join(data_dir, one_dir)
    if not os.path.isdir(gesture_dir):  # Skip non-directory entries, e.g: ./DS_Store
        continue

    for img_path in os.listdir(gesture_dir):  # loop through all img in one hand gesture directory
        data_aux = []

        x_min, y_min = float('inf'), float('inf')

        img = cv2.imread(data_dir + '/' + one_dir + '/' + img_path)
        # color conversion: cv2 uses BGR color but mediapipe expects RGB color
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:  # A list of detected hands, each represented as a set of 21 landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_min = min(x_min, x)
                    y_min = min(y_min, y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Normalization:
                    # Shifts all landmarks so the minimum x and y values are 0 (relative positioning)
                    data_aux.append(x - x_min)
                    data_aux.append(y - y_min)

            data.append(data_aux)
            labels.append(one_dir)

hands.close()
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
