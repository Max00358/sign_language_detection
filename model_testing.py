# This file uses trained model to test & see if it recognizes hand gestures in real time
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model.p
# model_dict = pickle.load(open('./model.p', 'rb'))
# pipeline_model = model_dict['model']
# SGD_model = pipeline_model.named_steps['sgdclassifier']
# scaler = pipeline_model.named_steps['standardscaler']

# Load SGD_model.p
model_dict = pickle.load(open('./SGD_model.p', 'rb'))
SGD_model = model_dict['sgdclassifier']
scaler = model_dict['scaler']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.8,
    max_num_hands=4,
    model_complexity=1
)

labels_dict = {
                0: 'A',
                1: 'B',
                2: 'C',
                3: 'D',
                4: 'E',
                5: 'F',
                6: 'G',
                7: 'H',
                8: 'I',
                9: 'J',
                10: 'K',
                11: 'L',
                12: 'M',
                13: 'N',
                14: 'O',
                15: 'P',
                16: 'Q',
                17: 'R',
                18: 'S',
                19: 'T',
                20: 'U',
                21: 'V',
                22: 'W',
                23: 'X',
                24: 'Y',
                25: 'Z',
                26: 'SPACE',
                27: 'YVL',
                }
dict_keys = list(labels_dict.keys())

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    H, W, _ignore = frame.shape

    cv2.putText(frame,
                'Press "Q" to quit | Press "C" to correct model',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
                )
    cv2.imshow('frame', frame)
    x1, y1, x2, y2 = 0, 0, 0, 0
    predicted_character = ''
    probabilities = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,                      # image to draw
                hand_landmarks,             # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            # Reset data for each hand
            normalized_data = []
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                normalized_data.append(x - min_x)
                normalized_data.append(y - min_y)

            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max_x * W) + 10
            y2 = int(max_y * H) + 10

            scaled_data = scaler.transform(np.asarray([normalized_data]))
            prediction = SGD_model.predict(scaled_data)
            predicted_character = labels_dict[int(prediction[0])]
            probabilities = SGD_model.predict_proba([np.asarray(normalized_data)])

    if predicted_character:
        confidence = np.max(probabilities) * 100
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame,
                    f"{predicted_character} ({confidence:.2f}%)",  # {num:.2f} means keeping 2 digits float val
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA)

    # put show frame outside if statement so that the frame updates even when no hands are detected
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c') and predicted_character:  # User wants to correct
        print(f"Current prediction: {predicted_character}")
        correct_label = input("Enter correct label: ").strip().upper()
        correct_index = [str(i) for i, label in labels_dict.items() if label == correct_label]
        if correct_index:
            correct_label = correct_index[0]
            SGD_model.partial_fit(scaled_data, [correct_label], classes=SGD_model.classes_)
            with open('SGD_model.p', 'wb') as f:
                pickle.dump({'sgdclassifier': SGD_model, 'scaler': scaler}, f)

            print("Model updated with corrected label.")
        else:
            print("Input label incorrect or does not exist :(")

cap.release()
cv2.destroyAllWindows()
hands.close()
