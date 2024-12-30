# This file is used to collect hand gesture data
import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.3,
                       max_num_hands=2)

data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dataset_size = 100

cap = cv2.VideoCapture(0)  # Use 0 for default camera
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

file_num = 0
while True:
    _ignore, frame = cap.read()
    cv2.putText(frame,
                'Press "S" to start | Press "Q" to quit',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
                )
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        while os.path.exists(data_dir + '/' + str(file_num)):
            file_num += 1
        new_dir = data_dir + '/' + str(file_num)
        os.makedirs(new_dir, exist_ok=True)
        print(f'Collecting data for class {file_num}')

        # collecting images...
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            cv2.putText(frame,
                        f'Collecting data: {counter + 1} of {dataset_size}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA
                        )
            cv2.waitKey(20)
            cv2.imshow('frame', frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    cv2.imwrite(new_dir + '/' + str(counter) + '.jpg', frame)
                    counter += 1

        # image collection complete, show new frame to indicate that
        i = 0
        while counter == dataset_size:
            # Capture a fresh frame to clear previous text
            ret, close_frame = cap.read()
            cv2.putText(close_frame,
                        'Data Collection Completed!',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                        )
            cv2.imshow('frame', close_frame)
            cv2.waitKey(1)
            i += 1
            if i == 50:
                break

        file_num += 1

hands.close()
cap.release()
cv2.destroyAllWindows()
