import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

x1 = y1 = x2 = y2 = 0

webcam = cv2.VideoCapture(0)

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

while True:
    ret, image = webcam.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for id, lm in enumerate(hand_landmarks):
                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)
                cv2.circle(image, (x, y), 5, (0,0,255), -1)

                if id == 8:
                    cv2.circle(image, (x, y), 8, (0,255,255), 3)
                    x1, y1 = x, y

                if id == 4:
                    cv2.circle(image, (x, y), 8, (0,0,255), 3)
                    x2, y2 = x, y

            dist = ((x2-x1)**2 + (y2-y1)**2)**0.5 // 4
            cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 5)

            if dist > 50:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

    cv2.imshow("Hand volume control using python", image)

    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
