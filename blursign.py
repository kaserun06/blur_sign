import cv2
import mediapipe as mp

# ===== MediaPipe Tasks =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

blur_strength = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    peace = False

    if result.hand_landmarks:

        lm = result.hand_landmarks[0]

        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y > lm[14].y
        pinky_up = lm[20].y > lm[18].y

        if index_up and middle_up and ring_up and pinky_up:
            peace = True

    if peace:
        blur_strength = min(61, blur_strength + 4)
    else:
        blur_strength = max(1, blur_strength - 4)

    if blur_strength > 1:
        k = blur_strength
        if k % 2 == 0:
            k += 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)

    cv2.putText(
        frame,
        "PEACE" if peace else "",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2,
    )

    cv2.imshow("Peace Blur", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()