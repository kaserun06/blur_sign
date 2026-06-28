import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions

options = BaseOptions(
    model_asset_path="hand_landmarker.task"
)

print("Model berhasil dibaca!")