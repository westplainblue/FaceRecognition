import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# フレームレート
desired_frame_rate = 60

# Webカメラの初期化
video_capture = cv2.VideoCapture(0)

# フレームレートを設定
video_capture.set(cv2.CAP_PROP_FPS, desired_frame_rate)

# 既知の顔の画像を読み込み、顔の特徴値（エンコーディング）を取得
aoi_image = face_recognition.load_image_file("aoi.jpg")
aoi_face_encoding = face_recognition.face_encodings(aoi_image)[0]

# 既知の顔のエンコーディングと名前をリストに格納
known_face_encodings = [
    aoi_face_encoding
]

known_face_names = [
    "Aoi"
]

# 精度の履歴を格納するリスト（最新のN個の精度を保持）
accuracy_history = []
max_history_size = 50  # 履歴の最大サイズ

# グラフの初期化
plt.ion()
fig, ax = plt.subplots()

# xdataとydataを定義
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-', animated=True)

def init():
    ax.set_xlim(0, max_history_size)
    ax.set_ylim(0, 1)
    return ln,

def update(frame):
    global xdata, ydata  # グローバル変数として扱う
    xdata.append(frame)
    ydata.append(accuracy_history[-1] if accuracy_history else 0)
    xdata = xdata[-max_history_size:]
    ydata = ydata[-max_history_size:]
    ln.set_data(xdata, ydata)
    return ln,

animation = FuncAnimation(fig, update, init_func=init, blit=True, save_count=max_history_size)

frame_counter = 0
while True:
    ret, frame = video_capture.read()
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            accuracy = 1 - face_distances[best_match_index]
            if accuracy > 0.5:
                name = known_face_names[best_match_index]
                accuracy_history.append(accuracy)
                if len(accuracy_history) > max_history_size:
                    accuracy_history.pop(0)
                plt.draw()

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        label = f'{name}'
        if name != "Unknown":
            label += f' ({accuracy:.2f})'
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1
    animation.event_source.interval = 1000 / desired_frame_rate  # グラフの更新間隔を調整

video_capture.release()
cv2.destroyAllWindows()
