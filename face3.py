import face_recognition
import cv2
import numpy as np

# フレームレート
desired_frame_rate = 60

# Webカメラの初期化
video_capture = cv2.VideoCapture(0)

# フレームレートを設定
video_capture.set(cv2.CAP_PROP_FPS, desired_frame_rate)

# 既知の顔の画像を読み込み、顔の特徴値（エンコーディング）を取得
aoi_image = face_recognition.load_image_file("aoi.jpg")
aoi_face_encoding = face_recognition.face_encodings(aoi_image)[0]
# 追加で登録する場合はここに追加していく(〇〇には名前を代入)
"""
〇〇_image = face_recognition.load_image_file("〇〇.jpg")
〇〇_face_encoding = face_recognition.face_encodings(〇〇_image)[0]
〇〇_image = face_recognition.load_image_file("〇〇.jpg")
〇〇_face_encoding = face_recognition.face_encodings(〇〇_image)[0]
"""
# 既知の顔のエンコーディングと名前をリストに格納
known_face_encodings = [
    aoi_face_encoding,
    """
    〇〇_face_encoding,
    〇〇_face_encoding
    """
]

known_face_names = [
    "Aoi",
    """
    "〇〇",
    "〇〇"
    """
]

while True:
    # Webカメラのフレームを読み込み
    ret, frame = video_capture.read()

    # フレームの色をBGRからRGBに変換（face_recognitionはRGBを利用）
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    # 現在のフレーム内のすべての顔の位置と顔の特徴値を検出
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # フレーム内で検出された各顔に対して
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 既知の顔と比較してマッチするかを確認
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # 既知の顔と最も近い距離の顔を見つける
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            accuracy = 1 - face_distances[best_match_index]
            # 精度が0.5以上の場合のみ名前を設定
            if accuracy > 0.5:
                name = known_face_names[best_match_index]

        # 顔の周りに四角を描画
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 名前と精度（精度が0.5以上の場合のみ）のラベルを描画
        label = f'{name}'
        if name != "Unknown":
            label += f' ({accuracy:.2f})'
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # 結果を画面に表示
    cv2.imshow('Video', frame)

    # 'q'キーが押されたらループから抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラリソースを解放
video_capture.release()
cv2.destroyAllWindows()
