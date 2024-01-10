import cv2
import dlib

# ビデオファイルを指定
cap = cv2.VideoCapture('/Users/nishiharaaoi/Desktop/2023 大学資料/人工知能応用A/programing/IVEBADDIE.mp4')

# Dlibの顔検出器と追跡器
detector = dlib.get_frontal_face_detector()
tracker = dlib.correlation_tracker()

# 追跡を開始するためのフラグ
tracking_face = False

# 出力ビデオ設定
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output = cv2.VideoWriter('/Users/nishiharaaoi/Desktop/2023 大学資料/人工知能応用A/programing/output1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not tracking_face:
        # 顔を検出
        faces = detector(gray)
        if len(faces) > 0:
            # 最初の顔を追跡
            tracker.start_track(frame, faces[0])
            tracking_face = True

    if tracking_face:
        # 顔の追跡を更新
        tracker.update(frame)
        pos = tracker.get_position()

        # 顔の位置を描画する前に座標が有効か確認
        if pos.left() != -1 and pos.top() != -1 and pos.right() != -1 and pos.bottom() != -1:
            top_left = (int(pos.left()), int(pos.top()))
            bottom_right = (int(pos.right()), int(pos.bottom()))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)

    # 処理されたフレームを出力ビデオに書き込む
    output.write(frame)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESCキーで終了
        break

cap.release()
output.release()
cv2.destroyAllWindows()
