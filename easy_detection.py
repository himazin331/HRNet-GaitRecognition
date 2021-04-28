import os
import cv2
import numpy as np

# HOG & SVM
hogdef = cv2.HOGDescriptor()
hogdef.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

data_dir = os.path.join(os.getcwd(), "data")
for c in os.listdir(data_dir):
    # 動画読み込み
    d = os.path.join(data_dir, c)
    mv = cv2.VideoCapture(d)

    c = c[:-4] + "a"
    os.makedirs(c, exist_ok=True)
    out_dir = os.path.join(os.getcwd(), c + "\\")
    for f in range(int(mv.get(cv2.CAP_PROP_FRAME_COUNT)) - 1):
        print(out_dir + c + str(f) + ".jpg")

        # フレーム取得
        _, frame = mv.read()

        # 人物検出
        found, _ = hogdef.detectMultiScale(frame)
        if type(found) is np.ndarray:
            # 保存
            for (x, y, w, h) in found:
                cv2.imwrite(out_dir + str(f) + ".jpg", frame[y:y + h, x:x + w])
    mv.release()
