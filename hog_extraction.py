import os
from skimage import io
from skimage.feature import hog
import numpy as np

train_dir = os.path.join(os.path.join(os.getcwd(), "dcped"), '1')  # 訓練フォルダセット

pj = lambda x: os.path.join(train_dir, x)  # トップディレクトリパス結合
# 訓練データパス
train_files = [[os.path.join(parent, child) for child in os.listdir(parent)] for parent in map(pj, os.listdir(train_dir))]

x, y = [], []
# HOG特徴量抽出
for label in range(len(train_files)):
    for train_file in train_files[label]:
        person_img = io.imread(train_file, as_gray=True)  # 画像読み込み

        # 特徴量抽出
        fd = hog(person_img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(3, 3), visualize=False)

        x.append(fd)
        y.append(label)

x = np.array(x)
y = np.array(y)

# 特徴量保存
np.savetxt("HOG_car_data.csv", x, fmt="%f", delimiter=",")
np.savetxt("HOG_car_target.csv", y, fmt="%.0f", delimiter=",")
