import sys
import os

import cv2
import numpy as np
import random

from sklearn import model_selection as ms
from sklearn import metrics

class SVM():
    def __init__(self):
        self.score_train = []
        self.score_test = []
        self.svm = cv2.ml.SVM_create()

    # 訓練
    def train_svm(self, xtrain, xtest, ytrain, ytest):
        # 3回学習実行
        for j in range(3):
            # 学習
            self.svm.train(xtrain, cv2.ml.ROW_SAMPLE, ytrain)

            # スコア格納
            self.score_train.append(self.score_svm(xtrain, ytrain))
            self.score_test.append(self.score_svm(xtest, ytest))

            # 偽陽性結果検出
            _, ypred = self.svm.predict(xtest)
            false_pos = np.logical_and((ytest.ravel() == -1), (ypred.ravel() == 1))
            if not np.any(false_pos):
                break

            # 偽陽性サンプルをデータセットにセット
            xtrain = np.concatenate((xtrain, xtest[false_pos, :]), axis=0)
            ytrain = np.concatenate((ytrain, ytest[false_pos]), axis=0)

        return self.svm

    # 評価
    def score_svm(self, x, y):
        _, ypred = self.svm.predict(x)
        return metrics.accuracy_score(y, ypred)

class HOG():
    def __init__(self, win_size, block_size, block_stride, cell_size, num_bins):
        self.win_size = win_size # 検出対象の最小領域
        self.block_size = block_size # ボックスサイズ
        self.block_stride = block_stride # ストライドサイズ
        self.cell_size = cell_size # セルサイズ
        self.num_bins = num_bins # ビン個数

        # HOG
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

        # Postiveデータ
        self.xpos = [] # 人物のHOG特徴量
        # Negativeデータ
        self.xneg = [] # 人物以外のHOG特徴量

    def hog(self):
        return self.hog

    # データセット作成
    def create_dataset(self):
        # 正解データセットから400枚分ランダムに取り込み
        for i in random.sample(range(1, 901), 400):
            # 読み込み
            pos_data_filename = os.getcwd()+"\\pedestrian_dataset\\true\\per{:05}.ppm".format(i)
            pos_data = cv2.imread(pos_data_filename)
            
            # HOG特徴量抽出しリストセット
            self.xpos.append(self.hog.compute(pos_data, (64, 64)))
        # データ加工
        self.xpos = np.array(self.xpos, dtype=np.float32)
        ypos = np.ones(self.xpos.shape[0], dtype=np.int32) # Positiveラベル

        # ROI
        hroi = pos_data.shape[0] # 高さ
        wroi = pos_data.shape[1] # 幅

        negfilename = os.getcwd()+"\\pedestrian_dataset\\false\\"
        # 不正解データの取り込み
        for fn in os.listdir(negfilename):
            # 読み込み
            filename = negfilename+fn 
            img = cv2.imread(filename)
            img = cv2.resize(img, (512, 512))

            # ROIのHOG特徴量セット(5回実行)
            for j in range(5):
                rand_y = random.randint(0, img.shape[0] - hroi) # Y座標
                rand_x = random.randint(0, img.shape[1] - wroi) # X座標
                roi = img[rand_y:rand_y + hroi, rand_x:rand_x + wroi, :] # ROI

                # HOG特徴量抽出しリストセット
                self.xneg.append(self.hog.compute(roi, (64, 64)))
        # データ加工
        self.xneg = np.array(self.xneg, dtype=np.float32)
        yneg = -np.ones(self.xneg.shape[0], dtype=np.int32) # Negativeラベル

        # 1つに集約
        X = np.concatenate((self.xpos, self.xneg))
        Y = np.concatenate((ypos, yneg))

        # データセット(訓練データとテストデータに分割)
        return ms.train_test_split(X, Y, test_size=0.2, random_state=42), (hroi, wroi)

def main():
    
    # HOG作成
    h = HOG((48, 96), (16, 16), (8, 8), (8, 8), 9)
    # データセット作成
    (xtrain, xtest, ytrain, ytest), roi_size = h.create_dataset()
    hroi, wroi = roi_size

    # SVM作成
    s = SVM()
    # SVM学習
    svm = s.train_svm(xtrain, xtest, ytrain, ytest)

    data_dir = os.path.join(os.getcwd(), "data")

    #! エラー発生
    """
    rho, _, _ = svm.getDecisionFunction(0)
    sv = svm.getSupportVectors()
    hog.setSVMDetector(np.append(sv.ravel(), rho))
    """

    stride = 16
    for c in os.listdir(data_dir):
        # 動画読み込み
        d = os.path.join(data_dir, c)
        mv = cv2.VideoCapture(d)

        c = c[:-4]
        os.makedirs(c, exist_ok=True)
        out_dir = os.path.join(os.getcwd(), c+"\\")
        for f in range(int(mv.get(cv2.CAP_PROP_FRAME_COUNT))-1):
            print(out_dir+c+str(f)+".jpg")

            # フレーム取得
            _, frame = mv.read()

            for ystart in np.arange(0, frame.shape[0], stride): # Y方向
                for xstart in np.arange(0, frame.shape[1], stride): # X方向
                    # フレーム高さをストライドフィルタが超えたらY増分
                    if ystart + hroi > frame.shape[0]:
                        continue
                    # フレーム高さをストライドフィルタが超えたらX増分
                    if xstart + wroi > frame.shape[1]:
                        continue

                    # ROI切り出し
                    roi = frame[ystart:ystart+hroi, xstart:xstart+wroi, :]
                    # HOG特徴量抽出
                    feat = np.array([h.hog.compute(roi, (64, 64))])
                    # SVM推定
                    _, ypred = svm.predict(feat)

                    #todo ypred=1のfeatを出力
                    """
                    human, _ = h.hog.detectMultiScale(frame)
                    if type(human) is np.ndarray:
                        for (x, y, w, h) in human:
                            cv2.rectangle(frame, (x, y),(x+w, y+h),(0,50,255), 3)
                    """
            #cv2.imwrite(out_dir+str(f)+".jpg", frame)
        mv.release()

if __name__ == "__main__":
    main()