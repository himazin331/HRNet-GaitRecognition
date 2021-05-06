import numpy as np
from sklearn import svm
import joblib
from sklearn.model_selection import GridSearchCV

# 特徴量読み込み
x = np.loadtxt("HOG_car_data.csv", delimiter=",")
y = np.loadtxt("HOG_car_target.csv", delimiter=",")

# ハイパーパラメータ候補
tuned_parameters = [{'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}]

# LinearSVC: 10000イテレーション, グリッドサーチ, 交差検証: 5分割
gscv = GridSearchCV(svm.LinearSVC(max_iter=10000,), tuned_parameters, cv=5)
gscv.fit(x, y) # ハイパーパラメータ探索
# 最適解モデル取得
svm_best = gscv.best_estimator_

# SVM学習
svm_best.fit(x, y)
# パラメータ保存
joblib.dump(svm_best, 'human_detector.pkl', compress=9)
