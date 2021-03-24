# 姿勢推定AIを用いた歩容認証の実証実験

Last Update: 2021/03/25

## 概要
姿勢推定を行うAIを用いた歩容認証の有効性を検証する。

## 使用ライブラリ
- Python3
- Tensorflow

## プロトタイプモデルアーキテクチャ決定について
|  決定事項  |  手法  |
| ---- | ---- |
|  HRNet入力データサイズ  |  人物検出resultのバウンディングボックスサイズを基に決定 |
|  各レイヤーのHP  |  [repo](https://github.com/stefanopini/simple-HRNet)のmodels/hrnet.pyより決定  |
| ステージ数 | [link]https://jingdongwang2017.github.io/Projects/HRNet)の図に基づく

モデルアーキテクチャは状況に応じて変更する。