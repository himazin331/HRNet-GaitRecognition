# 姿勢推定AIを用いた歩容認証の実証実験

Last Update: 2021/05/06

## 概要
姿勢推定を行うAIを用いた歩容認証の有効性を検証する。

## 大まかな開発ステップ
> 1.人物検出モジュール作成
>> 構想:<br>
>> 動画像からフレーム毎に切り出し、HoG+SVMによる人物検出した後、検出領域の切り取り&保存

> 2.HRNet作成
>> 構想:<br>
>> 主に[repo](https://github.com/stefanopini/simple-HRNet)を参考にアーキテクチャを決定し作成

> 3.訓練モジュール作成

> 4.ジョイント描写モジュール
>> 構想:<br>
>> 情報収集中

> 5.歩容認識モジュール作成
>> 構想:<br>
>> [link](https://ipsj.ixsq.nii.ac.jp/ej/index.php?active_action=repository_view_main_item_detail&page_id=13&block_id=8&item_id=186786&item_no=1)を参考に作成

> 6.インスタンスセグメンテーションの取り入れ
>> 構想:<br>
>> 情報収集中

## プロトタイプモデルアーキテクチャ決定について
|  決定事項  |  手法  |
| ---- | ---- |
|  HRNet入力データサイズ  |  人物検出resultのバウンディングボックスサイズを基に決定 |
|  各レイヤーのHP  |  [repo](https://github.com/stefanopini/simple-HRNet)のmodels/hrnet.pyより決定  |
| ステージ数/レイヤー深度 | [link](https://jingdongwang2017.github.io/Projects/HRNet)の図に基づく

モデルアーキテクチャは状況に応じて変更する。
