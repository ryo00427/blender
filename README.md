# NeRF (Neural Radiance Fields) Implementation

## 概要
このプロジェクトは、Neural Radiance Fields (NeRF) を使った3Dシーンの生成を目的としています。NeRFは、複数の2D画像から3Dシーンの詳細な表現を学習し、新しい視点からシーンを再構成する技術です。本実装では、PyTorchを使用してNeRFの基本的なモデルをゼロから構築しています。

## 目次
1. [セットアップ](#セットアップ)
2. [データセット](#データセット)
3. [モデルの構造](#モデルの構造)
4. [学習プロセス](#学習プロセス)

## セットアップ

### 必要なライブラリ
このプロジェクトを実行するには、以下のライブラリが必要です。

- Python 3.x
- PyTorch
- NumPy
- Pillow
- その他（詳細は`requirements.txt`を参照）


## データセット
spring_deneration.pyでblenderからデータセットを用意しました
コードをご確認ください。

## モデルの構造
モデルの構造
NeRFは、主に以下の2つのネットワークを使用します：

Coarse Radiance Field: 初期の粗い推定を行うモデル
Fine Radiance Field: より詳細な推定を行うモデル

#学習プロセス

学習は以下のようなプロセスで進行します：
データロード: データセットからランダムにピクセルをサンプリングし、カメラの位置に基づいた光線（o, d）を生成。
モデルの推論: 入力光線をモデルに入力し、RGB値を推定。
損失関数の計算: 予測されたRGBと実際の画像のRGBを比較して損失を計算。
逆伝播: 損失に基づいてモデルのパラメータを更新。


