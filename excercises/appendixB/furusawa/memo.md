## メモ

- "The goal of machine Learning is generalization"
  - inputとoutputの関係を良く近似する関数を作る
- 機械学習の問題
  - "Curse of Dimensionality"
    - 変数の数が多くなるになるにつれて指数関数的に解空間の次元が大きくなる
    - 解空間の次元が大きくなると計算量が大きくなりまくる
  - "Bias-Variance Trade-Off"
    - モデルの変数の数に対して観測データが少ないとoverfittingが発生しvarianceが大きくなる
    - 観測データが少ないとunderfittingが発生しbiasが大きくなる
- Deep Learning
  - 画像等を扱う場合、変数の数が極端に増えるので上記の問題が発生しやすく、線形回帰は難しい
  - 層を重ねるDeep Learningの手法によってこの問題を軽減できる
  - 問題に応じた様々なNetwork Architectureが考案される
    - Convolutional Networks
    - Recurrent Networks
    - Residual Networks
    - Generative Adversarial Networks
    - Autoencoders
    - etc...
  - ただしDeep Learningでもoverfittingは発生しがち
    - 様々な軽減手法
      - Data Augmentation
      - Capacity Reduction
      - Dropout
      - L1正則化・L2正則化
      - Early Stoppong
      - バッチ正則化

### コードかいてるときのメモ
- M1 MacではDocker上でTensorflowを回すことが出来ない
  - Tensorflow（というかPyPI）がAarch64に対応していない
- なので誰かが作ったパッケージをpipでインストールした

参照
- [M1 Mac + Dockerの環境でTensorflowをビルドする](https://qiita.com/tsukushibito/items/a5384e920c8ce6cc99fd)
- [M1搭載Macの環境を汚さずにDeep Learningしたい！](https://qiita.com/sonoisa/items/6d6b4a81169397a96dd8)
- [M1 MacのDocker上でTensorflowを入れる方法](https://note.com/naoki_official/n/n401891d27081)

### 勉強会のメモ

- CPUでもGPUでも実行速度あまり変わらなかった
  - mnistが人気なのはCPUでも結果が出るから？
- 畳み込みネットワークのfilterってなに？？
  - 複数のfilterを用意することによって情報量を下げすぎないようにする
  - filter自体もパラメータなので更新していく
    - ということは初期値に寄らないということ？
  - 参照：[kerasのConv2D（2次元畳み込み層）について調べてみた](https://qiita.com/kenichiro-yamato/items/60affeb7ca9f67c87a17)