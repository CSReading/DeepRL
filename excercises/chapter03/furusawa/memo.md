## Memo

- Q Learningは状態と行動の数が大きくなると辛くなる
  - 例えばAtari Gameだと状態の数は256^33600くらいになる
  - これをそのままQ Learningすることは不可能
- Deep Q Networkの登場
  - Q値の算出をdeep learningっぽく行う
- しかしこれにも３つの難点がある
  - Converge: state spaceが大きいと最適なQ関数に収束しないかもしれない
  - Correlation: 行動の時系列に相関があり、トレーニングデータに偏りが生まれてしまう
  - Convergence: deep learningのoutputは変化するので収束しないかもしれない
- 解決策
  - Experience Replay: 観測したinputデータを保存しておきそれを使って学習する
    - →Correlation問題の解消
  - Infrequent Updates of Target Weights: 毎回weightをupdateするのではなく、何回かごとにupdateする
    - →Convergence問題の解消

### 思ったこと
- Experience Replayのモチベーションとして、stateのpathが相関を生まれてしまうことを挙げているが、これってDeepじゃなくても発生する問題では？
  - state spaceが小さいと十分に探索できるのでそんなに大した問題では無い？
- デフォルトの学習率が0.0001ってめっちゃ小さくない？
  - だから学習遅い？なんでこの設定にしている？

### 注意点
- stable-baselineパッケージはメンテナンスモードになっており、最新環境に対応していない。そのためstable-baseline3を使う
- Google Colabで学習するとき、buffer sizeが大きいと使用メモリが制限を超えてしまうので注意
  - 100000くらいならいける

### 勉強会メモ
- total time stepが300万回くらいじゃないとちゃんと学習できないっぽい
- クラウドを使わないとちゃんと学習出来ない（ローカルでやるとメモリが死ぬ）
- DeepでExperience Replayを使う理由としてはDeepだとsparseになりやすいから？

### 参照
- [【深層強化学習,入門】Deep Q Network(DQN)の解説とPythonで実装　〜図を使って説明〜](https://www.tcom242242.net/entry/ai-2/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92/%E6%B7%B1%E5%B1%A4%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92/%E3%80%90%E6%B7%B1%E5%B1%A4%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%80%91deep_q_network_%E3%82%92tensorflow%E3%81%A7%E5%AE%9F%E8%A3%85/)
- [DQN（Deep Q Network）を理解したので、Gopherくんの図を使って説明](https://qiita.com/ishizakiiii/items/5eff79b59bce74fdca0d)
- [DQNの進化史 ①DeepMindのDQN](https://horomary.hatenablog.com/entry/2021/01/26/233351)