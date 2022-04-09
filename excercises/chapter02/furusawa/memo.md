## Memo

- ベルマン方程式を解けるならそれを解けば最適な戦略を出すことができる
  - 遷移関数や報酬関数がわかっていればそれを解けば良い
  - Value Iterationを使ってVを直接計算すればよい
- 遷移関数や報酬関数が分からない場合はベルマン方程式を解くことはできない
  - 現実の多くの場合はそれらが分からない
- その場合は、Valueを直接計算することが出来ないので「学習」をして真の値を見つけていく（観察したVから真のVを学習する）
  - 学習をするためにはexploitだけではなくexploreもする必要がある（epsilon-greedy method）
  - 予測で学習するか実績で学習するか（Monte-Carlo method - TD method）
  - policy base(on-policy)かvalue base(off-policy)か
- 「戦略の学習」っていうと仰々しいけど、実質はValueとかQ値の学習ってイメージ

### 疑問に思ったところ

- off-policyって言うけど、「Valueを最大化するような行動を取るpolicy」じゃないの？
- 例えば将棋とかチェスみたいなゲームの場合、自分の行動→相手の行動→次の自分のstateが実現となるが、この状態の遷移を遷移関数で確率的な遷移と捉えて良いのか？
- Q-learningとSARSAの使い所みたいなのが分からない
  - off-policyとon-policyの違いはどこで重要になるのか？
  - on-policyの方が慎重ってこと？
- SARSAでは必ずしも`np.argmax(self.Q[next_state])`の必要はないよね？
