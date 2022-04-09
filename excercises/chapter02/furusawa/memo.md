## Memo

- ベルマン方程式を解けるならそれを解けば最適な戦略を出すことができる
  - 遷移関数や報酬関数がわかっていればそれを解けば良い
  - Value Iterationを使ってVを直接計算すればよい
- 遷移関数や報酬関数が分からない場合はベルマン方程式を解くことはできない
  - 現実の多くの場合はそれらが分からない
- その場合は、Valueを直接計算することが出来ないので「学習」をして真の値を見つけていく（観察したVから真のVを学習する）
  - 学習をするためにはexploitだけではなくexploreもする必要がある（epsilon-greedy method）
  - 予測で学習するか実績で学習するか（TD method）
  - policy base(on-policy)かvalue base(off-policy)か
- 「戦略の学習」っていうと仰々しいけど、実質はValueとかQ値の学習ってイメージ

### 疑問に思ったところ

- off-policyって言うけど、「Valueを最大化するような行動を取るpolicy」じゃないの？
- 例えば将棋とかチェスみたいなゲームの場合、自分の行動→相手の行動→次の自分のstateが実現となるが、この状態の遷移を遷移関数で確率的な遷移と捉えて良いのか？
- Q-learningとSARSAの使い所みたいなのが分からない
  - off-policyとon-policyの違いはどこで重要になるのか？
  - on-policyの方が慎重ってこと？
- SARSAでは必ずしも`np.argmax(self.Q[next_state])`の必要はないよね？

### 勉強会のメモ
- 教科書の方は、SARSAのpolicyとしてepsilon-greedyを採用している
  - なので必ずしも`np.argmax(self.Q[next_state])`の必要はない
  - SARSA自体はpolicyの形を指定するものではない
- Q-learningとSARSAの違い
  - epsilonが0のときは両者は同じになる
  - SARSAは学習のときに、stateが変わった後に探索（=ベストじゃないかもしれない、、罰則をくらうかもしれない）するかもしれないことを考慮して行動を決める
  - Q-learningは、stateが変わった後にベストな戦略を取れると思って行動を決める
  - なのでイメージとしてはSARSAの方が慎重に戦略を更新していくような感じ
- Q-learningでalpha=1ならValue Iterationは同じ？？
  - Value Iterationは遷移関数や報酬関数が分かっているときに使える
- 罰則を大きくするとちゃんと学習できそう
  - 罰則を大きくすると将来の罰則を恐れて現時点からちゃんと行動するようなイメージ？
