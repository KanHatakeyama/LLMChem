# 分子の構造(Q)ー融点(A)データセットから説明(Reason: R)を自動生成して､説明付きで予測

# 情報
- オープンに進めています

# Setup
- git clone https://github.com/KanHatakeyama/LLMChem

# 関連Note(上から順にnew)
- [なぜ分子の融点が◯◯℃なのかをGPT-4に考えさせる際の試行錯誤メモ ](https://note.com/kan_hatakeyama/n/n84c84da8f551)
- [Explainableな構造ー物性の予測LLMモデルを作る研究の「目標とTODO」メモ](https://note.com/kan_hatakeyama/n/n56afe0df282a)
- [Explainableな構造ー物性データセットをLLMで自動生成する(定量的な説明ver) ](https://note.com/kan_hatakeyama/n/ndcdeaed60f48)
- [explainableな構造ー物性相関のLLM予測モデルのデータセットの自動生成 ](https://note.com/kan_hatakeyama/n/n8e5506240630)

- ![](contents/scheme.png)

# 結果の例
― [codeはこちら](https://github.com/KanHatakeyama/LLMChem/tree/20231216pub)
- 条件
  - ランダムに選択した10分子で比較
  - GPT3.5で理由生成&回答
  - ランダム性を考慮して､予測は独立に3回
  - 単一の予測において､範囲が10-20のように示された場合､10,20にplot
- 結果
  - オリジナルのモデル
    - "回答不能"な結果が帰ってきたデータが過半数を占めたので､そもそもプロットが少ない
    - 精度も微妙
    - ![](contents/wo_reason.png)

  - Rを生成して回答させたモデル
    - 予測精度が向上
    - ![](contents/w_reason.png)

# 研究進捗
- 2023/12/16
  - フレームを作る
- 12/24
  - 再帰などを実装
  - 融点データセットの計算を開始 (現在: 約300件/2.5万件)

# TODO
  - プロンプトチューニング
  - ファインチューニング
