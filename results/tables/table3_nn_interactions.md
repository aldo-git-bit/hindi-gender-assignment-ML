# Table 3: Neural network models without the primary predictor, with pairwise interactions

| Interaction               | Gender Acc   | Gender F1   | Gender AUC   | Gender ΔAcc   | Inflection Acc   | Inflection Macro F1   | Inflection AUC   | Inflection ΔAcc   |
|:--------------------------|:-------------|:------------|:-------------|:--------------|:-----------------|:----------------------|:-----------------|:------------------|
| Baseline (no interaction) | 74.50%       | 83.35%      | 0.7414       | –             | 41.17%           | 35.85%                | 0.7611           | –                 |
| Etymology × Derivation    | 75.06%       | 83.60%      | 0.7503       | +0.55%        | 43.49%           | 37.21%                | 0.7635           | +2.32%            |
| Etymology × Phonology     | 74.28%       | 83.39%      | 0.7409       | -0.22%        | 44.70%           | 37.92%                | 0.7652           | +3.53%            |
| Etymology × Semantic      | 74.61%       | 83.41%      | 0.7341       | +0.11%        | 40.29%           | 35.05%                | 0.7700           | -0.88%            |
| Phonology × Semantic      | 72.30%       | 82.19%      | 0.7232       | -2.21%        | 37.75%           | 33.02%                | 0.7378           | -3.42%            |
| Derivation × Semantic     | 74.17%       | 83.33%      | 0.7411       | -0.33%        | 38.08%           | 34.75%                | 0.7608           | -3.09%            |
| Derivation × Phonology    | 74.06%       | 83.15%      | 0.7369       | -0.44%        | 41.94%           | 36.03%                | 0.7567           | +0.77%            |
