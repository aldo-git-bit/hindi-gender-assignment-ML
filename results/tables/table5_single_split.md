# Table 5: Interpretable models, stratified 80/20 split

| Feature Set             | Model      | Gender Acc   | Gender F1   | Gender AUC   | Inflection Acc   | Inflection F1   | Inflection AUC   |
|:------------------------|:-----------|:-------------|:------------|:-------------|:-----------------|:----------------|:-----------------|
| Semantic                | LogReg     | 66.80%       | 52.12%      | 0.6178       | 54.97%           | 20.02%          | 0.6426           |
| Semantic                | RandForest | 67.88%       | 48.59%      | 0.6256       | 55.22%           | 17.86%          | 0.6489           |
| Semantic                | DecTree    | 67.80%       | 50.73%      | 0.6119       | 55.22%           | 19.21%          | 0.6008           |
| Semantic                | GradBoost  | 68.05%       | 50.44%      | 0.6224       | 54.55%           | 17.98%          | 0.6317           |
| Phonological            | LogReg     | 72.27%       | 62.13%      | 0.7165       | 57.78%           | 26.06%          | 0.6599           |
| Phonological            | RandForest | 71.27%       | 57.18%      | 0.6855       | 56.21%           | 19.54%          | 0.6440           |
| Phonological            | DecTree    | 70.53%       | 56.62%      | 0.6437       | 57.04%           | 24.42%          | 0.6204           |
| Phonological            | GradBoost  | 71.36%       | 61.38%      | 0.6793       | 57.62%           | 24.78%          | 0.6493           |
| Morphological           | LogReg     | 96.61%       | 96.12%      | 0.9835       | 72.35%           | 32.96%          | 0.7688           |
| Morphological           | RandForest | 96.61%       | 96.12%      | 0.9806       | 72.60%           | 30.75%          | 0.7713           |
| Morphological           | DecTree    | 96.61%       | 96.12%      | 0.9838       | 72.19%           | 32.67%          | 0.7698           |
| Morphological           | GradBoost  | 96.61%       | 96.12%      | 0.9834       | 72.19%           | 32.67%          | 0.7704           |
| Morphological (Ablated) | LogReg     | 69.45%       | 50.54%      | 0.5773       | 56.13%           | 18.68%          | 0.5751           |
| Morphological (Ablated) | RandForest | 69.29%       | 50.11%      | 0.5763       | 55.63%           | 17.76%          | 0.5731           |
| Morphological (Ablated) | DecTree    | 69.37%       | 50.32%      | 0.5553       | 55.96%           | 18.34%          | 0.5647           |
| Morphological (Ablated) | GradBoost  | 69.37%       | 50.32%      | 0.5763       | 55.96%           | 18.34%          | 0.5736           |
| Etymological            | LogReg     | 66.39%       | 39.90%      | 0.5503       | 53.64%           | 11.64%          | 0.6747           |
| Etymological            | RandForest | 66.39%       | 39.90%      | 0.5503       | 53.64%           | 11.64%          | 0.6732           |
| Etymological            | DecTree    | 66.39%       | 39.90%      | 0.5503       | 53.64%           | 11.64%          | 0.6732           |
| Etymological            | GradBoost  | 66.39%       | 39.90%      | 0.5503       | 53.64%           | 11.64%          | 0.6747           |
| All Features            | LogReg     | 96.69%       | 96.24%      | 0.9791       | 74.42%           | 45.68%          | 0.8642           |
| All Features            | RandForest | 96.52%       | 96.04%      | 0.9795       | 73.10%           | 35.91%          | 0.8779           |
| All Features            | DecTree    | 96.77%       | 96.33%      | 0.9795       | 74.17%           | 43.15%          | 0.8696           |
| All Features            | GradBoost  | 96.85%       | 96.42%      | 0.9856       | 75.91%           | 43.61%          | 0.8965           |
| All Features (Ablated)  | LogReg     | 74.75%       | 68.65%      | 0.7537       | 61.92%           | 35.98%          | 0.7708           |
| All Features (Ablated)  | RandForest | 73.10%       | 60.34%      | 0.7250       | 59.35%           | 26.43%          | 0.7759           |
| All Features (Ablated)  | DecTree    | 72.85%       | 63.64%      | 0.6926       | 60.02%           | 31.19%          | 0.7379           |
| All Features (Ablated)  | GradBoost  | 73.84%       | 64.73%      | 0.7269       | 62.09%           | 33.18%          | 0.7916           |
