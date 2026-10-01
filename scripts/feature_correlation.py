"""Feature redundancy analysis (paper Section 4.2).

- Pearson correlation among binary features.
- Cramér's V among the nominal semantic features; animacy and HindiWordNet
  category are highly associated (V = 0.863), which motivated removing animacy.
- Supporting evidence: WordNet category purity with respect to animacy and the
  accuracy of predicting animacy from WordNet category alone.

Output: results/feature_correlation.json
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from scipy.stats.contingency import association

from hindi_gender.data import RESULTS_DIR, load_dataset
from hindi_gender.evaluation import save_json

BINARY = ["e_english", "e_persian_arabic", "e_sanskrit", "m_compound", "m_derived", "s_animacy"]
NOMINAL = ["s_animacy", "s_category_usas", "s_category_wnet"]


def main():
    df = load_dataset()
    binary = df[BINARY].assign(s_animacy=(df["s_animacy"] == "animate").astype(int))
    pearson = binary.corr()
    cramers_v = {
        f"{a} ~ {b}": association(pd.crosstab(df[a], df[b]), method="cramer")
        for a, b in combinations(NOMINAL, 2)
    }

    animacy_share = pd.crosstab(df["s_category_wnet"], df["s_animacy"], normalize="index")
    pure = ((animacy_share["animate"] > 0.95) | (animacy_share["inanimate"] > 0.95)).mean()
    majority_animacy = df.groupby("s_category_wnet")["s_animacy"].agg(lambda s: s.mode()[0])
    predicted = (df["s_category_wnet"].map(majority_animacy) == df["s_animacy"]).mean()

    print("Pearson correlation (binary features):")
    print(pearson.round(3).to_string())
    print("\nCramér's V (nominal features):")
    for pair, v in cramers_v.items():
        print(f"  {pair:40s} {v:.3f}")
    print(f"\nWordNet categories >95% animate or inanimate: {100 * pure:.1f}% "
          f"of {len(animacy_share)} categories")
    print(f"Animacy predicted from WordNet category: {100 * predicted:.1f}% accuracy")

    save_json(
        {"pearson_binary": pearson.round(4).to_dict(), "cramers_v": cramers_v,
         "wnet_categories": len(animacy_share), "wnet_animacy_purity": pure,
         "animacy_from_wnet_accuracy": predicted},
        RESULTS_DIR / "feature_correlation.json",
    )


if __name__ == "__main__":
    main()
