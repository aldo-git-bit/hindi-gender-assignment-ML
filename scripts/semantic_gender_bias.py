"""Masculine skew by HindiWordNet semantic category (paper Sections 3.2.2 and 6.3).

For each category, a 95% Wilson confidence interval for the proportion of
masculine nouns is compared with the corpus baseline (66.4%). "Animate-centric"
categories are those with more than half animate nouns (n >= 5); together they
are 84.2% masculine, and three (Person, Occupation, Mythological Character)
deviate significantly from the baseline.

Output: results/semantic_gender_bias.csv, results/semantic_gender_bias.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from hindi_gender.data import RESULTS_DIR, load_dataset
from hindi_gender.evaluation import save_json

MIN_N_ALL = 10       # minimum category size for the full table
MIN_N_ANIMATE = 5    # minimum category size for the animate-centric analysis


def main():
    df = load_dataset()
    baseline = (df["m_gender"] == "M").mean()

    rows = []
    for category, group in df.groupby("s_category_wnet"):
        n, k = len(group), int((group["m_gender"] == "M").sum())
        low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
        rows.append({
            "category": category, "n": n, "n_masculine": k, "pct_masculine": 100 * k / n,
            "ci_low": 100 * low, "ci_high": 100 * high,
            "pct_animate": 100 * (group["s_animacy"] == "animate").mean(),
            "significant": not (low <= baseline <= high),
        })
    table = pd.DataFrame(rows).sort_values("pct_masculine", ascending=False)

    animate = table[(table["n"] >= MIN_N_ANIMATE) & (table["pct_animate"] >= 50)]
    animate_nouns = df[df["s_category_wnet"].isin(animate["category"])]
    animate_pct = 100 * (animate_nouns["m_gender"] == "M").mean()

    print(f"Baseline: {100 * baseline:.2f}% masculine")
    print(f"\nAnimate-centric categories ({len(animate)}, n = {len(animate_nouns)}): "
          f"{animate_pct:.1f}% masculine")
    print(animate[["category", "n", "pct_animate", "pct_masculine", "significant"]]
          .round(1).to_string(index=False))

    reported = table[table["n"] >= MIN_N_ALL]
    print(f"\nCategories with n >= {MIN_N_ALL}: {len(reported)}, "
          f"{int(reported['significant'].sum())} deviate significantly from the baseline")

    out = RESULTS_DIR / "semantic_gender_bias.csv"
    reported.round(2).to_csv(out, index=False)
    print(f"Saved {out}")
    save_json(
        {"baseline_pct_masculine": 100 * baseline,
         "animate_centric": {"n_categories": len(animate), "n_nouns": len(animate_nouns),
                             "pct_masculine": animate_pct,
                             "categories": animate.round(2).to_dict(orient="records")}},
        RESULTS_DIR / "semantic_gender_bias.json",
    )


if __name__ == "__main__":
    main()
