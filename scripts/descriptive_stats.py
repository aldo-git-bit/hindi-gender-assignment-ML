"""Dataset overview (paper Table 2).

Output: results/tables/table2_dataset_overview.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from hindi_gender.data import RESULTS_DIR, load_dataset


def pct(n, total):
    return f"{100 * n / total:.2f}%"


def main():
    df = load_dataset()
    n = len(df)
    masc = df["m_gender"] == "M"

    overview = pd.DataFrame(
        [
            ("Total nouns", n),
            ("Fully declinable (classes 1-4)", df["m_inflection"].isin([1, 2, 3, 4]).sum()),
            ("Zero/Irregular (classes 0, 5)", df["m_inflection"].isin([0, 5]).sum()),
            ("Compounds", (df["m_compound"] == 1).sum()),
            ("Derived nouns", (df["m_derived"] == 1).sum()),
            ("Animate nouns", (df["s_animacy"] == "animate").sum()),
            ("Nouns in gendered pairs", (df["r_same_lemma_type"] == "neutral").sum()),
        ],
        columns=["Category", "Count"],
    )
    overview["% of Total"] = overview["Count"].map(lambda c: pct(c, n))

    gender = pd.DataFrame(
        [("Masculine", masc.sum()), ("Feminine", (~masc).sum())], columns=["Gender", "Count"]
    )
    gender["% of Total"] = gender["Count"].map(lambda c: pct(c, n))

    def by_group(mask_by_label):
        rows = [(label, mask.sum(), pct((mask & masc).sum(), mask.sum()), pct(mask.sum(), n))
                for label, mask in mask_by_label.items()]
        return rows

    labels = {1: "1", 2: "2", 3: "3", 4: "4", 0: "Zero inflection (class 0)", 5: "Irregular (class 5)"}
    inflection = pd.DataFrame(
        by_group({labels[c]: df["m_inflection"] == c for c in labels}),
        columns=["Inflection Class", "Count", "% Masc.", "% Total"],
    )

    no_loan = (df[["e_english", "e_persian_arabic", "e_sanskrit"]] == 0).all(axis=1)
    etymology = pd.DataFrame(
        by_group({
            "Arabic-Persian": df["e_persian_arabic"] == 1,
            "English": df["e_english"] == 1,
            "Sanskrit": df["e_sanskrit"] == 1,
            "Native": no_loan,
        }),
        columns=["Etymology", "Count", "% Masc.", "% Total"],
    )

    tables = [overview, gender, inflection, etymology]
    text = "\n\n".join(t.to_markdown(index=False) for t in tables)
    print(text)
    out = RESULTS_DIR / "tables" / "table2_dataset_overview.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Table 2: Dataset overview\n\n" + text + "\n", encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
