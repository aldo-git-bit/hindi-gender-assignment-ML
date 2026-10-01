"""Feature encoding for the interpretable (scikit-learn) models.

Each linguistic feature class is encoded as a numeric DataFrame:

- Morphological: m_compound, one-hot m_inflection (class 0 dropped), and
  one-hot m_deriv_cat_important (underived nouns are all-zero).
- Phonological: multi-hot encoding of the 160 selected script n-grams
  (p_ngrams_script_important).
- Semantic: one-hot USAS categories and HindiWordNet categories.
- Etymological: the three binary lexical-stratum flags (native = all zero).
"""

import re

import numpy as np
import pandas as pd

FEATURE_CLASSES = ("semantic", "phonological", "morphological", "etymological")


def morphological_features(df):
    out = pd.DataFrame({"m_compound": df["m_compound"]})
    inflection = pd.get_dummies(df["m_inflection"], prefix="m_inflection", drop_first=True, dtype=int)
    out = pd.concat([out, inflection], axis=1)
    deriv = df["m_deriv_cat_important"].fillna("none")
    for category in deriv.unique():
        if category != "none":
            out[f"m_deriv_{category}"] = (deriv == category).astype(int)
    return out


def phonological_features(df, column="p_ngrams_script_important"):
    ngram_sets = df[column].fillna("").map(lambda s: set(filter(None, s.split(","))))
    vocabulary = sorted(set().union(*ngram_sets))
    return pd.DataFrame(
        {f"p_ngram_{ng}": ngram_sets.map(lambda s, ng=ng: int(ng in s)) for ng in vocabulary},
        index=df.index,
    )


def semantic_features(df):
    columns = {}
    for source, prefix in (("s_category_usas", "s_usas"), ("s_category_wnet", "s_wnet")):
        values = df[source].fillna("unknown")
        for category in sorted(values.unique()):
            if category != "unknown":
                name = re.sub(r"[^a-zA-Z0-9_]", "_", str(category))
                columns[f"{prefix}_{name}"] = (values == category).astype(int)
    return pd.DataFrame(columns, index=df.index)


def etymological_features(df):
    return df[["e_english", "e_persian_arabic", "e_sanskrit"]].copy()


def build_feature_classes(df):
    """Return a dict mapping each feature class name to its encoded DataFrame."""
    return {
        "semantic": semantic_features(df),
        "phonological": phonological_features(df),
        "morphological": morphological_features(df),
        "etymological": etymological_features(df),
    }


def gender_target(df):
    """Binary gender target: F = 0, M = 1."""
    return (df["m_gender"] == "M").astype(int).to_numpy()


def inflection_columns(frame):
    return [c for c in frame.columns if c.startswith("m_inflection_")]


def gender_feature_sets(classes):
    """The seven feature sets used to predict gender (Table 5, left)."""
    morph = classes["morphological"]
    morph_no_infl = morph.drop(columns=inflection_columns(morph))
    sem, phon, etym = classes["semantic"], classes["phonological"], classes["etymological"]
    return {
        "semantic": sem,
        "phonological": phon,
        "morphological": morph,
        "etymological": etym,
        "morphology_minus_inflection": morph_no_infl,
        "all_features": pd.concat([sem, phon, morph, etym], axis=1),
        "all_features_minus_inflection": pd.concat([sem, phon, morph_no_infl, etym], axis=1),
    }


def inflection_feature_sets(classes, gender):
    """The seven feature sets used to predict inflection class (Table 5, right).

    Inflection class is the target, so it is removed from the morphological
    features and replaced by gender (F = 0, M = 1).
    """
    morph = classes["morphological"]
    morph_no_gender = morph.drop(columns=inflection_columns(morph))
    morph_with_gender = morph_no_gender.assign(m_gender=gender)
    sem, phon, etym = classes["semantic"], classes["phonological"], classes["etymological"]
    return {
        "semantic": sem,
        "phonological": phon,
        "morphological": morph_with_gender,
        "morphology_minus_gender": morph_no_gender,
        "etymological": etym,
        "all_features": pd.concat([sem, phon, morph_with_gender, etym], axis=1),
        "all_features_minus_gender": pd.concat([sem, phon, morph_no_gender, etym], axis=1),
    }


def gender_cv_feature_sets(df):
    """Feature sets used for the gender cross-validation experiment (Table 6, left).

    This encoding was used to produce the published cross-validation results and
    differs from ``build_feature_classes`` in two ways: raw columns are one-hot
    encoded directly with ``pd.get_dummies``, so each distinct *list* of
    important n-grams in ``p_ngrams_script_important`` becomes a single
    indicator column rather than one column per n-gram; and the column order
    of the morphological block differs. It is kept as-is so that Table 6 is
    reproducible.
    """
    groups = {"semantic": [], "phonological": [], "morphological": [], "etymological": []}
    for col in df.columns:
        if col in ("s_category_usas", "s_category_wnet"):
            groups["semantic"].append(col)
        elif col in ("e_english", "e_persian_arabic", "e_sanskrit"):
            groups["etymological"].append(col)
        elif col in ("m_inflection", "m_compound", "m_deriv_cat_important"):
            groups["morphological"].append(col)
        elif col == "p_ngrams_script_important":
            groups["phonological"].append(col)

    def encode(cols):
        frame = df[cols].copy()
        for col in cols:
            if col == "m_inflection":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            elif df[col].dtype == "object":
                dummies = pd.get_dummies(df[col], prefix=col)
            else:
                continue
            frame = pd.concat([frame.drop(columns=col), dummies], axis=1)
        return frame

    classes = {name: encode(cols) for name, cols in groups.items()}
    return gender_feature_sets(classes)


def to_matrix(frame):
    return frame.to_numpy(dtype=np.float64)


# Pairwise significance tests reported for each task: (set1, set2, bootstrap_auc).
GENDER_COMPARISONS = [
    ("phonological", "morphological", True),
    ("morphology_minus_inflection", "morphological", False),
    ("morphological", "all_features", True),
    ("all_features_minus_inflection", "all_features", False),
    ("phonological", "all_features_minus_inflection", True),
]

INFLECTION_COMPARISONS = [
    ("phonological", "morphological", False),
    ("morphology_minus_gender", "morphological", True),
    ("morphological", "all_features", True),
    ("morphological", "all_features_minus_gender", True),
    ("all_features_minus_gender", "all_features", False),
    ("phonological", "all_features_minus_gender", False),
]
