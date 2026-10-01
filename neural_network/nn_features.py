"""Input encoding for the neural network models.

- Categorical features (derivational category, USAS and HindiWordNet
  categories) are integer-encoded for embedding layers.
- Binary features: m_compound and four mutually exclusive lexical-stratum
  flags (native, English, Persian-Arabic, Sanskrit).
- Phonology: multi-hot encoding of the 20 most frequent script n-grams
  (p_ngrams_script_topk).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL = ("m_deriv_cat_important", "s_category_usas", "s_category_wnet")
PHONOLOGY = "p_ngrams_script_topk"


def multi_hot(column):
    ngram_lists = column.fillna("").map(lambda s: [ng.strip() for ng in s.split(",") if ng.strip()])
    vocabulary = sorted(set().union(*ngram_lists))
    index = {ng: i for i, ng in enumerate(vocabulary)}
    matrix = np.zeros((len(column), len(vocabulary)), dtype=np.float32)
    for row, ngrams in enumerate(ngram_lists):
        for ng in ngrams:
            matrix[row, index[ng]] = 1.0
    return matrix


def encode_features(df):
    """Return (categorical, vocab_sizes, binary, phonology) arrays for ``df``."""
    df = df.copy()
    df["m_deriv_cat_important"] = df["m_deriv_cat_important"].fillna("underived")

    categorical, vocab_sizes = {}, {}
    for col in CATEGORICAL:
        encoder = LabelEncoder()
        categorical[col] = encoder.fit_transform(df[col].astype(str))
        vocab_sizes[col] = len(encoder.classes_)

    loans = ["e_english", "e_persian_arabic", "e_sanskrit"]
    native = (df[loans] == 0).all(axis=1)
    binary = pd.concat([df["m_compound"], native.rename("e_native"), df[loans]], axis=1).to_numpy(np.float32)

    return categorical, vocab_sizes, binary, multi_hot(df[PHONOLOGY])
