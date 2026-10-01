# Hindi Noun Paradigm Dataset

`hindi_noun_paradigms.csv` contains 6,040 Hindi noun paradigms (one row per
lemma–gender pairing) extracted from the Hindi Treebank and annotated by native
speaker analysts for morphological, semantic, lexical-stratum and phonological
features. It is released under the
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
license (see [`LICENSE`](LICENSE)). The construction, annotation procedure and
quality control are described in Section 3 of the paper, and a datasheet is
provided in [`DATASHEET.md`](DATASHEET.md).

- **Encoding:** UTF-8 CSV. Hindi forms are in Devanagari; phonetic forms use IPA.
- **Gender:** 4,010 masculine (66.39%), 2,030 feminine (33.61%).
- **Inflection class:** 6 classes (Table 1 of the paper); class 4 is the most
  frequent (53.63%).
- **Missing values:** empty cells occur only in the `*_important` and `*_topk`
  columns, where they mean that no selected category or n-gram applies to the
  noun (e.g., an underived noun has no derivational category).

## Columns

| Column | Type | Values | Description |
|---|---|---|---|
| **Record** | | | |
| `r_id` | integer | – | Unique row identifier |
| `r_lemma_script` | string | – | Lemma (citation form) in Devanagari |
| `r_stem_script` | string | – | Stem in Devanagari, with inflectional endings removed from the singular direct form |
| `r_stem_phonetic` | string | – | Stem in IPA, normalized for schwa deletion |
| `r_gloss_english` | string | – | English gloss |
| `r_same_lemma_type` | categorical | 3 | Relationship to rows sharing the same lemma: `NotAppl` (lemma occurs once), `neutral` (one lemma used with both genders, listed once per gender, e.g. अधिकारी 'officer'; the "gendered pairs" of Table 2), `homophones` (distinct words spelled alike, e.g. कक्षा 'class') |
| **Morphological** | | | |
| `m_gender` | categorical | `M`, `F` | Grammatical gender |
| `m_inflection` | categorical | `0`–`5` | Inflection class: 1–4 are the declinable classes of Table 1, 0 is zero inflection (indeclinable), 5 is English loans with English plural morphology |
| `m_deriv_cat` | categorical | 90 | Derivational category, or `underived` |
| `m_deriv_cat_important` | categorical | 17 | Derivational categories retained by Wilson-interval feature selection (Section 4.3); empty otherwise |
| `m_deriv_cat_topk` | categorical | 10 | The 10 most frequent derivational categories; empty otherwise |
| `m_compound` | binary | `0`, `1` | Compound noun |
| `m_derived` | binary | `0`, `1` | Derived noun |
| **Semantic** | | | |
| `s_animacy` | binary | `animate`, `inanimate` | Animacy |
| `s_category_usas` | categorical | 22 | Top-level UCREL USAS semantic domain (a single letter, e.g. `F` = Food and Drink), assigned by PyMUSAS to the English gloss |
| `s_category_wnet` | categorical | 108 | Highest-level HindiWordNet hypernym (e.g. `Person`, `Occupation`, `Abstract`) |
| **Lexical stratum (etymology)** | | | |
| `e_english` | binary | `0`, `1` | English loanword |
| `e_persian_arabic` | binary | `0`, `1` | Persian or Arabic loanword |
| `e_sanskrit` | binary | `0`, `1` | Sanskrit loanword |
| | | | Nouns with all three flags `0` are *Native*. The flags are mutually exclusive. |
| **Phonological** (comma-separated lists of n-grams of length 1–3 from the stem) | | | |
| `p_ngrams_script` | multi-category | 5,385 | All script n-grams |
| `p_ngrams_phonetic` | multi-category | 4,265 | All phonetic n-grams |
| `p_ngrams_script_important` | multi-category | 160 | Script n-grams retained by Wilson-interval feature selection; used by the interpretable models |
| `p_ngrams_phonetic_important` | multi-category | 151 | Phonetic n-grams retained by Wilson-interval feature selection |
| `p_ngrams_script_topk` | multi-category | 20 | The 20 most frequent script n-grams; used by the neural networks |
| `p_ngrams_phonetic_topk` | multi-category | 20 | The 20 most frequent phonetic n-grams |

The `Values` column gives the number of distinct values for categorical and
multi-category columns.

## Feature selection

For the `*_important` columns, each n-gram or derivational category was kept
only if the 95% Wilson confidence interval of its masculine proportion excluded
the corpus baseline of 66.4% masculine (paper Section 4.3).

## Loading the data

```python
import pandas as pd

df = pd.read_csv("data/hindi_noun_paradigms.csv")
ngrams = df["p_ngrams_script_important"].fillna("").str.split(",")
```
