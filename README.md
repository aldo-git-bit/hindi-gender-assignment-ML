# Hindi Gender Assignment: Inflection Class vs. Surface Cues

Code and data for the paper:

> John Alderete, Yogya Agrawal, Tra My Nguyen, Puja Shah, and Aanchan Mohan.
> **Paradigms over Surface Cues: Inflection Class Drives Hindi Gender Assignment
> in Expert-Annotated Noun Paradigms.** Accepted at the 6th Multilingual
> Representation Learning (MRL) Workshop 2026, co-located with EMNLP, Budapest,
> Hungary. *Proceedings to appear shortly.*

The repository contains the **Hindi Noun Paradigm Dataset**, 6,040 Hindi nouns
annotated by native speakers for inflection class, gender, derivational
morphology, semantic category, lexical stratum and stem phonology, together with
the code that produces every table, figure and statistic reported in the paper.

## Overview

We ask which properties of a Hindi noun predict its grammatical gender. Four
interpretable classifiers (logistic regression, decision tree, random forest,
gradient boosting) are trained on seven feature sets built from four linguistic
feature classes, and the interdependence of gender and inflection class is
probed with neural networks that model pairwise feature interactions.

Main findings:

- **Inflection class dominates.** Feature sets containing inflection class
  predict gender at about 97% accuracy (AUC ≈ 0.98). Adding phonological,
  semantic and etymological features brings no significant improvement
  (McNemar's test, p > 0.44).
- **Surface cues are weak.** Without inflection class, accuracy falls to about
  73%. Stem phonology is the strongest remaining signal (≈71%); semantic and
  etymological features barely exceed the 66.4% majority baseline.
- **The dependence is asymmetric.** Inflection class nearly determines gender,
  but gender only partially predicts inflection class (≈74% vs. a 53.6%
  baseline). Inflection class is partly recoverable from phonology and
  etymology without gender, and feature interactions help inflection-class
  prediction much more than gender prediction.
- **Lexical gender bias.** Animate-centric semantic categories are 84.2%
  masculine, against the 66.4% corpus baseline.

## Repository structure

```
├── data/
│   ├── hindi_noun_paradigms.csv    # the dataset (CC BY 4.0)
│   ├── README.md                   # column-by-column data dictionary
│   ├── DATASHEET.md                # datasheet (paper Appendix C)
│   └── LICENSE                     # CC BY 4.0
├── src/hindi_gender/               # shared code
│   ├── data.py                     #   paths and dataset loading
│   ├── features.py                 #   feature encoding and feature sets
│   ├── models.py                   #   classifiers with fixed hyperparameters
│   └── evaluation.py               #   metrics, significance tests, experiment runners
├── scripts/                        # one script per analysis in the paper (see below)
├── neural_network/
│   ├── nn_features.py              # input encoding for the neural networks
│   └── train_interaction.py        # MLP with optional pairwise interaction module
├── results/                        # outputs of the scripts, as reported in the paper
│   ├── *.json, *.csv               #   metrics and statistical tests
│   ├── neural_network/             #   test metrics for the 14 neural network models
│   ├── tables/                     #   Tables 2, 3, 5 and 6 in Markdown
│   └── figures/                    #   Figure 1
├── Makefile                        # runs the full pipeline
├── requirements.txt                # dependencies for everything except the neural networks
├── requirements-nn.txt             # adds TensorFlow for the neural networks
├── CITATION.cff
└── LICENSE                         # MIT (code)
```

## Installation

Python 3.11 is recommended (the results were produced with Python 3.11 on
macOS).

```bash
git clone https://github.com/aldo-git-bit/hindi-gender-assignment-ML.git
cd hindi-gender-assignment-ML
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt       # interpretable models and analyses
pip install -r requirements-nn.txt    # optional: neural networks (TensorFlow)
```

## Reproducing the paper

All scripts are run from the repository root and write to `results/`. The
results already in `results/` are the ones reported in the paper.

```bash
make all           # analyses, interpretable models, cross-validation, figure and tables
make grid-search   # Appendix B grid search (540 random forests)
make nn            # the 14 neural network models of Table 3
```

On a laptop CPU (Apple M4 Pro), `make all` takes about 10 minutes and
`make grid-search` about 18 minutes.

| Paper | Script | Output |
|---|---|---|
| Table 2 (dataset overview) | `scripts/descriptive_stats.py` | `results/tables/table2_dataset_overview.md` |
| §3.2.2, §6.3 (semantic gender bias) | `scripts/semantic_gender_bias.py` | `results/semantic_gender_bias.{csv,json}` |
| §4.2 (feature correlation) | `scripts/feature_correlation.py` | `results/feature_correlation.json` |
| Table 5, left; Figure 1, left; §5.1 | `scripts/predict_gender.py` | `results/gender_prediction.json` |
| Table 5, right; Figure 1, right; §5.2 | `scripts/predict_inflection.py` | `results/inflection_prediction.json` |
| Table 6 (5-fold cross-validation) | `scripts/predict_gender_cv.py`, `scripts/predict_inflection_cv.py` | `results/*_prediction_cv.json` |
| §5.1–5.2 (feature importance) | `scripts/feature_importance.py` | `results/feature_importance.json` |
| Appendix B, Table 4 (grid search) | `scripts/grid_search_semantic_rf.py` | `results/semantic_rf_grid_search.json` |
| Table 3 (neural network interactions) | `neural_network/train_interaction.py` | `results/neural_network/<task>_<interaction>.json` |
| Figure 1 | `scripts/make_figure.py` | `results/figures/figure1_accuracy.{pdf,png}` |
| Tables 3, 5, 6 | `scripts/make_tables.py` | `results/tables/` |

A single neural network model is trained with, for example:

```bash
python neural_network/train_interaction.py --task inflection --interaction etym_phon
```

where `--task` is `gender` or `inflection` and `--interaction` is one of
`none`, `etym_deriv`, `etym_phon`, `etym_sem`, `phon_sem`, `deriv_sem`,
`deriv_phon`.

### Experimental setup

- **Feature sets.** Semantic (USAS and HindiWordNet categories, one-hot),
  phonological (160 selected stem n-grams, multi-hot), morphological
  (inflection class or gender, derivational category, compound status),
  etymological (lexical-stratum flags), all features combined, and "ablated"
  versions of the morphological and combined sets without the primary
  predictor (inflection class for gender; gender for inflection class).
- **Interpretable models** use fixed hyperparameters for every feature set
  (`src/hindi_gender/models.py`): logistic regression (C = 0.1), random forest
  (100 trees, max depth 15, min samples per leaf 5), decision tree (max depth
  10, min samples per leaf 5) and gradient boosting (100 trees, max depth 5,
  min samples per leaf 5, learning rate 0.05).
- **Evaluation.** Stratified 80/20 split and 5-fold stratified
  cross-validation, random seed 42. Metrics are accuracy, macro F1 and AUC-ROC
  (one-vs-rest, macro-averaged for inflection class). Feature sets are compared
  with McNemar's test and bootstrap (1,000 resamples) confidence intervals for
  AUC differences.
- **Neural networks** embed the categorical features, add an optional outer
  product of two feature groups compressed by a dense layer, and use three
  hidden layers (128-64-32, batch normalization, ReLU, dropout 0.3/0.3/0.2).
  They are trained with Adam (learning rate 10⁻³, halved when validation loss
  has not improved for 15 epochs, minimum 10⁻⁶), batch size 64, class-weighted
  loss and early stopping (patience 30) on validation accuracy (gender) or
  validation macro F1 (inflection class), on a stratified 70/15/15 split.

### Notes on reproducibility

The code is the code that produced the published numbers. Rerunning it
reproduces Tables 2, 5 and 6, Figure 1, Appendix B and the statistics in
Sections 3–5 exactly (up to rounding in the last digit). Some details of the implementation are worth stating
explicitly:

- **Cross-validation features for gender (Table 6, left).** The gender
  cross-validation script one-hot encodes the raw `p_ngrams_script_important`
  strings, so each distinct *combination* of n-grams is one indicator rather
  than each n-gram being its own feature (`gender_cv_feature_sets` in
  `src/hindi_gender/features.py`). This mainly affects the phonological and
  ablated all-features rows of Table 6, which are therefore lower than the
  corresponding single-split results in Table 5. The inflection
  cross-validation uses the same encoding as the single-split experiments.
- **Neural network optimization.** The training configuration is as listed
  above: the learning-rate schedule halves the rate after 15 epochs without
  improvement in validation loss, and there is no learning-rate warmup or
  weight decay.
- **Class weights for gender (neural networks).** The binary class weights in
  `train_interaction.py` give the larger weight to the majority (masculine)
  class, i.e. the reverse of inverse-frequency weighting. Inflection-class
  models use scikit-learn's balanced (inverse-frequency) weights.
- **Neural network variance.** The Table 3 models were trained without
  deterministic TensorFlow kernels, and retraining them yields accuracies that
  differ from the published values by up to about one percentage point. The
  script now enables deterministic operations, so repeated runs on the same
  machine give identical results. The published metrics are provided in
  `results/neural_network/`.
- **Feature importance.** The gender feature-class shares (§5.1) come from a
  random forest with scikit-learn default settings; the inflection-class
  importances (§5.2) come from the regularized random forest of Table 5.

## Citation

If you use the code or the dataset, please cite:

```bibtex
@inproceedings{alderete-etal-2026-paradigms,
  title     = {Paradigms over Surface Cues: Inflection Class Drives {H}indi Gender Assignment in Expert-Annotated Noun Paradigms},
  author    = {Alderete, John and Agrawal, Yogya and Nguyen, Tra My and Shah, Puja and Mohan, Aanchan},
  booktitle = {Proceedings of the 6th Workshop on Multilingual Representation Learning (MRL 2026)},
  address   = {Budapest, Hungary},
  year      = {2026},
  note      = {To appear}
}
```

## License

- **Code:** [MIT License](LICENSE).
- **Data:** [Creative Commons Attribution 4.0 International](data/LICENSE)
  (CC BY 4.0).

Both licenses allow reuse for any purpose, provided the authors are credited.

## Acknowledgments

We thank Rajesh Bhatt for his comments and questions and Joti Balani for
organizational support. This research was supported by a Social Sciences and
Humanities Research Council of Canada grant (435-2020-0193). Some of the
experimental analysis scripts were developed with the assistance of Claude Code
(Anthropic); all code and results were verified by the authors.

## Contact

John Alderete, Simon Fraser University — [alderete@sfu.ca](mailto:alderete@sfu.ca)
