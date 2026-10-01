# Datasheet: Hindi Noun Paradigm Dataset

This datasheet follows the framework of Gebru et al. (2021), *Datasheets for
Datasets*, and reproduces Appendix C of the accompanying paper.

## Motivation

**For what purpose was the dataset created?**
To enable the first large-scale computational analysis of grammatical gender
assignment in Hindi. It provides high-quality annotations for inflectional
morphology, phonology, semantics and etymology, supporting research on
morphological processing and gender bias.

**Who created the dataset?** The authors of the paper (John Alderete, Yogya
Agrawal, Tra My Nguyen, Puja Shah, Aanchan Mohan).

**Who funded the creation of the dataset?** A Social Sciences and Humanities
Research Council of Canada grant (435-2020-0193) held by the first author.

## Composition

**What do the instances represent?** Each instance is a unique Hindi noun
paradigm (lemma).

**How many instances are there?** 6,040.

**Does the dataset contain all possible instances or is it a sample?** It is a
sample derived from the Hindi Treebank (Bhat et al., 2017), representing nouns
found in news and formal text. It does not cover the entire Hindi lexicon.

**What data does each instance consist of?** See [`README.md`](README.md) for
the full list of fields:

- Record: ID, lemma (script), stem (script and phonetic), English gloss.
- Morphological: gender (M/F), inflection class (0–5), derivational category
  (90 values), compound and derived status.
- Semantic: animacy, USAS category (coarse), HindiWordNet category (fine).
- Lexical stratum: Native, Sanskrit, Persian-Arabic or English.
- Phonological: character and phoneme n-grams of the stem.

**Are there recommended data splits?** No. The paper used a stratified 80/20
split, 5-fold stratified cross-validation, and a 70/15/15 split (neural
networks), all with random seed 42; the scripts in this repository recreate
them. The splits are not part of the dataset.

**Are there any known errors, sources of noise, or redundancies?** The USAS
semantic categories were assigned to machine-translated English glosses and may
contain noise from polysemy or translation errors. Animacy is highly redundant
with the HindiWordNet category (Cramér's V = 0.863).

**Is anything missing from individual instances?** Many treebank paradigms are
partial (not all four number × case cells are attested). The full paradigm can
be recovered from the inflection class label.

**Is the dataset self-contained?** Yes. It was derived using external resources
(the Hindi Treebank, HindiWordNet, the PyMUSAS tagger, McGregor's *Oxford
Hindi-English Dictionary*), but no external resources are needed to use it.

**Does the dataset contain data that might be offensive?** It contains terms
related to sensitive topics (e.g., crime, violence) that occur in news corpora,
presented as lexical entries without context.

**Does the dataset reflect social biases?** Yes. Animate-centric semantic
categories are strongly skewed toward masculine gender; for example, 89.6% of
Occupation nouns are masculine, compared with a 66.4% corpus baseline.

## Collection process

**How was the data collected?** The word list was extracted from the CoNLL-U
files of the Hindi Treebank (UD release). Etymological tags were extracted from
McGregor (1993) and supplemented by manual review for English loans. Semantic
categories were derived from HindiWordNet (CFILT, IIT Bombay) and from the
PyMUSAS tagger applied to English glosses (Google Cloud Translation). For the
1,389 nouns not found in HindiWordNet, common abbreviations were tagged
manually and the remainder were assigned an existing HindiWordNet category by a
large language model.

**Who was involved and how were they compensated?** Manual annotation and
adjudication were performed by two native Hindi speakers and a non-native data
analyst, all co-authors, recruited through Simon Fraser University job boards
and selected after two interview rounds. They were paid standard university
hourly rates and offered co-authorship contingent on their contributions to
model development and analysis.

**Over what timeframe was the data collected?** Over the year preceding the
paper's submission.

## Preprocessing, cleaning and labeling

Duplicate lemmas arising from encoding inconsistencies were resolved, and
lemmas occurring with more than one gender were adjudicated to distinguish true
homophones from gendered pairs (e.g., laṛkā 'boy' vs. laṛkī 'girl').
Inflection classes were annotated manually using a five-class schema derived
from published grammars, extended with a sixth class for English loan plurals.
Annotators co-annotated a 200-noun development set, formalized their decisions
in a rubric, annotated the remaining data independently, and jointly reviewed
ambiguous cases; a final consistency check reviewed all nouns in each
inflection class together. The code for feature encoding is in this repository.

## Uses

**Has the dataset been used already?** Yes, to train interpretable classifiers
and neural networks for gender and inflection-class prediction in the
accompanying paper.

**What other tasks could it be used for?** Training morphological analyzers,
improving gender agreement in machine translation and natural language
generation, probing multilingual representations for inflection class and
gender, and studying sociolinguistic bias in the Hindi lexicon.

**Are there tasks for which it should not be used?** The dataset uses a strict
binary grammatical gender. It should not be used as a normative standard to
enforce binary gender agreement where sensitivity to non-binary or gender-fluid
identities is needed. Generative models trained on it should use mitigation
strategies to avoid propagating the masculine skew of professional and
high-agency terms.

## Distribution and maintenance

**How is the dataset distributed?** Through this GitHub repository.

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

**Who maintains the dataset?** The first author, John Alderete
(alderete@sfu.ca).

**Will the dataset be updated?** Corrections and new instances may be added;
updates will be announced in this repository.
