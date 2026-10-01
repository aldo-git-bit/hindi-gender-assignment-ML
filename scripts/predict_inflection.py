"""Predict inflection class (6-way) from each feature set (paper Table 5, right;
Figure 1, right).

Inflection class is the target, so gender takes its place among the
morphological features. Four interpretable classifiers are trained on a
stratified 80/20 split for each of seven feature sets; AUC-ROC is one-vs-rest,
macro-averaged.

Output: results/inflection_prediction.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from hindi_gender.data import RESULTS_DIR, load_dataset
from hindi_gender.evaluation import run_single_split, save_json, statistical_analysis
from hindi_gender.features import (
    INFLECTION_COMPARISONS, build_feature_classes, gender_target, inflection_feature_sets, to_matrix,
)


def main():
    df = load_dataset()
    y = df["m_inflection"].to_numpy()
    feature_sets = inflection_feature_sets(build_feature_classes(df), gender_target(df))
    classes, counts = np.unique(y, return_counts=True)
    majority_class = int(classes[np.argmax(counts)])
    majority_baseline = counts.max() / len(y)
    print(f"{len(y)} nouns; majority baseline (class {majority_class}) = {majority_baseline:.4f}")

    results, predictions = {}, {}
    for name, frame in feature_sets.items():
        print(f"\n[{name}] {frame.shape[1]} features")
        results[name], predictions[name] = run_single_split(
            to_matrix(frame), y, multiclass=True, scale_within_split=True
        )

    stats = statistical_analysis(predictions, majority_class, INFLECTION_COMPARISONS)
    save_json(
        {"task": "inflection_class", "evaluation": "stratified 80/20 split",
         "class_distribution": dict(zip(classes.tolist(), counts.tolist())),
         "majority_class": majority_class, "majority_baseline": majority_baseline,
         "results": results, "statistical_analysis": stats},
        RESULTS_DIR / "inflection_prediction.json",
    )


if __name__ == "__main__":
    main()
