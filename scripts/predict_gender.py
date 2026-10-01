"""Predict grammatical gender from each feature set (paper Table 5, left; Figure 1, left).

Four interpretable classifiers are trained on a stratified 80/20 split for each
of seven feature sets, and the best model per set is compared against the
majority baseline and against other sets with McNemar's test and bootstrap
AUC-ROC confidence intervals.

Output: results/gender_prediction.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler

from hindi_gender.data import RESULTS_DIR, load_dataset
from hindi_gender.evaluation import run_single_split, save_json, statistical_analysis
from hindi_gender.features import (
    GENDER_COMPARISONS, build_feature_classes, gender_feature_sets, gender_target, to_matrix,
)


def main():
    df = load_dataset()
    y = gender_target(df)
    feature_sets = gender_feature_sets(build_feature_classes(df))
    majority_baseline = max(np.mean(y), 1 - np.mean(y))
    print(f"{len(y)} nouns; majority baseline (masculine) = {majority_baseline:.4f}")

    results, predictions = {}, {}
    for name, frame in feature_sets.items():
        print(f"\n[{name}] {frame.shape[1]} features")
        X = StandardScaler().fit_transform(to_matrix(frame))
        results[name], predictions[name] = run_single_split(X, y, multiclass=False, scale_within_split=False)

    stats = statistical_analysis(predictions, majority_class=1, comparisons=GENDER_COMPARISONS)
    save_json(
        {"task": "gender", "evaluation": "stratified 80/20 split", "majority_baseline": majority_baseline,
         "results": results, "statistical_analysis": stats},
        RESULTS_DIR / "gender_prediction.json",
    )


if __name__ == "__main__":
    main()
