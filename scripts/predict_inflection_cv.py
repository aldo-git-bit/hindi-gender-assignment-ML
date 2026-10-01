"""Five-fold stratified cross-validation of inflection class prediction
(paper Table 6, right).

Cross-validation places all 60 class-5 nouns in a test fold once, instead of
the ~12 available in a single 80/20 split. Significance tests use the
out-of-fold predictions concatenated over all folds.

Output: results/inflection_prediction_cv.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from hindi_gender.data import RESULTS_DIR, load_dataset
from hindi_gender.evaluation import run_cross_validation, save_json, statistical_analysis
from hindi_gender.features import (
    INFLECTION_COMPARISONS, build_feature_classes, gender_target, inflection_feature_sets, to_matrix,
)


def main():
    df = load_dataset()
    y = df["m_inflection"].to_numpy()
    feature_sets = inflection_feature_sets(build_feature_classes(df), gender_target(df))
    classes, counts = np.unique(y, return_counts=True)
    majority_class = int(classes[np.argmax(counts)])

    results, predictions = {}, {}
    for name, frame in feature_sets.items():
        print(f"\n[{name}] {frame.shape[1]} features")
        results[name], predictions[name] = run_cross_validation(
            to_matrix(frame), y, multiclass=True, scale_within_fold=True
        )

    stats = statistical_analysis(predictions, majority_class, INFLECTION_COMPARISONS)
    save_json(
        {"task": "inflection_class", "evaluation": "5-fold stratified cross-validation",
         "majority_class": majority_class, "majority_baseline": counts.max() / len(y),
         "results": results, "statistical_analysis": stats},
        RESULTS_DIR / "inflection_prediction_cv.json",
    )


if __name__ == "__main__":
    main()
