"""Five-fold stratified cross-validation of gender prediction (paper Table 6, left).

Significance tests use the out-of-fold predictions concatenated over all folds.
Features are encoded with ``gender_cv_feature_sets``, the encoding used for the
published cross-validation results (see its docstring and the README).

Output: results/gender_prediction_cv.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler

from hindi_gender.data import RESULTS_DIR, load_dataset
from hindi_gender.evaluation import run_cross_validation, save_json, statistical_analysis
from hindi_gender.features import GENDER_COMPARISONS, gender_cv_feature_sets, gender_target, to_matrix


def main():
    df = load_dataset()
    y = gender_target(df)
    feature_sets = gender_cv_feature_sets(df)
    majority_baseline = max(np.mean(y), 1 - np.mean(y))

    results, predictions = {}, {}
    for name, frame in feature_sets.items():
        print(f"\n[{name}] {frame.shape[1]} features")
        X = StandardScaler().fit_transform(to_matrix(frame))
        results[name], predictions[name] = run_cross_validation(X, y, multiclass=False, scale_within_fold=False)

    stats = statistical_analysis(predictions, majority_class=1, comparisons=GENDER_COMPARISONS)
    save_json(
        {"task": "gender", "evaluation": "5-fold stratified cross-validation",
         "majority_baseline": majority_baseline, "results": results, "statistical_analysis": stats},
        RESULTS_DIR / "gender_prediction_cv.json",
    )


if __name__ == "__main__":
    main()
