"""Exhaustive grid search for the Semantic / Random Forest configuration
(paper Appendix B, Table 4).

Compares the fixed configuration used throughout the paper against the best of
540 hyperparameter combinations on the same stratified 80/20 gender split, to
show that tuning changes accuracy by less than one percentage point.

Output: results/semantic_rf_grid_search.json
"""

import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from hindi_gender.data import RANDOM_STATE, RESULTS_DIR, load_dataset
from hindi_gender.evaluation import save_json, scores
from hindi_gender.features import gender_target, semantic_features, to_matrix

PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [10, 15, 20, 30, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
}

FIXED_PARAMS = {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5,
                "min_samples_split": 2, "max_features": "sqrt"}


def evaluate(params, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(**params, random_state=RANDOM_STATE).fit(X_train, y_train)
    return scores(model, X_test, y_test, multiclass=False)[0]


def main():
    df = load_dataset()
    X, y = to_matrix(semantic_features(df)), gender_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    fixed = evaluate(FIXED_PARAMS, X_train, X_test, y_train, y_test)
    print(f"Fixed configuration: accuracy={fixed['accuracy']:.4f}")

    combinations = [dict(zip(PARAM_GRID, values)) for values in itertools.product(*PARAM_GRID.values())]
    best_params, best = None, {"accuracy": 0.0}
    start = time.time()
    for i, params in enumerate(combinations, 1):
        metrics = evaluate(params, X_train, X_test, y_train, y_test)
        if metrics["accuracy"] > best["accuracy"]:
            best_params, best = params, metrics
        if i % 50 == 0:
            print(f"  {i}/{len(combinations)} evaluated, best accuracy {best['accuracy']:.4f} "
                  f"({(time.time() - start) / 60:.1f} min)")

    print(f"Best configuration: {best_params}, accuracy={best['accuracy']:.4f}")
    print(f"Improvement over fixed configuration: {100 * (best['accuracy'] - fixed['accuracy']):+.2f} points")
    save_json(
        {"feature_set": "semantic", "model": "random_forest", "task": "gender",
         "n_combinations": len(combinations), "param_grid": PARAM_GRID,
         "fixed_model": {"hyperparameters": FIXED_PARAMS, "metrics": fixed},
         "best_model": {"hyperparameters": best_params, "metrics": best},
         "improvement": {k: best[k] - fixed[k] for k in fixed}},
        RESULTS_DIR / "semantic_rf_grid_search.json",
    )


if __name__ == "__main__":
    main()
