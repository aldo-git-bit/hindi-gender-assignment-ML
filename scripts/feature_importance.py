"""Random Forest feature importance on the All Features set (paper Sections 5.1-5.2).

- Gender: share of total impurity-based importance held by each feature class
  (morphological features, i.e. mostly inflection class, account for 77.5%).
  As in the paper, this uses a Random Forest with scikit-learn defaults
  (100 trees) on unscaled features.
- Inflection class: importance of individual features for the regularized
  Random Forest from Table 5 (gender alone accounts for 53.1%, about 12x the
  next most important feature).

Both use the same stratified 80/20 split as the Table 5 experiments.

Output: results/feature_importance.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hindi_gender.data import RANDOM_STATE, RESULTS_DIR, load_dataset
from hindi_gender.evaluation import save_json
from hindi_gender.features import (
    build_feature_classes, gender_feature_sets, gender_target, inflection_feature_sets, to_matrix,
)
from hindi_gender.models import make_models

CLASS_PREFIXES = {"m_": "morphological", "p_": "phonological", "s_": "semantic", "e_": "etymological"}


def feature_class(name):
    return next(cls for prefix, cls in CLASS_PREFIXES.items() if name.startswith(prefix))


def summarize(importances):
    by_class = importances.groupby(importances.index.map(feature_class)).sum().sort_values(ascending=False)
    top = importances.sort_values(ascending=False).head(20)
    return {"by_feature_class": by_class.to_dict(), "top_features": top.to_dict()}


def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def main():
    df = load_dataset()
    classes = build_feature_classes(df)
    gender = gender_target(df)

    # Gender: default Random Forest, unscaled features.
    frame = gender_feature_sets(classes)["all_features"]
    X_train, _, y_train, _ = split(to_matrix(frame), gender)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(X_train, y_train)
    gender_imp = pd.Series(rf.feature_importances_, index=frame.columns)
    gender_summary = summarize(gender_imp)
    gender_summary["inflection_class_columns"] = float(gender_imp.filter(like="m_inflection_").sum())

    # Inflection: regularized Random Forest from Table 5, scaled within the split.
    frame = inflection_feature_sets(classes, gender)["all_features"]
    X_train, _, y_train, _ = split(to_matrix(frame), df["m_inflection"].to_numpy())
    rf = make_models(multiclass=True)["random_forest"].fit(StandardScaler().fit_transform(X_train), y_train)
    infl_imp = pd.Series(rf.feature_importances_, index=frame.columns)
    infl_summary = summarize(infl_imp)

    print("Gender - importance by feature class:")
    for cls, value in gender_summary["by_feature_class"].items():
        print(f"  {cls:15s} {100 * value:5.1f}%")
    print("\nInflection class - top features:")
    for feat, value in list(infl_summary["top_features"].items())[:5]:
        print(f"  {feat:25s} {100 * value:5.1f}%")

    save_json({"gender": gender_summary, "inflection_class": infl_summary}, RESULTS_DIR / "feature_importance.json")


if __name__ == "__main__":
    main()
