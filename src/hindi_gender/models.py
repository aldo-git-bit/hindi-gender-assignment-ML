"""Interpretable classifiers with the fixed, conservative hyperparameters used
for every feature set (paper Section 4.5 and Appendix B)."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .data import RANDOM_STATE

MODEL_NAMES = ("logistic_regression", "random_forest", "decision_tree", "gradient_boosting")


def make_models(multiclass=False, random_state=RANDOM_STATE):
    """Return fresh, unfitted instances of the four classifiers."""
    if multiclass:
        logreg = LogisticRegression(
            C=0.1, max_iter=1000, random_state=random_state, multi_class="multinomial", solver="lbfgs"
        )
    else:
        logreg = LogisticRegression(C=0.1, max_iter=1000, random_state=random_state)
    return {
        "logistic_regression": logreg,
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_leaf=5, max_features="sqrt", random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=5, learning_rate=0.05, random_state=random_state
        ),
    }
