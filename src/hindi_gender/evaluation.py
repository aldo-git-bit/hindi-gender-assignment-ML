"""Metrics, significance tests and experiment runners shared by the scripts."""

import json

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from .data import RANDOM_STATE
from .models import make_models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc_roc(y_true, proba, classes=None):
    """Binary AUC-ROC, or one-vs-rest macro-averaged AUC-ROC for multiclass."""
    if proba.ndim == 1:
        return roc_auc_score(y_true, proba)
    classes = np.unique(y_true) if classes is None else classes
    return roc_auc_score(label_binarize(y_true, classes=classes), proba, average="macro", multi_class="ovr")


def scores(model, X, y, multiclass, classes=None):
    """Accuracy, macro F1 and AUC-ROC of a fitted model, plus its predictions."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    if not multiclass:
        proba = proba[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
        "auc_roc": auc_roc(y, proba, classes),
    }
    return metrics, y_pred, proba


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, y_pred1, y_pred2, name1="Model 1", name2="Model 2"):
    """McNemar's test (with continuity correction) for two paired classifiers."""
    correct1 = np.asarray(y_pred1) == np.asarray(y_true)
    correct2 = np.asarray(y_pred2) == np.asarray(y_true)
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    if b + c == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, 1)
    acc1, acc2 = accuracy_score(y_true, y_pred1), accuracy_score(y_true, y_pred2)
    return {
        "model1_name": name1,
        "model2_name": name2,
        "model1_accuracy": acc1,
        "model2_accuracy": acc2,
        "accuracy_difference": acc2 - acc1,
        "discordant_pairs": {"model1_only_correct": b, "model2_only_correct": c},
        "test_statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def bootstrap_auc_test(y_true, proba1, proba2, name1="Model 1", name2="Model 2",
                       n_bootstrap=1000, confidence_level=0.95, random_state=RANDOM_STATE):
    """Bootstrap confidence interval for the AUC-ROC difference (model 2 - model 1)."""
    y_true = np.asarray(y_true)
    classes = np.unique(y_true) if proba1.ndim > 1 else None
    auc1, auc2 = auc_roc(y_true, proba1, classes), auc_roc(y_true, proba2, classes)
    observed = auc2 - auc1

    np.random.seed(random_state)
    diffs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            diffs.append(auc_roc(y_true[idx], proba2[idx], classes) - auc_roc(y_true[idx], proba1[idx], classes))
        except ValueError:  # resample is missing a class
            continue
    diffs = np.array(diffs)

    alpha = 1 - confidence_level
    low, high = np.percentile(diffs, 100 * alpha / 2), np.percentile(diffs, 100 * (1 - alpha / 2))
    contains_zero = bool(low <= 0 <= high)
    return {
        "model1_name": name1,
        "model2_name": name2,
        "model1_auc": auc1,
        "model2_auc": auc2,
        "auc_difference": observed,
        "confidence_interval": [low, high],
        "confidence_level": confidence_level,
        "contains_zero": contains_zero,
        "significant": not contains_zero,
        "n_bootstrap": len(diffs),
    }


def best_model(predictions):
    """Name of the most accurate model among ``{model: {'y_true', 'y_pred', ...}}``."""
    return max(predictions, key=lambda m: accuracy_score(predictions[m]["y_true"], predictions[m]["y_pred"]))


def statistical_analysis(all_predictions, majority_class, comparisons):
    """Compare the best model of each feature set against the majority baseline
    (McNemar) and run the requested pairwise comparisons.

    ``comparisons`` is a list of ``(set1, set2, bootstrap_auc)`` tuples; in each
    test ``set2`` is the second model, so differences are ``set2 - set1``.
    """
    first = next(iter(all_predictions.values()))
    y_true = first[next(iter(first))]["y_true"]
    baseline_pred = np.full(len(y_true), majority_class)
    best = {name: best_model(preds) for name, preds in all_predictions.items()}

    def label(name):
        return f"{name} ({best[name]})"

    out = {"feature_set_vs_baseline": {}, "pairwise_comparisons": {}, "bootstrap_auc_tests": {}}
    for name, preds in all_predictions.items():
        out["feature_set_vs_baseline"][name] = mcnemar_test(
            y_true, baseline_pred, preds[best[name]]["y_pred"], "majority_baseline", label(name)
        )
    for set1, set2, with_bootstrap in comparisons:
        p1, p2 = all_predictions[set1][best[set1]], all_predictions[set2][best[set2]]
        key = f"{set1}_vs_{set2}"
        out["pairwise_comparisons"][key] = mcnemar_test(y_true, p1["y_pred"], p2["y_pred"], label(set1), label(set2))
        if with_bootstrap:
            out["bootstrap_auc_tests"][key] = bootstrap_auc_test(
                y_true, p1["y_proba"], p2["y_proba"], label(set1), label(set2)
            )
    return out


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_single_split(X, y, multiclass, scale_within_split, test_size=0.2, random_state=RANDOM_STATE):
    """Fit the four classifiers on a stratified 80/20 split.

    If ``scale_within_split`` is True the scaler is fitted on the training
    portion only; otherwise ``X`` is expected to be scaled already.
    """
    classes = np.unique(y) if multiclass else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if scale_within_split:
        scaler = StandardScaler()
        X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    results, predictions = {}, {}
    for name, model in make_models(multiclass, random_state).items():
        model.fit(X_train, y_train)
        test, y_pred, proba = scores(model, X_test, y_test, multiclass, classes)
        train, _, _ = scores(model, X_train, y_train, multiclass, classes)
        results[name] = {**test, **{f"train_{k}": v for k, v in train.items()}, "n_features": X.shape[1]}
        if multiclass:
            results[name]["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        predictions[name] = {"y_true": y_test, "y_pred": y_pred, "y_proba": proba}
        print(f"  {name:20s} acc={test['accuracy']:.4f}  macro_f1={test['macro_f1']:.4f}  "
              f"auc={test['auc_roc']:.4f}  (train acc={train['accuracy']:.4f})")
    return results, predictions


def run_cross_validation(X, y, multiclass, scale_within_fold, n_folds=5, random_state=RANDOM_STATE):
    """Stratified k-fold cross-validation of the four classifiers.

    Returns mean and standard deviation of each metric across folds, plus the
    out-of-fold predictions concatenated over all folds (for significance tests).
    """
    classes = np.unique(y) if multiclass else None
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    metrics = {name: [] for name in make_models(multiclass)}
    oof = {name: {"y_true": [], "y_pred": [], "y_proba": []} for name in metrics}

    for train_idx, test_idx in folds.split(X, y):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        if scale_within_fold:
            scaler = StandardScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
        for name, model in make_models(multiclass, random_state).items():
            model.fit(X_train, y_train)
            test, y_pred, proba = scores(model, X_test, y_test, multiclass, classes)
            train, _, _ = scores(model, X_train, y_train, multiclass, classes)
            metrics[name].append({**test, **{f"train_{k}": v for k, v in train.items()}})
            oof[name]["y_true"].append(y_test)
            oof[name]["y_pred"].append(y_pred)
            oof[name]["y_proba"].append(proba)

    results, predictions = {}, {}
    for name, fold_metrics in metrics.items():
        summary = {}
        for key in fold_metrics[0]:
            values = [m[key] for m in fold_metrics]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        results[name] = {**summary, "n_features": X.shape[1], "n_folds": n_folds}
        predictions[name] = {k: np.concatenate(v) for k, v in oof[name].items()}
        print(f"  {name:20s} acc={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}  "
              f"macro_f1={summary['macro_f1_mean']:.4f}±{summary['macro_f1_std']:.4f}  "
              f"auc={summary['auc_roc_mean']:.4f}±{summary['auc_roc_std']:.4f}")
    return results, predictions


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"Saved {path}")
