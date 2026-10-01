"""Figure 1: accuracy by feature set for gender (left) and inflection class (right).

Reads results/gender_prediction.json and results/inflection_prediction.json.
Uses the Okabe-Ito colour-blind-safe palette with hatching so that the figure
also reads in greyscale.

Output: results/figures/figure1_accuracy.{pdf,png}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hindi_gender.data import RESULTS_DIR
from hindi_gender.models import MODEL_NAMES

MODEL_LABELS = ["Logistic\nRegression", "Random\nForest", "Decision\nTree", "Gradient\nBoosting"]
COLORS = ["#E69F00", "#009E73", "#56B4E9", "#D55E00"]
HATCHES = ["", "///", "...", "xxx"]

GENDER_SETS = [
    ("semantic", "Semantic"), ("phonological", "Phono"), ("morphological", "Morph"),
    ("morphology_minus_inflection", "Morph\n(no inflection)"), ("etymological", "Etymological"),
    ("all_features", "All\nFeatures"), ("all_features_minus_inflection", "All\n(no inflection)"),
]
INFLECTION_SETS = [
    ("semantic", "Semantic"), ("phonological", "Phono"), ("morphological", "Morph"),
    ("morphology_minus_gender", "Morph\n(no gender)"), ("etymological", "Etymological"),
    ("all_features", "All\nFeatures"), ("all_features_minus_gender", "All\n(no gender)"),
]


def panel(ax, data, feature_sets, title):
    x = np.arange(len(feature_sets))
    width = 0.2
    for i, model in enumerate(MODEL_NAMES):
        accuracy = [data["results"][key][model]["accuracy"] for key, _ in feature_sets]
        ax.bar(x + width * (i - 1.5), accuracy, width, color=COLORS[i], edgecolor="black",
               linewidth=0.8, hatch=HATCHES[i], label=MODEL_LABELS[i])
    ax.axhline(data["majority_baseline"], color="red", linestyle="--", linewidth=1.5, label="Majority Baseline")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in feature_sets], fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.05, 0.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    gender = json.loads((RESULTS_DIR / "gender_prediction.json").read_text())
    inflection = json.loads((RESULTS_DIR / "inflection_prediction.json").read_text())

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    panel(left, gender, GENDER_SETS, "Predicting Gender")
    left.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    panel(right, inflection, INFLECTION_SETS, "Predicting Inflection")
    right.legend(fontsize=8, loc="upper left", frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout()

    out = RESULTS_DIR / "figures" / "figure1_accuracy"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        print(f"Saved {out.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
