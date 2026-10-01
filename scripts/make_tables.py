"""Render paper Tables 3, 5 and 6 as Markdown from the result files.

Output: results/tables/table3_nn_interactions.md
        results/tables/table5_single_split.md
        results/tables/table6_cross_validation.md
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from hindi_gender.data import RESULTS_DIR
from hindi_gender.models import MODEL_NAMES

MODEL_LABELS = dict(zip(MODEL_NAMES, ["LogReg", "RandForest", "DecTree", "GradBoost"]))
FEATURE_SETS = [  # (label, gender key, inflection key)
    ("Semantic", "semantic", "semantic"),
    ("Phonological", "phonological", "phonological"),
    ("Morphological", "morphological", "morphological"),
    ("Morphological (Ablated)", "morphology_minus_inflection", "morphology_minus_gender"),
    ("Etymological", "etymological", "etymological"),
    ("All Features", "all_features", "all_features"),
    ("All Features (Ablated)", "all_features_minus_inflection", "all_features_minus_gender"),
]
INTERACTIONS = [
    ("none", "Baseline (no interaction)"), ("etym_deriv", "Etymology × Derivation"),
    ("etym_phon", "Etymology × Phonology"), ("etym_sem", "Etymology × Semantic"),
    ("phon_sem", "Phonology × Semantic"), ("deriv_sem", "Derivation × Semantic"),
    ("deriv_phon", "Derivation × Phonology"),
]


def load(name):
    return json.loads((RESULTS_DIR / name).read_text())


def pct(x, sd=None):
    return f"{100 * x:.2f}%" if sd is None else f"{100 * x:.2f}%±{100 * sd:.2f}%"


def auc(x, sd=None):
    return f"{x:.4f}" if sd is None else f"{x:.4f}±{sd:.4f}"


def whitebox_table(gender, inflection, cv):
    rows = []
    for label, g_key, i_key in FEATURE_SETS:
        for model in MODEL_NAMES:
            row = {"Feature Set": label, "Model": MODEL_LABELS[model]}
            for task, data, key in (("Gender", gender, g_key), ("Inflection", inflection, i_key)):
                m = data["results"][key][model]
                if cv:
                    row[f"{task} Acc"] = pct(m["accuracy_mean"], m["accuracy_std"])
                    row[f"{task} F1"] = pct(m["macro_f1_mean"], m["macro_f1_std"])
                    row[f"{task} AUC"] = auc(m["auc_roc_mean"], m["auc_roc_std"])
                else:
                    row[f"{task} Acc"] = pct(m["accuracy"])
                    row[f"{task} F1"] = pct(m["macro_f1"])
                    row[f"{task} AUC"] = auc(m["auc_roc"])
            rows.append(row)
    return pd.DataFrame(rows)


def nn_table():
    rows = []
    base = {task: load(f"neural_network/{task}_none.json")["accuracy"] for task in ("gender", "inflection")}
    for key, label in INTERACTIONS:
        g = load(f"neural_network/gender_{key}.json")
        i = load(f"neural_network/inflection_{key}.json")
        delta = lambda m, task: "–" if key == "none" else f"{100 * (m['accuracy'] - base[task]):+.2f}%"
        rows.append({
            "Interaction": label,
            "Gender Acc": pct(g["accuracy"]), "Gender F1": pct(g["f1"]), "Gender AUC": auc(g["auc"]),
            "Gender ΔAcc": delta(g, "gender"),
            "Inflection Acc": pct(i["accuracy"]), "Inflection Macro F1": pct(i["macro_f1"]),
            "Inflection AUC": auc(i["auc_roc"]), "Inflection ΔAcc": delta(i, "inflection"),
        })
    return pd.DataFrame(rows)


def write(table, filename, title):
    out = RESULTS_DIR / "tables" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    text = table.to_markdown(index=False, disable_numparse=True)
    out.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
    print(f"{title}\n{text}\nSaved {out}\n")


def main():
    if all((RESULTS_DIR / "neural_network" / f"{t}_{k}.json").exists()
           for t in ("gender", "inflection") for k, _ in INTERACTIONS):
        write(nn_table(), "table3_nn_interactions.md",
              "Table 3: Neural network models without the primary predictor, with pairwise interactions")
    else:
        print("Skipping Table 3: neural network results not found (run `make nn`).\n")
    write(whitebox_table(load("gender_prediction.json"), load("inflection_prediction.json"), cv=False),
          "table5_single_split.md", "Table 5: Interpretable models, stratified 80/20 split")
    write(whitebox_table(load("gender_prediction_cv.json"), load("inflection_prediction_cv.json"), cv=True),
          "table6_cross_validation.md", "Table 6: Interpretable models, 5-fold cross-validation (mean ± std)")


if __name__ == "__main__":
    main()
