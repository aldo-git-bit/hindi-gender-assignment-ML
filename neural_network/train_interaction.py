"""Train an MLP with an optional pairwise feature-interaction module (paper
Section 4.6, Section 5.3 and Table 3).

The primary predictor of each task is excluded: gender models do not see
inflection class, and inflection-class models do not see gender.

Usage:
    python neural_network/train_interaction.py --task gender --interaction none
    python neural_network/train_interaction.py --task inflection --interaction etym_phon

Interactions:
    none        baseline, no interaction module
    etym_deriv  Etymology x Derivation
    etym_phon   Etymology x Phonology
    etym_sem    Etymology x Semantic
    phon_sem    Phonology x Semantic
    deriv_sem   Derivation x Semantic
    deriv_phon  Derivation x Phonology

Outputs:
    results/neural_network/<task>_<interaction>.json   test-set metrics
    neural_network/checkpoints/<task>_<interaction>/best_model.keras
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from hindi_gender.data import RANDOM_STATE, RESULTS_DIR, load_dataset  # noqa: E402
from nn_features import encode_features  # noqa: E402

INTERACTIONS = ["none", "etym_deriv", "etym_phon", "etym_sem", "phon_sem", "deriv_sem", "deriv_phon"]

# Embedding sizes for the categorical inputs.
EMBEDDING_DIMS = {"m_deriv_cat_important": 8, "s_category_usas": 8, "s_category_wnet": 32}

# Size of the flattened outer product for each interaction.
INTERACTION_DIMS = {
    "etym_deriv": 4 * 8,
    "etym_phon": 4 * 20,
    "etym_sem": 4 * 32,
    "phon_sem": 20 * 32,
    "deriv_sem": 8 * 32,
    "deriv_phon": 8 * 20,
}


def outer_product(tensors):
    a, b = tensors
    return tf.einsum("bi,bj->bij", a, b)


def build_model(interaction, vocab_sizes, n_phonology, num_classes):
    inputs, embedded = {}, {}
    for name, dim in EMBEDDING_DIMS.items():
        inputs[name] = keras.Input(shape=(1,), name=name)
        emb = keras.layers.Embedding(vocab_sizes[name], dim, name=f"{name}_emb")(inputs[name])
        embedded[name] = keras.layers.Flatten()(emb)

    # Binary input: [m_compound, e_native, e_english, e_persian_arabic, e_sanskrit]
    inputs["binary"] = keras.Input(shape=(5,), name="binary")
    inputs["phonology"] = keras.Input(shape=(n_phonology,), name="phonology")

    features = list(embedded.values()) + [inputs["binary"], inputs["phonology"]]

    if interaction != "none":
        etymology = keras.layers.Lambda(lambda x: x[:, 1:5])(inputs["binary"])
        operands = {
            "etym": etymology,
            "deriv": embedded["m_deriv_cat_important"],
            "sem": embedded["s_category_wnet"],
            "phon": inputs["phonology"],
        }
        left, right = interaction.split("_")
        product = keras.layers.Lambda(outer_product)([operands[left], operands[right]])
        product = keras.layers.Flatten()(product)
        features.append(
            keras.layers.Dense(min(128, INTERACTION_DIMS[interaction]), activation="relu",
                               name="interaction_processor")(product)
        )

    x = keras.layers.Concatenate()(features)
    for units, dropout in [(128, 0.3), (64, 0.3), (32, 0.2)]:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(dropout)(x)

    if num_classes == 1:
        output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        output = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inputs=inputs, outputs=output)


class MacroF1Callback(keras.callbacks.Callback):
    """Track validation macro F1 each epoch and keep the best weights."""

    def __init__(self, val_inputs, val_target):
        super().__init__()
        self.val_inputs, self.val_target = val_inputs, val_target
        self.best_f1, self.best_weights = 0.0, None

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.val_inputs, verbose=0).argmax(axis=1)
        macro_f1 = f1_score(self.val_target, pred, average="macro")
        logs["val_macro_f1"] = macro_f1
        if macro_f1 > self.best_f1:
            self.best_f1, self.best_weights = macro_f1, self.model.get_weights()

    def restore_best_weights(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["gender", "inflection"])
    parser.add_argument("--interaction", required=True, choices=INTERACTIONS)
    args = parser.parse_args()

    # Seed Python, NumPy and TensorFlow, and use deterministic kernels so that
    # repeated runs on the same machine give identical results.
    keras.utils.set_random_seed(RANDOM_STATE)
    tf.config.experimental.enable_op_determinism()

    df = load_dataset()
    categorical, vocab_sizes, binary, phonology = encode_features(df)
    if args.task == "gender":
        target, num_classes = (df["m_gender"] == "M").astype(int).to_numpy(), 1
    else:
        target, num_classes = df["m_inflection"].to_numpy(), 6

    # Stratified 70/15/15 train/validation/test split.
    indices = np.arange(len(target))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=RANDOM_STATE, stratify=target)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15 / 0.85, random_state=RANDOM_STATE,
                                          stratify=target[train_val_idx])
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    def model_inputs(idx):
        data = {name: categorical[name][idx] for name in EMBEDDING_DIMS}
        data.update(binary=binary[idx], phonology=phonology[idx])
        return data

    train_x, val_x, test_x = model_inputs(train_idx), model_inputs(val_idx), model_inputs(test_idx)
    train_y, val_y, test_y = target[train_idx], target[val_idx], target[test_idx]

    model = build_model(args.interaction, vocab_sizes, phonology.shape[1], num_classes)
    run_name = f"{args.task}_{args.interaction}"
    checkpoint_dir = HERE / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / "best_model.keras")
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6)

    if num_classes == 1:
        # Class weights as used for the published results. Note that this
        # assigns the larger weight to the majority (masculine) class.
        n_pos = train_y.sum()
        n_neg = len(train_y) - n_pos
        class_weight = {0: len(train_y) / (2 * n_pos), 1: len(train_y) / (2 * n_neg)}
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy",
                      metrics=["accuracy", keras.metrics.AUC(name="auc")])
        f1_callback = None
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
            reduce_lr,
        ]
    else:
        weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_y)
        class_weight = dict(enumerate(weights))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        # Early stopping on validation macro F1 keeps the model from
        # collapsing onto the majority inflection class.
        f1_callback = MacroF1Callback(val_x, val_y)
        callbacks = [
            f1_callback,
            keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=30),
            keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_macro_f1", mode="max",
                                            save_best_only=True),
            reduce_lr,
        ]
    print(f"Class weights: {class_weight}")

    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=200, batch_size=64,
              class_weight=class_weight, callbacks=callbacks, verbose=2)
    if f1_callback is not None:
        f1_callback.restore_best_weights()

    evaluation = model.evaluate(test_x, test_y, verbose=0)
    proba = model.predict(test_x, verbose=0)
    if num_classes == 1:
        pred = (proba > 0.5).astype(int).ravel()
        metrics = {
            "accuracy": float(evaluation[1]),
            "auc": float(evaluation[2]),
            "loss": float(evaluation[0]),
            "f1": float(f1_score(test_y, pred)),
            "precision": float(precision_score(test_y, pred)),
            "recall": float(recall_score(test_y, pred)),
        }
    else:
        pred = proba.argmax(axis=1)
        classes = np.unique(test_y)
        metrics = {
            "accuracy": float(evaluation[1]),
            "loss": float(evaluation[0]),
            "macro_f1": float(f1_score(test_y, pred, average="macro")),
            "weighted_f1": float(f1_score(test_y, pred, average="weighted")),
            "auc_roc": float(roc_auc_score(label_binarize(test_y, classes=classes), proba,
                                           average="macro", multi_class="ovr")),
        }

    out = RESULTS_DIR / "neural_network" / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
