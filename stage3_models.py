"""
=============================================================
STAGE 3 — MODELS
=============================================================
3 lightweight Keras models — all FL compatible:

  Model 1: Autoencoder      → unlabeled access.log
  Model 2: MLP Binary       → csic_binary.csv
  Model 3: MLP Multiclass   → multiclass_attacks.csv

All use get_weights() / set_weights() for real FL.
=============================================================
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import callbacks as tf_cb
    TF_OK = True
except Exception as e:
    print(f"[Warning] TensorFlow unavailable: {e}")
    TF_OK = False


# ═══════════════════════════════════════════════════════════════
# MODEL 1 — AUTOENCODER
# Trains on: access.log (unlabeled, 815 rows)
# Purpose:   detect anomalies via reconstruction error
# FL:        ✅ get_weights() / set_weights()
# ═══════════════════════════════════════════════════════════════

class AutoencoderModel:

    def __init__(self, input_dim, epochs=30, batch_size=64):
        self.input_dim  = input_dim
        self.epochs     = epochs
        self.batch_size = batch_size
        self.threshold  = None
        self.model      = None
        if TF_OK:
            self._build()

    def _build(self):
        inp = keras.Input(shape=(self.input_dim,))
        # Encoder
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(8, activation='relu',
                          name='bottleneck')(x)
        # Decoder
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(self.input_dim,
                           activation='linear')(x)

        self.model = keras.Model(inp, out)
        self.model.compile(optimizer='adam', loss='mse')
        print(f"[Model 1] Autoencoder built | "
              f"Params: {self.model.count_params():,}")

    def fit(self, X_normal):
        if not TF_OK or self.model is None:
            print("[Model 1] Skipped (TF unavailable)")
            return self
        print(f"[Model 1] Training Autoencoder on "
              f"{len(X_normal)} normal samples...")

        Xtr, Xv = train_test_split(
            X_normal, test_size=0.1, random_state=42)

        self.model.fit(
            Xtr, Xtr,
            validation_data=(Xv, Xv),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[
                tf_cb.EarlyStopping(
                    patience=5,
                    restore_best_weights=True)
            ]
        )
        pred = self.model.predict(X_normal, verbose=0)
        mse  = np.mean(np.square(X_normal - pred), axis=1)
        self.threshold = float(np.percentile(mse, 95))
        print(f"  Done. Threshold: {self.threshold:.6f}")
        return self

    def score(self, X) -> np.ndarray:
        """Returns anomaly score 0→1 per request."""
        if not TF_OK or self.model is None:
            return np.zeros(len(X))
        pred = self.model.predict(X, verbose=0)
        mse  = np.mean(np.square(X - pred), axis=1)
        return np.clip(mse / (self.threshold + 1e-8), 0, 1)

    def get_weights(self):
        return self.model.get_weights() if self.model else []

    def set_weights(self, weights):
        if self.model:
            self.model.set_weights(weights)

    def save(self):
        if self.model:
            self.model.save("models/autoencoder.keras")
            joblib.dump(self.threshold,
                        "models/ae_threshold.pkl")
            print("  Saved → models/autoencoder.keras")

    def load(self):
        if TF_OK:
            self.model = keras.models.load_model(
                "models/autoencoder.keras")
            self.threshold = joblib.load(
                "models/ae_threshold.pkl")
        return self


# ═══════════════════════════════════════════════════════════════
# MODEL 2 — MLP BINARY CLASSIFIER
# Trains on: csic_binary.csv (61,065 rows)
# Purpose:   Normal vs Attack probability
# FL:        ✅ get_weights() / set_weights()
# Fix 2+3:   label smoothing + smaller model + dropout
# ═══════════════════════════════════════════════════════════════

class MLPBinaryModel:

    def __init__(self, input_dim=41, epochs=50,
                 batch_size=256):
        self.input_dim  = input_dim
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model      = None
        if TF_OK:
            self._build()

    def _build(self):
        inp = keras.Input(shape=(self.input_dim,))
        # Fix 3: smaller model than XGBoost equivalent
        x = layers.Dense(128, activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        out = layers.Dense(1, activation='sigmoid')(x)

        self.model = keras.Model(inp, out)
        # Fix 2: label smoothing = 0.05 for binary
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(
                label_smoothing=0.05),
            metrics=['accuracy']
        )
        print(f"[Model 2] MLP Binary built | "
              f"Params: {self.model.count_params():,}")

    def fit(self, X, y):
        if not TF_OK or self.model is None:
            return self
        print(f"[Model 2] Training MLP Binary | "
              f"Normal:{(y==0).sum()} Attack:{(y==1).sum()}")

        # Class weights for imbalance
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        total = len(y)
        cw    = {0: total/(2*n_neg), 1: total/(2*n_pos)}

        Xtr, Xv, ytr, yv = train_test_split(
            X, y, test_size=0.2,
            random_state=42, stratify=y)

        self.model.fit(
            Xtr, ytr,
            validation_data=(Xv, yv),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=cw,
            verbose=0,
            callbacks=[
                tf_cb.EarlyStopping(
                    patience=8,
                    restore_best_weights=True),
                tf_cb.ReduceLROnPlateau(
                    patience=4, factor=0.5)
            ]
        )
        yp = (self.model.predict(
            Xv, verbose=0).ravel() >= 0.5).astype(int)
        f1 = f1_score(yv, yp, zero_division=0)
        print(f"  Val F1: {f1:.4f}")
        print(classification_report(
            yv, yp,
            target_names=['Normal','Attack'],
            zero_division=0))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not TF_OK or self.model is None:
            return np.full(len(X), 0.5)
        return self.model.predict(X, verbose=0).ravel()

    def predict(self, X, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_weights(self):
        return self.model.get_weights() if self.model else []

    def set_weights(self, weights):
        if self.model:
            self.model.set_weights(weights)

    def save(self):
        if self.model:
            self.model.save("models/mlp_binary.keras")
            print("  Saved → models/mlp_binary.keras")

    def load(self):
        if TF_OK:
            self.model = keras.models.load_model(
                "models/mlp_binary.keras")
        return self


# ═══════════════════════════════════════════════════════════════
# MODEL 3 — MLP MULTICLASS
# Trains on: multiclass_attacks.csv (200k rows)
# Purpose:   classify 8 attack types
# FL:        ✅ get_weights() / set_weights()
# Fix 2+3:   label smoothing=0.1 + smaller model + dropout=0.4
# ═══════════════════════════════════════════════════════════════

class MLPMulticlassModel:

    CLASSES = ['Normal','SQLi','XSS','CommandInjection',
               'PathTraversal','RFI','LFI','SSRF']

    def __init__(self, input_dim=41, epochs=50,
                 batch_size=256):
        self.input_dim  = input_dim
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model      = None
        self.le         = LabelEncoder()
        self.classes_   = None
        self.n_classes  = None

    def _build(self, n_classes):
        inp = keras.Input(shape=(self.input_dim,))
        # Fix 3: smaller model — stops memorizing synthetic data
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)   # Fix 3: stronger dropout
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16, activation='relu')(x)
        out = layers.Dense(n_classes,
                           activation='softmax')(x)

        self.model = keras.Model(inp, out)
        # Fix 2: label smoothing = 0.1 — stops overconfidence
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.CategoricalCrossentropy(
                label_smoothing=0.1),
            metrics=['accuracy']
        )
        print(f"[Model 3] MLP Multiclass built | "
              f"Params: {self.model.count_params():,} | "
              f"Classes: {n_classes}")

    def fit(self, X, y_labels):
        if not TF_OK:
            return self
        print(f"[Model 3] Training MLP Multiclass | "
              f"{len(X)} samples")

        y_enc       = self.le.fit_transform(y_labels)
        self.classes_ = list(self.le.classes_)
        self.n_classes = len(self.classes_)

        print(f"  Classes: {self.classes_}")
        print(pd.Series(y_labels).value_counts().to_string())

        self._build(self.n_classes)

        y_cat = tf.keras.utils.to_categorical(
            y_enc, self.n_classes)

        Xtr, Xv, ytr, yv, yenc_tr, yenc_v = train_test_split(
            X, y_cat, y_enc,
            test_size=0.2,
            random_state=42,
            stratify=y_enc)

        self.model.fit(
            Xtr, ytr,
            validation_data=(Xv, yv),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[
                tf_cb.EarlyStopping(
                    patience=8,
                    restore_best_weights=True),
                tf_cb.ReduceLROnPlateau(
                    patience=4, factor=0.5)
            ]
        )
        yp = np.argmax(
            self.model.predict(Xv, verbose=0), axis=1)
        f1 = f1_score(
            yenc_v, yp, average='weighted', zero_division=0)
        print(f"  Val Weighted F1: {f1:.4f}")
        print(classification_report(
            yenc_v, yp,
            target_names=self.classes_,
            zero_division=0))
        return self

    def predict(self, X) -> np.ndarray:
        if not TF_OK or self.model is None:
            return np.array(['Normal'] * len(X))
        enc = np.argmax(
            self.model.predict(X, verbose=0), axis=1)
        return self.le.inverse_transform(enc)

    def predict_proba(self, X) -> np.ndarray:
        if not TF_OK or self.model is None:
            return np.full((len(X), 8), 1/8)
        return self.model.predict(X, verbose=0)

    def get_weights(self):
        return self.model.get_weights() if self.model else []

    def set_weights(self, weights):
        if self.model:
            self.model.set_weights(weights)

    def save(self):
        if self.model:
            self.model.save("models/mlp_multiclass.keras")
            joblib.dump(
                {'le': self.le, 'classes': self.classes_,
                 'n_classes': self.n_classes},
                "models/label_encoder.pkl")
            print("  Saved → models/mlp_multiclass.keras")
            print("  Saved → models/label_encoder.pkl")

    def load(self):
        if TF_OK:
            self.model = keras.models.load_model(
                "models/mlp_multiclass.keras")
            d = joblib.load("models/label_encoder.pkl")
            self.le        = d['le']
            self.classes_  = d['classes']
            self.n_classes = d['n_classes']
        return self


# ═══════════════════════════════════════════════════════════════
# FUSION ENGINE
# Combines 3 model scores into one final decision
# ═══════════════════════════════════════════════════════════════

class FusionEngine:

    def __init__(self, threshold=0.50):
        self.threshold = threshold

    def fuse(self, ae_scores, mlp_scores,
             atk_types) -> pd.DataFrame:
        """
        Fusion formula:
          final = 0.30 × ae_score + 0.70 × mlp_binary_score

        ae_score   → from Autoencoder (unlabeled anomaly)
        mlp_score  → from MLP Binary  (supervised attack prob)
        """
        rows = []
        for i in range(len(ae_scores)):
            ae  = float(ae_scores[i])
            mlp = float(mlp_scores[i])
            atype = str(atk_types[i])

            final = 0.30 * ae + 0.70 * mlp
            final = round(final, 4)

            conf = ('HIGH'   if final > 0.70 else
                    'MEDIUM' if final > 0.50 else 'LOW')

            rows.append({
                'ae_score':    round(ae,  4),
                'mlp_score':   round(mlp, 4),
                'final_score': final,
                'is_attack':   int(final >= self.threshold),
                'attack_type': atype,
                'confidence':  conf,
            })
        return pd.DataFrame(rows)
