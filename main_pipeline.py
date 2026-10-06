"""
=============================================================
MAIN PIPELINE — Web Server Log Cyber Attack Detection
=============================================================
Final Year Project | Web Server Node (Apache Logs)

3 Lightweight Keras Models — All FL Compatible:
  Model 1: Autoencoder      → access.log (unlabeled)
  Model 2: MLP Binary       → csic_binary.csv
  Model 3: MLP Multiclass   → multiclass_attacks.csv

Run:
    python main_pipeline.py
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Stage imports ─────────────────────────────────────────────
from stage1_data_loader import (
    load_unlabeled_apache,
    load_binary_csic,
    load_multiclass,
)
from stage2_features import (
    extract_features,
    extract_window_features,
    fit_and_save_scaler,
    scale,
    ALL_FEATURES,
    WINDOW_FEATURES,
)
from stage3_models import (
    AutoencoderModel,
    MLPBinaryModel,
    MLPMulticlassModel,
    FusionEngine,
)
from stage4_federated import run_federated_learning
from stage5_inference import (
    run_inference,
    print_report,
    save_outputs,
)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data files ─────────────────────────────────────────
    "unlabeled_log":      "data/access.log",
    "binary_labeled_csv": "data/csic_binary.csv",
    "multiclass_csv":     "data/multiclass_attacks.csv",

    # ── Sampling (set None to use all rows) ────────────────
    "multiclass_sample":  50000,   # 200k rows → sample 50k
    "binary_sample":      None,    # use all 61k rows

    # ── Model settings ─────────────────────────────────────
    "anomaly_threshold":  0.50,
    "ae_epochs":          30,
    "mlp_epochs":         50,

    # ── Federated Learning ─────────────────────────────────
    "fl_rounds":          3,

    # ── Output ─────────────────────────────────────────────
    "host_name":          "apache-web-server-node",
}


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_full_pipeline():
    print("\n" + "═"*60)
    print("  WEB SERVER LOG — CYBER ATTACK DETECTION")
    print("  Final Year Project | Web Server Node")
    print("  Models: Autoencoder + MLP Binary + MLP Multiclass")
    print("═"*60)

    # ── STAGE 1: Load datasets ────────────────────────────
    print("\n▶ STAGE 1: Loading Datasets")
    print("─"*40)

    df_unlabeled  = load_unlabeled_apache(
        CONFIG["unlabeled_log"])
    df_binary     = load_binary_csic(
        CONFIG["binary_labeled_csv"],
        CONFIG["binary_sample"])
    df_multiclass = load_multiclass(
        CONFIG["multiclass_csv"],
        CONFIG["multiclass_sample"])

    has_unlabeled  = len(df_unlabeled)  > 0
    has_binary     = len(df_binary)     > 0
    has_multiclass = len(df_multiclass) > 0

    print(f"\n  Summary:")
    print(f"    access.log       : {len(df_unlabeled):>6} rows")
    print(f"    csic_binary      : {len(df_binary):>6} rows")
    print(f"    multiclass       : {len(df_multiclass):>6} rows")

    # ── STAGE 2: Feature Extraction ───────────────────────
    print("\n▶ STAGE 2: Feature Extraction")
    print("─"*40)

    if has_unlabeled:
        df_unlabeled = extract_features(df_unlabeled)
        df_unlabeled = extract_window_features(df_unlabeled)

    if has_binary:
        df_binary = extract_features(df_binary)
        for col in WINDOW_FEATURES:
            df_binary[col] = 0.0

    if has_multiclass:
        df_multiclass = extract_features(df_multiclass)
        df_multiclass = extract_window_features(df_multiclass)

    # Fit scaler on all data combined
    all_dfs = [d for d in [df_unlabeled, df_binary,
                             df_multiclass] if len(d) > 0]
    df_combined = pd.concat(all_dfs, ignore_index=True)
    scaler = fit_and_save_scaler(df_combined)

    # Scale each dataset
    X_unlabeled = (scale(df_unlabeled, scaler)
                   if has_unlabeled
                   else np.empty((0, len(ALL_FEATURES))))
    X_binary    = (scale(df_binary, scaler)
                   if has_binary
                   else np.empty((0, len(ALL_FEATURES))))
    X_multi     = (scale(df_multiclass, scaler)
                   if has_multiclass
                   else np.empty((0, len(ALL_FEATURES))))

    y_binary = (df_binary['label']
                .fillna(0).astype(int).values
                if has_binary else np.array([]))
    y_multi  = (df_multiclass['attack_type']
                .fillna('Normal').values
                if has_multiclass else np.array([]))

    # All data for summary
    X_all = np.vstack([x for x in [X_unlabeled,
                                     X_binary,
                                     X_multi] if len(x) > 0])
    print(f"\n  Total feature matrix: {X_all.shape}")

    # ── STAGE 3: Train Models ─────────────────────────────
    print("\n▶ STAGE 3: Training Models")
    print("─"*40)

    n_features = len(ALL_FEATURES)

    # Model 1: Autoencoder — trained on unlabeled logs
    ae = AutoencoderModel(n_features, epochs=CONFIG["ae_epochs"])
    if has_unlabeled:
        ae.fit(X_unlabeled)
    elif has_binary:
        # fallback: train on normal rows from binary
        normal_mask = y_binary == 0
        if normal_mask.sum() > 10:
            ae.fit(X_binary[normal_mask])
    ae.save()

    # Model 2: MLP Binary — trained on CSIC data
    mlp_b = MLPBinaryModel(n_features, epochs=CONFIG["mlp_epochs"])
    if has_binary and len(np.unique(y_binary)) > 1:
        mlp_b.fit(X_binary, y_binary)
    elif has_multiclass:
        y_bin_fallback = (y_multi != 'Normal').astype(int)
        mlp_b.fit(X_multi, y_bin_fallback)
    mlp_b.save()

    # Model 3: MLP Multiclass — trained on multiclass CSV
    mlp_m = MLPMulticlassModel(n_features,
                                epochs=CONFIG["mlp_epochs"])
    if has_multiclass:
        mlp_m.fit(X_multi, y_multi)
    mlp_m.save()

    print("\n  ✅ All 3 models saved to models/")
    print(f"     autoencoder.keras    ← unlabeled logs")
    print(f"     mlp_binary.keras     ← csic binary")
    print(f"     mlp_multiclass.keras ← attack types")

    # ── STAGE 4: Federated Learning ───────────────────────
    print("\n▶ STAGE 4: Federated Learning")
    print("─"*40)

    X_fl = X_binary if has_binary else X_multi
    y_fl = (y_binary if has_binary
            else (y_multi != 'Normal').astype(int))

    fl_log = run_federated_learning(
        ae_model      = ae,
        mlp_binary    = mlp_b,
        mlp_multi     = mlp_m,
        X_local       = X_fl,
        y_binary_local= y_fl,
        n_rounds      = CONFIG["fl_rounds"]
    )

    # Save models after FL weight update
    ae.save()
    mlp_b.save()
    print("  ✅ Models updated with global FL weights")

    # ── STAGE 5: Inference ────────────────────────────────
    print("\n▶ STAGE 5: Inference")
    print("─"*40)

    fusion = FusionEngine(threshold=CONFIG["anomaly_threshold"])

    # Run on all labeled data
    X_infer  = np.vstack([x for x in [X_binary, X_multi]
                           if len(x) > 0])
    df_infer = pd.concat(
        [d for d in [df_binary, df_multiclass] if len(d) > 0],
        ignore_index=True
    )

    alerts = run_inference(
        X          = X_infer,
        df_orig    = df_infer,
        ae_model   = ae,
        mlp_binary = mlp_b,
        mlp_multi  = mlp_m,
        fusion     = fusion,
        threshold  = CONFIG["anomaly_threshold"],
        host       = CONFIG["host_name"]
    )

    # ── STAGE 6: Output ───────────────────────────────────
    print("\n▶ STAGE 6: Generating Reports")
    print("─"*40)

    print_report(alerts)
    save_outputs(alerts, CONFIG["host_name"])

    print("\n✅  Pipeline Complete!")
    print(f"    Models  → models/")
    print(f"    Alerts  → outputs/threat_alerts.json")
    print(f"    CSV     → outputs/threat_alerts.csv\n")

    return alerts, ae, mlp_b, mlp_m, scaler


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_full_pipeline()
