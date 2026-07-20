# -*- coding: utf-8 -*-
"""
config.py — Chemins et constantes de l'application Streamlit.
⚠️ SEULE SECTION À ADAPTER selon l'environnement (VM Liora, local, etc.).
"""
from pathlib import Path

# Racine de l'application (dossier 05_Streamlit)
APP_ROOT = Path(__file__).resolve().parent

# --- Données ---------------------------------------------------------------
# Résumé du dataset (bundlé dans le repo → l'onglet Exploration marche sans le dataset).
SUMMARY_CSV = APP_ROOT / "data" / "mvtec_summary.csv"

# CSV des features déjà calculées par feature_engineering.py (optionnel).
FEATURES_CSV = APP_ROOT / "data" / "features_engineered.csv"

# Racine des images redimensionnées (optionnel — pour la démo sur images réelles).
# Sur la VM : adapter au dossier dataBase_resized_128 / Resized_128.
DATASET_ROOT = APP_ROOT / "data" / "dataBase_resized_128"

# --- Modèles ---------------------------------------------------------------
# Templates moyens par catégorie (fichiers .npy), générés par prepare_assets.py.
TEMPLATES_DIR = APP_ROOT / "models" / "templates"

# Modèle de classification sur features (joblib/pickle) — à déposer par l'équipe.
FEATURE_MODEL_PATH = APP_ROOT / "models" / "anomaly_model.joblib"
SCALER_PATH        = APP_ROOT / "models" / "scaler.joblib"

# Autoencodeur Keras (.keras ou .h5) — à déposer par l'équipe.
AUTOENCODER_PATH = APP_ROOT / "models" / "autoencoder.keras"

# --- Paramètres image ------------------------------------------------------
DIMENSION = 128
TARGET_SIZE = (DIMENSION, DIMENSION)   # (largeur, hauteur)

# 12 features utilisées par le modèle (identiques à FEATURE_COLS de feature_engineering.py)
FEATURE_COLS = [
    "diff_area_ratio", "diff_n_components", "diff_max_blob_area", "diff_mean_intensity",
    "lap_var", "sobel_var", "edge_density",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "lbp_entropy",
]

GLCM_LEVELS = 24

# Catégories (les 3 pilotes du projet ; élargir au besoin)
CATEGORIES = ["bottle", "carpet", "screw"]

# Palette
COLOR_GOOD = "#2E7D32"
COLOR_ANOMALY = "#C62828"
COLOR_PRIMARY = "#1F3864"
