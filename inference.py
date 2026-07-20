# -*- coding: utf-8 -*-
"""
inference.py — Extraction de features et scoring d'anomalie pour la démo Streamlit.

Principe : on réutilise EXACTEMENT la logique de feature_engineering.py (Ludovic).
- Si feature_engineering.py est importable (présent sur la VM), on importe ses fonctions.
- Sinon, on utilise les ré-implémentations locales ci-dessous (mêmes formules).

⚠️ Aucune fonction ici ne réentraîne un modèle : on charge des artefacts déjà produits
(templates .npy, modèle joblib, autoencodeur Keras) ou on calcule un score déterministe
basé sur la différence au template (baseline).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

import config as C

# --- Dépendances image optionnelles (présentes sur la VM, pas garanties en local) ---
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# On tente d'importer le module des collègues (source de vérité des features).
try:
    import feature_engineering as FE   # nécessite feature_engineering.py dans le PYTHONPATH
    _HAS_FE = True
except Exception:
    FE = None
    _HAS_FE = False


# ════════════════════════════════════════════════════════════════════════
# PRÉTRAITEMENT
# ════════════════════════════════════════════════════════════════════════
def to_rgb_array(pil_img, size=C.TARGET_SIZE) -> np.ndarray:
    """Convertit une image PIL en array RGB uint8 redimensionné (HxWx3)."""
    img = pil_img.convert("RGB").resize(size)
    return np.asarray(img).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════
# TEMPLATES (moyenne des images train/good d'une catégorie)
# ════════════════════════════════════════════════════════════════════════
def load_template(category: str):
    """Charge le template .npy d'une catégorie s'il existe (généré par prepare_assets.py)."""
    p = C.TEMPLATES_DIR / f"{category}.npy"
    if p.exists():
        return np.load(p)
    return None


def available_templates() -> list[str]:
    if not C.TEMPLATES_DIR.exists():
        return []
    return sorted(p.stem for p in C.TEMPLATES_DIR.glob("*.npy"))


# ════════════════════════════════════════════════════════════════════════
# CARTE DE DIFFÉRENCE AU TEMPLATE (cœur de la détection non-supervisée)
# ════════════════════════════════════════════════════════════════════════
def diff_map(img_rgb: np.ndarray, template_rgb: np.ndarray):
    """
    Renvoie (carte_diff_gris, masque_binaire, dict_features_diff).
    Reproduit diff_features() de feature_engineering.py :
    absdiff -> GaussianBlur -> seuillage Otsu -> érosion/dilatation -> composantes connexes.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) requis pour la carte de différence.")
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_tpl = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray_img, gray_tpl)
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    n_comp, _, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    feats = {
        "diff_area_ratio":    float(mask.sum() / 255) / mask.size,
        "diff_n_components":  max(n_comp - 1, 0),
        "diff_max_blob_area": float(areas.max()) if len(areas) else 0.0,
        "diff_mean_intensity": float(diff.mean()),
    }
    return diff, mask, feats


# ════════════════════════════════════════════════════════════════════════
# FEATURES COMPLÈTES (12 colonnes = FEATURE_COLS)
# ════════════════════════════════════════════════════════════════════════
def extract_features(img_rgb: np.ndarray, template_rgb: np.ndarray) -> dict:
    """
    Renvoie les 12 features du modèle. Si feature_engineering.py est importable,
    on utilise SES fonctions (source de vérité) ; sinon, ré-implémentation locale.
    """
    if _HAS_FE:
        feats = {}
        feats.update(FE.diff_features(img_rgb, template_rgb))
        feats.update(FE.texture_edge_features(img_rgb))
        return {k: feats[k] for k in C.FEATURE_COLS}

    # --- Fallback local (mêmes formules que feature_engineering.py) ---
    if not (_HAS_CV2 and _HAS_SKIMAGE):
        raise RuntimeError("cv2 + scikit-image requis pour extraire les features en local.")
    _, _, dfeat = diff_map(img_rgb, template_rgb)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_var = float(cv2.Sobel(gray, cv2.CV_64F, 1, 0).var())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean() / 255)
    gq = (gray.astype(np.float64) * (C.GLCM_LEVELS - 1) / 255).astype(np.uint8)
    glcm = graycomatrix(gq, distances=[1], angles=[0], levels=C.GLCM_LEVELS,
                        symmetric=True, normed=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    hist = hist[hist > 0]
    lbp_entropy = float(-(hist * np.log2(hist)).sum())
    feats = {
        **dfeat,
        "lap_var": lap_var, "sobel_var": sobel_var, "edge_density": edge_density,
        "glcm_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "glcm_energy": float(graycoprops(glcm, "energy")[0, 0]),
        "glcm_correlation": float(graycoprops(glcm, "correlation")[0, 0]),
        "lbp_entropy": lbp_entropy,
    }
    return {k: feats[k] for k in C.FEATURE_COLS}


# ════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES (sans réentraînement)
# ════════════════════════════════════════════════════════════════════════
def load_feature_model():
    """Charge (modèle, scaler) joblib si présents, sinon (None, None)."""
    try:
        import joblib
        model = joblib.load(C.FEATURE_MODEL_PATH) if C.FEATURE_MODEL_PATH.exists() else None
        scaler = joblib.load(C.SCALER_PATH) if C.SCALER_PATH.exists() else None
        return model, scaler
    except Exception:
        return None, None


def load_autoencoder():
    """Charge l'autoencodeur Keras s'il existe, sinon None."""
    try:
        if not C.AUTOENCODER_PATH.exists():
            return None
        from tensorflow import keras
        return keras.models.load_model(C.AUTOENCODER_PATH, compile=False)
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════
# SCORING
# ════════════════════════════════════════════════════════════════════════
def score_baseline(img_rgb, template_rgb) -> dict:
    """
    Score d'anomalie BASELINE (fonctionne sans modèle entraîné) :
    intensité moyenne de la différence au template. Plus c'est haut, plus c'est anormal.
    """
    _, mask, feats = diff_map(img_rgb, template_rgb)
    return {"score": feats["diff_mean_intensity"], "features": feats, "mask": mask}


def score_autoencoder(img_rgb, model) -> dict:
    """Erreur de reconstruction de l'autoencodeur = score d'anomalie."""
    x = img_rgb.astype("float32") / 255.0
    recon = model.predict(x[None, ...], verbose=0)[0]
    err_map = np.mean((x - recon) ** 2, axis=-1)   # carte d'erreur par pixel
    return {"score": float(err_map.mean()), "recon": (recon * 255).astype(np.uint8),
            "err_map": err_map}
