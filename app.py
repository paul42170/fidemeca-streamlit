# -*- coding: utf-8 -*-
"""
app.py — Application Streamlit — Détection d'anomalies (Projet DS Fidémeca / MVTec AD).

Lancement :  streamlit run app.py

Structure (onglets) :
  1. 🏭 Contexte              — problème métier + approche
  2. 🔍 Exploration           — les 5 graphiques du dataset
  3. 🛠️ Pré-processing & FE    — resize + features (avec démo template/diff)
  4. 🤖 Modélisation & Démo    — upload d'une image → score d'anomalie (baseline / RF / AutoEncoder / CNN)
  5. 📊 Comparatif de l'équipe — tous les modèles testés par l'équipe (GitHub), avec ou sans démo live
  6. 📌 Conclusion            — résultats, limites, perspectives

Aucun modèle n'est réentraîné : on charge des artefacts déjà produits (models/…), qu'ils
viennent du dépôt Streamlit lui-même ou du dépôt GitHub d'équipe (CNN 15 catégories).
Un mode "baseline" (différence au template) rend la démo fonctionnelle même sans modèle final.
"""
import numpy as np
import pandas as pd
import streamlit as st

import config as C

# ──────────────────────────────────────────────────────────────────────────
# CONFIG PAGE + STYLE
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fidémeca — Détection d'anomalies", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .main .block-container {{ padding-top: 2rem; }}
  h1, h2, h3 {{ color: {C.COLOR_PRIMARY}; }}
  .kpi {{ background:#F2F6FC; border-radius:12px; padding:14px 18px; text-align:center; }}
  .kpi .v {{ font-size:1.6rem; font-weight:700; color:{C.COLOR_PRIMARY}; }}
  .kpi .l {{ font-size:.8rem; color:#555; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES (mis en cache)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(C.SUMMARY_CSV)
    df["total"] = df.train_good + df.test_good + df.test_defaut
    df["part_defaut_test"] = df.test_defaut / (df.test_good + df.test_defaut)
    return df


df = load_summary()


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Détection d'anomalies")
    st.caption("Projet fil rouge — Bootcamp MLE, cohorte Avril 2026")
    st.markdown("**Équipe :** Paul Fournel · Alex Mac-Kame · Ludovic Marquant · Fabrice Masola")
    st.markdown("**Entreprise (cas) :** Fidémeca — mécanique de précision")
    st.markdown("**Dataset :** MVTec AD (15 catégories, 5 354 images)")
    st.divider()
    st.caption("Modèles chargés (aucun réentraînement) :")
    from inference import (available_templates, load_feature_model, available_autoencoders,
                           available_cnn_variants)
    _tpls = available_templates()
    _fm, _sc = load_feature_model()
    _ae_cats = available_autoencoders()
    _cnn_variants = available_cnn_variants()
    st.write("• Templates :", f"{len(_tpls)} dispo" if _tpls else "❌ aucun")
    st.write("• Modèle features (RF) :", "✅" if _fm else "❌ absent")
    st.write("• Autoencodeurs :", f"✅ {', '.join(_ae_cats)}" if _ae_cats else "❌ aucun")
    st.write("• CNN 15 catégories :", f"✅ {len(_cnn_variants)} variante(s)" if _cnn_variants else "❌ absent")
    st.caption("Détail des modèles testés par toute l'équipe → onglet 📊 Comparatif.")


# ──────────────────────────────────────────────────────────────────────────
# ONGLETS
# ──────────────────────────────────────────────────────────────────────────
t_ctx, t_expl, t_prep, t_model, t_cmp, t_ccl = st.tabs(
    ["🏭 Contexte", "🔍 Exploration", "🛠️ Pré-processing & FE", "🤖 Modélisation & Démo",
     "📊 Comparatif de l'équipe", "📌 Conclusion"]
)

# ══════════════════════════════════════════════════════════════════════════
# 1. CONTEXTE
# ══════════════════════════════════════════════════════════════════════════
with t_ctx:
    st.header("Contrôle qualité automatisé chez Fidémeca")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
**Le problème métier.** Fidémeca usine des pièces de précision (aéronautique, médical…).
Le contrôle qualité est aujourd'hui **manuel et par échantillonnage** : lent (~5 min/pièce),
coûteux, sujet à l'erreur humaine, avec **~5 % de rebuts** et un coût de non-qualité estimé
à **~300 k€/an**.

**L'objectif.** Fournir un système capable de **repérer automatiquement les pièces défectueuses**
à partir d'une simple image, pour contrôler **100 %** des pièces et réduire les rebuts.

**L'approche : détection d'anomalies *non-supervisée*.**
On n'apprend au modèle **que l'apparence des pièces conformes** (dossier `train/good`).
Toute pièce qui s'écarte trop de cette « norme » est signalée comme anormale.
C'est la logique du dataset de référence **MVTec AD**.
        """)
    with c2:
        st.info("**Pourquoi non-supervisé ?**\n\nEn production, on dispose de beaucoup de pièces "
                "conformes mais de peu d'exemples de chaque défaut possible. Apprendre la seule "
                "« normalité » évite d'avoir à collecter et étiqueter tous les types de défauts.")
        st.metric("Objectif temps de contrôle", "< 15 s", "-5 min")
        st.metric("Objectif taux de rebuts", "< 3 %", "-2 pts")

    st.divider()
    k = st.columns(4)
    tot = int(df.total.sum())
    kpis = [("Catégories", f"{len(df)}"), ("Images totales", f"{tot:,}".replace(",", " ")),
            ("Images d'apprentissage", f"{int(df.train_good.sum()):,}".replace(",", " ")),
            ("Images de test", f"{int(df.test_good.sum()+df.test_defaut.sum()):,}".replace(",", " "))]
    for col, (lab, val) in zip(k, kpis):
        col.markdown(f"<div class='kpi'><div class='v'>{val}</div><div class='l'>{lab}</div></div>",
                     unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 2. EXPLORATION
# ══════════════════════════════════════════════════════════════════════════
with t_expl:
    st.header("Exploration du jeu de données")
    st.caption("Chiffres issus du rapport d'exploration (data/mvtec_summary.csv).")

    import matplotlib.pyplot as plt

    def bar(ax, x, y, title, ylabel, color=C.COLOR_PRIMARY, pct=False, horizontal=False):
        if horizontal:
            ax.barh(x, y, color=color)
        else:
            ax.bar(x, y, color=color)
            ax.set_xticklabels(x, rotation=45, ha="right")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_ylabel(ylabel)
        if pct:
            ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")

    col1, col2 = st.columns(2)
    d = df.sort_values("categorie")

    with col1:
        st.subheader("1. Composition par catégorie")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(d.categorie, d.train_good, label="train (normal)", color="#4C72B0")
        ax.bar(d.categorie, d.test_good, bottom=d.train_good, label="test good", color="#2E7D32")
        ax.bar(d.categorie, d.test_defaut, bottom=d.train_good + d.test_good, label="test défaut", color="#C62828")
        ax.set_xticklabels(d.categorie, rotation=45, ha="right"); ax.legend(fontsize=7)
        ax.set_ylabel("Nb d'images"); st.pyplot(fig)
        st.caption("Le train ne contient que des pièces normales — cohérent avec le non-supervisé.")

        st.subheader("3. Résolutions")
        rc = d.resolution.value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(rc.index, rc.values, color=C.COLOR_PRIMARY); ax.set_xlabel("Nb catégories")
        st.pyplot(fig)
        st.caption("Hétérogénéité des résolutions → redimensionnement obligatoire au preprocessing.")

    with col2:
        st.subheader("2. % de défauts dans le test")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(d.categorie, d.part_defaut_test, color="#C62828")
        ax.set_xticklabels(d.categorie, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%"); ax.set_ylabel("% défaut")
        st.pyplot(fig)
        st.caption("Test volontairement riche en défauts → juger avec AUROC/Recall, pas l'accuracy.")

        st.subheader("4. Modes couleur")
        mc = d["mode"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.pie(mc.values, labels=[f"{i} ({v})" for i, v in mc.items()],
               autopct="%1.0f%%", colors=["#4C72B0", "#B0B0B0"])
        st.pyplot(fig)
        st.caption("3 catégories en niveaux de gris (grid, screw, zipper) → uniformiser les canaux.")

    st.divider()
    st.subheader("5. Nombre de types de défauts par catégorie")
    fig, ax = plt.subplots(figsize=(11, 3))
    dd = d.sort_values("nb_types_defauts", ascending=False)
    ax.bar(dd.categorie, dd.nb_types_defauts, color=C.COLOR_PRIMARY)
    ax.set_xticklabels(dd.categorie, rotation=45, ha="right"); ax.set_ylabel("Nb types")
    st.pyplot(fig)
    st.caption("cable = 8 types (le plus complexe) ; toothbrush = 1 seul (« defective »).")

    with st.expander("Voir le tableau détaillé"):
        st.dataframe(df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# 3. PRÉ-PROCESSING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
with t_prep:
    st.header("Pré-processing & Feature Engineering")
    st.markdown("""
**1. Redimensionnement.** Les images (700×700 à 1024×1024) sont ramenées à **128×128**
(48 Ko vs 2,3 Mo), ce qui uniformise les entrées et allège le calcul.

**2. Template de référence.** Pour chaque catégorie, on calcule la **moyenne pixel-à-pixel
des images `train/good`** → une image « pièce parfaite ». Une anomalie se traduit par des
**zones de forte différence** entre l'image testée et ce template.

**3. Trois familles de features** (12 retenues pour le modèle) :
- **Différence au template** (4) : `diff_mean_intensity`, `diff_area_ratio`, `diff_max_blob_area`, `diff_n_components`.
- **Texture & contours** (7) : variance Laplacien/Sobel, densité Canny, GLCM (contraste, homogénéité, énergie, corrélation), entropie LBP.
- **Couleur** (RGB) pour les catégories couleur.

**4. Augmentation & anomalies synthétiques** : rotations/flips/bruit + méthode **Cut-Paste**
(collage d'un patch pour simuler un défaut), puisqu'on n'a pas de vrais défauts à l'entraînement.
    """)

    st.subheader("Top features discriminantes (validation statistique)")
    st.caption("Test de Mann-Whitney U + delta de Cliff (taille d'effet). |δ| grand = très discriminant.")
    top = pd.DataFrame({
        "Feature": ["diff_mean_intensity", "diff_max_blob_area", "lbp_entropy", "diff_area_ratio",
                    "glcm_energy", "diff_mean_intensity", "laplacian_variance", "canny_edge_density"],
        "Catégorie": ["bottle", "bottle", "bottle", "bottle", "screw", "carpet", "bottle", "carpet"],
        "Delta de Cliff": [-0.643, 0.594, -0.550, 0.531, -0.497, -0.484, -0.461, 0.422],
        "p-value": ["5.7e-15", "<0.001", "<0.001", "<0.001", "<0.001", "<0.001", "<0.001", "<0.001"],
    })
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.info("Les **features de différence au template** sont les plus puissantes (surtout `bottle`, "
            "surface lisse). Les features de texture complètent pour les surfaces répétitives (`carpet`).")

    st.divider()
    st.subheader("🔬 Démo : carte de différence au template")
    st.caption("Charge une image de test + son template pour visualiser où le modèle « voit » l'anomalie.")
    from inference import load_template, to_rgb_array, diff_map, available_templates
    tpls = available_templates()
    if not tpls:
        st.warning("Aucun template disponible. Génère-les une fois avec `python prepare_assets.py` "
                   "(voir README) pour activer cette démo.")
    else:
        cc = st.columns([1, 2])
        cat = cc[0].selectbox("Catégorie (template)", tpls, key="prep_cat")
        up = cc[1].file_uploader("Image de pièce (PNG/JPG)", type=["png", "jpg", "jpeg"], key="prep_up")
        if up is not None:
            from PIL import Image
            img = to_rgb_array(Image.open(up))
            tpl = load_template(cat)
            try:
                diff, mask, feats = diff_map(img, tpl)
                g1, g2, g3, g4 = st.columns(4)
                g1.image(img, caption="Image testée", use_container_width=True)
                g2.image(tpl, caption="Template (pièce moyenne)", use_container_width=True)
                g3.image(diff, caption="Carte de différence", use_container_width=True, clamp=True)
                g4.image(mask, caption="Zones détectées", use_container_width=True, clamp=True)
                st.write("**Features de différence :**", {k: round(v, 4) for k, v in feats.items()})
            except Exception as e:
                st.error(f"Erreur : {e}")

# ══════════════════════════════════════════════════════════════════════════
# 4. MODÉLISATION & DÉMO
# ══════════════════════════════════════════════════════════════════════════
with t_model:
    st.header("Modélisation & démonstration")
    st.markdown("""
La démo **charge un modèle déjà entraîné** (aucun réentraînement dans l'appli, conformément
aux consignes du mentor). Quatre modes selon ce qui est disponible dans `models/` :
""")
    from inference import (load_template, to_rgb_array, score_baseline, extract_features,
                           load_feature_model, load_autoencoder, score_autoencoder,
                           available_templates, available_autoencoders, available_cnn_variants,
                           load_cnn, score_cnn)

    ae_cats = available_autoencoders()
    cnn_variants = available_cnn_variants()
    mode = st.radio("Mode de détection", [
        "Baseline — différence au template (toujours dispo)",
        "Modèle sur features — Random Forest (joblib)",
        f"AutoEncodeur (Keras) — {', '.join(ae_cats) if ae_cats else 'aucun disponible'}",
        f"CNN supervisé — 15 catégories (Keras) — {len(cnn_variants)} variante(s)",
    ], horizontal=False)

    tpls = available_templates()
    colL, colR = st.columns([1, 2])
    cat = colL.selectbox("Catégorie de la pièce", tpls or C.CATEGORIES)
    variant = None
    if mode.startswith("CNN"):
        variant = colL.selectbox("Variante du CNN (ablation — voir Partie IV du rapport)", cnn_variants)
        seuil = 50.0
    else:
        seuil = colL.slider("Seuil de décision (anomalie si score >)", 0.0, 60.0, 12.0, 0.5)
    up = colR.file_uploader("Déposer une image de pièce à contrôler", type=["png", "jpg", "jpeg"])

    if up is not None:
        from PIL import Image
        img = to_rgb_array(Image.open(up))
        tpl = load_template(cat)
        st.image(img, width=220, caption="Pièce à contrôler")

        try:
            if mode.startswith("Baseline"):
                if tpl is None:
                    st.warning("Template manquant pour cette catégorie (voir prepare_assets.py).")
                else:
                    res = score_baseline(img, tpl)
                    score = res["score"]
                    verdict = score > seuil
                    st.metric("Score d'anomalie (diff. moyenne)", f"{score:.2f}",
                              delta="ANOMALIE" if verdict else "conforme",
                              delta_color="inverse" if verdict else "normal")
                    st.image(res["mask"], caption="Zones suspectes", width=220, clamp=True)
                    (st.error if verdict else st.success)(
                        "🔴 Pièce probablement DÉFECTUEUSE" if verdict else "🟢 Pièce conforme")

            elif mode.startswith("Modèle sur features"):
                if cat not in C.FEATURE_MODEL_CATEGORIES:
                    st.warning(f"Le modèle Random Forest a été entraîné sur les catégories pilotes "
                               f"**{', '.join(C.FEATURE_MODEL_CATEGORIES)}** uniquement. Le résultat sur "
                               f"« {cat} » n'a pas été validé par l'équipe — à titre indicatif.")
                model, scaler = load_feature_model()
                if model is None or tpl is None:
                    st.warning("Modèle joblib (`models/anomaly_model.joblib`) et/ou template absent.")
                else:
                    feats = extract_features(img, tpl)
                    X = np.array([[feats[c] for c in C.FEATURE_COLS]])
                    if scaler is not None:
                        X = scaler.transform(X)
                    pred = model.predict(X)[0]
                    verdict = int(pred) in (1, -1)  # -1 = anomalie pour IsolationForest/OneClassSVM
                    st.write("**Features extraites :**", {k: round(feats[k], 3) for k in C.FEATURE_COLS})
                    (st.error if verdict else st.success)(
                        "🔴 DÉFECTUEUSE" if verdict else "🟢 Conforme")

            elif mode.startswith("AutoEncodeur"):
                if cat not in ae_cats:
                    st.warning(f"Aucun autoencodeur entraîné pour « {cat} ». Catégories disponibles : "
                               f"{', '.join(ae_cats) if ae_cats else 'aucune'}.")
                else:
                    ae = load_autoencoder(cat)
                    res = score_autoencoder(img, ae)
                    verdict = res["score"] > (seuil / 1000)  # échelle MSE
                    m1, m2, m3 = st.columns(3)
                    m1.image(img, caption="Entrée", use_container_width=True)
                    m2.image(res["recon"], caption="Reconstruction", use_container_width=True)
                    m3.image(res["err_map"], caption="Carte d'erreur", use_container_width=True, clamp=True)
                    st.metric("Erreur de reconstruction (MSE)", f"{res['score']:.5f}")
                    (st.error if verdict else st.success)(
                        "🔴 DÉFECTUEUSE" if verdict else "🟢 Conforme")

            else:  # CNN supervisé
                cnn = load_cnn(variant) if variant else None
                if cnn is None:
                    st.warning("Aucune variante de CNN disponible dans `models/`.")
                else:
                    res = score_cnn(img, cnn)
                    st.caption("Modèle entraîné sur les 15 catégories confondues (pas de template requis) "
                               "— voir l'ablation Flatten vs GAP et 64×64 vs 128×128, Partie IV du rapport.")
                    st.metric("Probabilité de défaut (sortie sigmoïde)", f"{res['score']:.3f}")
                    st.progress(min(max(res["score"], 0.0), 1.0))
                    (st.error if res["verdict"] else st.success)(
                        "🔴 Pièce probablement DÉFECTUEUSE" if res["verdict"] else "🟢 Pièce conforme")
        except Exception as e:
            st.error(f"Erreur pendant le scoring : {e}")
    else:
        st.info("Dépose une image ci-dessus pour lancer la détection.")

    st.caption("ℹ️ Le mode *baseline* fonctionne dès qu'un template existe. Les autres modes s'appuient "
               "sur des modèles réellement entraînés par l'équipe et versionnés sur GitHub "
               "(`DataScientest-Studio/avr26_bmle_ds_anomalies`).")

# ══════════════════════════════════════════════════════════════════════════
# 5. COMPARATIF DE TOUS LES MODÈLES DE L'ÉQUIPE
# ══════════════════════════════════════════════════════════════════════════
with t_cmp:
    st.header("Comparatif de tous les modèles testés par l'équipe")
    st.caption("Synthèse du rapport final — code complet sur le dépôt GitHub "
               "`DataScientest-Studio/avr26_bmle_ds_anomalies` (branches de chaque membre, fusionnées sur `main`).")

    st.subheader("Partie III — Reconnaissance de la catégorie d'objet")
    st.caption("Jeu de 7 590 fichiers (391 images/catégorie après augmentation, 128×128×3).")
    reco_df = pd.DataFrame({
        "Modèle": ["Random Forest", "RNN Dense", "CNN", "LeNet", "AutoEncoder (test sur bottle)"],
        "Auteur(s)": ["Fabrice", "Fabrice", "Fabrice / Paul", "Fabrice", "Fabrice"],
        "Principe": ["Sur features engineered", "Réseau dense", "Convolution + pooling",
                     "Architecture LeNet", "Reconstruction"],
        "Erreurs de classement": ["47 images", "< 47 (non chiffré)", "4 images", "0", "0 (83/83 bottles)"],
    })
    st.dataframe(reco_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Partie IV — Détection d'anomalies (métrique : AUC-ROC)")
    st.caption("Moyenne sur les 15 catégories, sauf mention contraire. Source : rapport final de l'équipe.")

    modele_df = pd.DataFrame({
        "Modèle": ["AutoEncoder convolutif", "CNN scratch — 64×64 (Flatten)",
                   "CNN scratch — 128×128 (Flatten)", "CNN scratch — 64×64 (GAP, ablation)",
                   "Transfer Learning — EfficientNetB0", "Transfer Learning — ResNet50",
                   "PaDiM (features pré-entraînées)"],
        "Auteur(s)": ["Fabrice", "Paul (VM Liora)", "Ludovic", "Paul (ablation)", "Alex", "Alex", "Alex"],
        "Protocole": ["Non supervisé (one-class)", "Supervisé", "Supervisé", "Supervisé",
                      "Supervisé", "Supervisé", "Non supervisé (one-class)"],
        "AUC-ROC moyen": [0.718, 0.748, 0.7295, 0.542, 0.937, 0.930, None],
        "Disponible en démo live": ["✅ (bottle, carpet, screw)", "✅", "✅", "✅", "❌ poids non versionnés",
                                     "❌ poids non versionnés", "❌ non finalisé"],
    })
    st.dataframe(modele_df, use_container_width=True, hide_index=True)

    import matplotlib.pyplot as plt
    plot_df = modele_df.dropna(subset=["AUC-ROC moyen"]).sort_values("AUC-ROC moyen")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = [C.COLOR_ANOMALY if v < 0.6 else (C.COLOR_PRIMARY if v > 0.9 else "#8FA8D6")
              for v in plot_df["AUC-ROC moyen"]]
    ax.barh(plot_df["Modèle"], plot_df["AUC-ROC moyen"], color=colors)
    for i, v in enumerate(plot_df["AUC-ROC moyen"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC-ROC moyen (15 catégories)")
    ax.axvline(0.99, color="#555", linestyle="--", linewidth=1)
    ax.text(0.99, -0.7, "État de l'art\n(PatchCore) ≈ 0.99", fontsize=8, color="#555", ha="center")
    st.pyplot(fig)

    st.info("**Ce qui a le plus fait progresser la performance :** le passage d'une approche non "
            "supervisée pure (AutoEncoder) au transfer learning supervisé (EfficientNetB0, ResNet50) — "
            "les features génériques ImageNet généralisent nettement mieux, en particulier sur les "
            "textures répétitives où l'AutoEncoder et le CNN from scratch échouaient le plus.")
    st.caption("EfficientNetB0/ResNet50/PaDiM ne sont pas chargés en démo live : leurs poids (plusieurs "
               "centaines de Mo par catégorie) n'ont pas été versionnés sur GitHub — voir le rapport final "
               "pour le détail complet des courbes ROC et matrices de confusion par catégorie.")

# ══════════════════════════════════════════════════════════════════════════
# 6. CONCLUSION
# ══════════════════════════════════════════════════════════════════════════
with t_ccl:
    st.header("Conclusion & perspectives")
    st.markdown("""
**Ce qui marche.** La **différence au template** sépare bien conformes et défectueux sur les
surfaces lisses (`bottle`), confirmé statistiquement (delta de Cliff |δ| > 0,5, p < 10⁻¹⁵).
Les features de texture (GLCM, LBP) complètent pour les surfaces répétitives (`carpet`).
Le **transfer learning** (EfficientNetB0/ResNet50) surclasse largement l'approche non
supervisée pure (AUC moyen 0,937 contre 0,718 pour l'AutoEncoder).

**Limites.** Catégories à forte variabilité d'orientation (`screw`) ou à texture répétitive
(`carpet`, `tile`) plus difficiles pour les approches non supervisées. Absence de GPU sur la VM
Liora : profondeur des architectures et nombre d'epochs limités.

**Perspectives (voir rapport final, Partie V).**
- Finaliser **PaDiM / PatchCore** (état de l'art, AUC ≈ 0,99) sur les 15 catégories.
- Ajuster le **seuil de décision par catégorie** via la courbe ROC plutôt qu'un seuil fixe.
- Infrastructure **GPU / PyTorch** pour tester des architectures plus profondes.
- Étendre la **classification multi-classes** du type de défaut, aujourd'hui esquissée.
    """)
    st.success("Applicable en contrôle qualité **en complément — plutôt qu'en remplacement —** "
               "d'un contrôle humain, en priorisant l'inspection manuelle sur les pièces à fort score d'anomalie.")
    st.caption("Rapport final, notebooks et code complet : dépôt GitHub d'équipe "
               "`DataScientest-Studio/avr26_bmle_ds_anomalies` + Google Drive.")
