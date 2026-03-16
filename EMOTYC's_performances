
```python
import pandas as pd

df1 = pd.read_excel('/content/final_gold.xlsx')
df2 = pd.read_excel('/content/final_predictions_2.xlsx')
```


# Calcul de l'AIA


```python
import pandas as pd
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

# ──────────────────────────────────────────────
# 1. Lecture des fichiers XLSX
# ──────────────────────────────────────────────
gold = pd.read_excel("./gold_labels.xlsx")
pred = pd.read_excel("./predictions.xlsx")

# ──────────────────────────────────────────────
# 2. Identification des 11 étiquettes communes
# ──────────────────────────────────────────────
# Colonnes présentes dans les deux fichiers
common_cols = sorted(set(gold.columns) & set(pred.columns))

# On exclut les colonnes qui ne sont pas des étiquettes émotionnelles
#   (adapter cette liste si vos fichiers contiennent d'autres méta-colonnes)
NON_LABEL_COLS = {"id", "text", "sentence", "source", "index", "Unnamed: 0"}
emotion_labels = [c for c in common_cols if c not in NON_LABEL_COLS]

print(f"Étiquettes émotionnelles communes trouvées ({len(emotion_labels)}) :")
print(emotion_labels)
assert len(emotion_labels) == 11, (
    f"Attendu 11 étiquettes, trouvé {len(emotion_labels)}. "
    f"Vérifiez NON_LABEL_COLS ou vos fichiers."
)

# ──────────────────────────────────────────────
# 3. Alignement des lignes (même ordre)
# ──────────────────────────────────────────────
# Si une colonne 'id' ou 'text' existe, on aligne dessus
id_col = None
for candidate in ("id", "text", "sentence"):
    if candidate in gold.columns and candidate in pred.columns:
        id_col = candidate
        break

if id_col:
    print(f"\nAlignement des lignes sur la colonne « {id_col} »")
    merged = gold[[id_col] + emotion_labels].merge(
        pred[[id_col] + emotion_labels],
        on=id_col,
        suffixes=("_gold", "_pred"),
    )
    y_gold = merged[[f"{l}_gold" for l in emotion_labels]].values
    y_pred = merged[[f"{l}_pred" for l in emotion_labels]].values
    n_samples = len(merged)
else:
    # Pas de colonne d'identification → on suppose le même ordre
    print("\nPas de colonne d'alignement : on suppose le même ordre de lignes.")
    assert len(gold) == len(pred), "Les deux fichiers n'ont pas le même nombre de lignes."
    y_gold = gold[emotion_labels].values
    y_pred = pred[emotion_labels].values
    n_samples = len(gold)

print(f"Nombre d'exemples comparés : {n_samples}\n")

# ──────────────────────────────────────────────
# 4. Calcul de l'AIA par étiquette
# ──────────────────────────────────────────────
print("=" * 75)
print(f"{'Étiquette':<20} {'Accuracy':>9} {'Kappa':>9} {'F1':>9} {'Precision':>10} {'Recall':>9}")
print("-" * 75)

kappas, f1s, accs = [], [], []

for i, label in enumerate(emotion_labels):
    g = y_gold[:, i]
    p = y_pred[:, i]

    acc = accuracy_score(g, p)
    kappa = cohen_kappa_score(g, p)
    f1 = f1_score(g, p, zero_division=0)
    prec = precision_score(g, p, zero_division=0)
    rec = recall_score(g, p, zero_division=0)

    accs.append(acc)
    kappas.append(kappa)
    f1s.append(f1)

    print(f"{label:<20} {acc:>9.4f} {kappa:>9.4f} {f1:>9.4f} {prec:>10.4f} {rec:>9.4f}")

print("-" * 75)

# ──────────────────────────────────────────────
# 5. Métriques globales
# ──────────────────────────────────────────────
macro_kappa = np.mean(kappas)
macro_f1 = np.mean(f1s)
macro_acc = np.mean(accs)

# Exact-match (toutes les 11 étiquettes identiques pour un exemple)
exact_match = np.all(y_gold == y_pred, axis=1).mean()

# Micro-F1 (tous les labels mis à plat)
micro_f1 = f1_score(y_gold.ravel(), y_pred.ravel(), zero_division=0)

print(f"{'MACRO-MOYENNE':<20} {macro_acc:>9.4f} {macro_kappa:>9.4f} {macro_f1:>9.4f}")
print(f"{'MICRO-F1':<20} {'':>9} {'':>9} {micro_f1:>9.4f}")
print(f"{'EXACT MATCH':<20} {exact_match:>9.4f}")
print("=" * 75)

# ──────────────────────────────────────────────
# 6. (Optionnel) Export des résultats
# ──────────────────────────────────────────────
results_df = pd.DataFrame({
    "label": emotion_labels,
    "accuracy": accs,
    "cohen_kappa": kappas,
    "f1": f1s,
})
results_df.to_excel("./aia_results.xlsx", index=False)
print("\nRésultats exportés dans ./aia_results.xlsx")#!/usr/bin/env python3
"""
Calcul de l'Accord Inter-Annotateurs (AIA) entre deux fichiers XLSX
contenant des annotations émotionnelles.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

sns.set_theme(style="whitegrid", font_scale=1.05)

# Les 11 étiquettes émotionnelles communes
EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Calcul de l'AIA entre gold labels et prédictions (XLSX)"
    )
    p.add_argument(
        "--gold",
        default="./gold_labels.xlsx",
        help="XLSX des annotations gold (défaut: ./gold_labels.xlsx)"
    )
    p.add_argument(
        "--pred",
        default="./predictions.xlsx",
        help="XLSX des prédictions (défaut: ./predictions.xlsx)"
    )
    p.add_argument(
        "--out_dir",
        default="./resultats_aia",
        help="Dossier de sortie (défaut: ./resultats_aia)"
    )
    p.add_argument(
        "--label_gold",
        default="Gold",
        help="Label pour les annotations gold"
    )
    p.add_argument(
        "--label_pred",
        default="Pred",
        help="Label pour les prédictions"
    )
    p.add_argument(
        "--id_col",
        default=None,
        help="Nom de la colonne ID pour le merge (auto-détecté si non spécifié)"
    )
    p.add_argument(
        "--text_col",
        default=None,
        help="Nom de la colonne texte (auto-détecté si non spécifié)"
    )
    return p.parse_args()


def load_emotions_from_xlsx(path, emotions=EMOTIONS):
    """
    Charge un fichier XLSX et extrait les colonnes émotionnelles.
    Retourne un DataFrame avec les émotions et les colonnes d'identification.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier non trouvé: {path}")

    df = pd.read_excel(path)
    print(f"  Chargé: {path}")
    print(f"    → {len(df)} lignes, {len(df.columns)} colonnes")

    # Vérifier quelles émotions sont présentes
    emotions_found = [e for e in emotions if e in df.columns]
    emotions_missing = [e for e in emotions if e not in df.columns]

    if emotions_missing:
        print(f"    ⚠ Émotions manquantes: {emotions_missing}")
    print(f"    ✓ Émotions trouvées: {len(emotions_found)}/{len(emotions)}")

    return df, emotions_found


def find_common_columns(df1, df2, exclude_emotions=True):
    """
    Trouve les colonnes communes entre deux DataFrames pour le merge.
    """
    common = set(df1.columns) & set(df2.columns)

    if exclude_emotions:
        common = common - set(EMOTIONS)

    # Priorité aux colonnes d'ID typiques
    id_candidates = ['idx', 'id', 'ID', 'index', 'row_id', 'message_id']
    for col in id_candidates:
        if col in common:
            return col

    # Sinon, utiliser l'index
    return None


def auto_detect_text_column(df):
    """
    Détecte automatiquement la colonne contenant le texte.
    """
    text_candidates = ['TEXT', 'text', 'Text', 'message', 'MESSAGE', 'content', 'CONTENT']
    for col in text_candidates:
        if col in df.columns:
            return col
    return None


def main():
    args = parse_args()

    # Créer le dossier de sortie
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    lg, lp = args.label_gold, args.label_pred

    print(f"\n{'='*70}")
    print(f"  CALCUL DE L'ACCORD INTER-ANNOTATEURS (AIA)")
    print(f"  {lg} vs {lp}")
    print(f"{'='*70}\n")

    # ── Chargement des fichiers XLSX ──
    print("Chargement des fichiers...")
    df_gold, emo_gold = load_emotions_from_xlsx(args.gold)
    df_pred, emo_pred = load_emotions_from_xlsx(args.pred)

    # Émotions communes aux deux fichiers
    emotions_common = [e for e in EMOTIONS if e in emo_gold and e in emo_pred]
    print(f"\n✓ {len(emotions_common)} émotions communes: {emotions_common}")

    if len(emotions_common) == 0:
        print("⚠ Aucune émotion commune trouvée!")
        return

    # ── Merge des DataFrames ──
    print("\nFusion des fichiers...")

    # Déterminer la colonne de merge
    id_col = args.id_col or find_common_columns(df_gold, df_pred)

    if id_col and id_col in df_gold.columns and id_col in df_pred.columns:
        print(f"  → Merge sur la colonne: '{id_col}'")
        merged = pd.merge(
            df_gold, df_pred,
            on=id_col,
            how="inner",
            suffixes=("_gold", "_pred")
        )
    else:
        # Merge par index (position)
        print("  → Merge par position (index)")
        min_len = min(len(df_gold), len(df_pred))
        df_gold_subset = df_gold.iloc[:min_len].copy()
        df_pred_subset = df_pred.iloc[:min_len].copy()

        # Renommer les colonnes émotionnelles
        gold_rename = {e: f"{e}_gold" for e in emotions_common}
        pred_rename = {e: f"{e}_pred" for e in emotions_common}

        df_gold_subset = df_gold_subset.rename(columns=gold_rename)
        df_pred_subset = df_pred_subset.rename(columns=pred_rename)

        # Ajouter les colonnes de prédiction au gold
        for e in emotions_common:
            df_gold_subset[f"{e}_pred"] = df_pred_subset[f"{e}_pred"].values

        merged = df_gold_subset
        merged['idx'] = range(len(merged))
        id_col = 'idx'

    N = len(merged)
    print(f"✓ {N} messages comparables")

    if N == 0:
        print("⚠ Aucun message comparable!")
        return

    # Détecter la colonne texte
    text_col = args.text_col or auto_detect_text_column(merged)
    if text_col:
        # Gérer les colonnes dupliquées après merge
        if f"{text_col}_gold" in merged.columns:
            text_col = f"{text_col}_gold"
        elif text_col not in merged.columns:
            text_col = None

    # ── Calcul des métriques par émotion ──
    print(f"\n{'='*70}")
    print(f"  MÉTRIQUES D'ACCORD PAR ÉMOTION")
    print(f"{'='*70}\n")

    stats_rows = []
    for e in emotions_common:
        col_gold = f"{e}_gold" if f"{e}_gold" in merged.columns else e
        col_pred = f"{e}_pred" if f"{e}_pred" in merged.columns else e

        # Convertir en int et gérer les NaN
        v_gold = pd.to_numeric(merged[col_gold], errors='coerce').fillna(0).astype(int).values
        v_pred = pd.to_numeric(merged[col_pred], errors='coerce').fillna(0).astype(int).values

        # Métriques
        agree = (v_gold == v_pred).sum()
        gold_only = ((v_gold == 1) & (v_pred == 0)).sum()  # Faux négatifs
        pred_only = ((v_gold == 0) & (v_pred == 1)).sum()  # Faux positifs

        # Cohen's Kappa
        try:
            kappa = cohen_kappa_score(v_gold, v_pred)
        except:
            kappa = np.nan

        # Prévalence
        prevalence_gold = v_gold.sum() / N * 100
        prevalence_pred = v_pred.sum() / N * 100

        stats_rows.append({
            "Émotion": e,
            "Accord (%)": round(agree / N * 100, 1),
            "Kappa": round(kappa, 3) if not np.isnan(kappa) else "N/A",
            f"{lg} seul (FN)": gold_only,
            f"{lp} seul (FP)": pred_only,
            "Divergences": gold_only + pred_only,
            f"Préval. {lg} (%)": round(prevalence_gold, 1),
            f"Préval. {lp} (%)": round(prevalence_pred, 1),
        })

    df_stats = pd.DataFrame(stats_rows)
    print(df_stats.to_string(index=False))

    # ── Exact match (toutes émotions identiques) ──
    exact_matches = []
    n_divergences_list = []

    for idx, row in merged.iterrows():
        match_count = 0
        for e in emotions_common:
            col_gold = f"{e}_gold" if f"{e}_gold" in merged.columns else e
            col_pred = f"{e}_pred" if f"{e}_pred" in merged.columns else e

            v1 = int(pd.to_numeric(row[col_gold], errors='coerce') or 0)
            v2 = int(pd.to_numeric(row[col_pred], errors='coerce') or 0)

            if v1 == v2:
                match_count += 1

        exact_matches.append(match_count == len(emotions_common))
        n_divergences_list.append(len(emotions_common) - match_count)

    merged["exact_match"] = exact_matches
    merged["n_divergences"] = n_divergences_list

    n_exact = sum(exact_matches)

    print(f"\n{'='*70}")
    print(f"  RÉSUMÉ GLOBAL")
    print(f"{'='*70}")
    print(f"  Messages comparés     : {N}")
    print(f"  Exact match           : {n_exact}/{N} ({n_exact/N*100:.1f}%)")
    print(f"  Avec ≥1 divergence    : {N - n_exact}/{N} ({(N-n_exact)/N*100:.1f}%)")

    # Accord moyen
    accord_values = [r["Accord (%)"] for r in stats_rows]
    kappa_values = [r["Kappa"] for r in stats_rows if r["Kappa"] != "N/A"]

    print(f"\n  Accord moyen          : {np.mean(accord_values):.1f}%")
    if kappa_values:
        print(f"  Kappa moyen           : {np.mean(kappa_values):.3f}")

    # ── Interprétation du Kappa ──
    print(f"\n  Interprétation du Kappa de Cohen:")
    print(f"    < 0.00  : Désaccord")
    print(f"    0.00-0.20: Accord très faible")
    print(f"    0.21-0.40: Accord faible")
    print(f"    0.41-0.60: Accord modéré")
    print(f"    0.61-0.80: Accord substantiel")
    print(f"    0.81-1.00: Accord presque parfait")

    # ── Figures ──
    print(f"\nGénération des figures...")

    # Figure 1: Accord par émotion
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if v >= 90 else "#f39c12" if v >= 80 else "#e74c3c"
              for v in df_stats["Accord (%)"]]

    y_pos = range(len(emotions_common))
    bars = ax.barh(
        [e for e in reversed(emotions_common)],
        df_stats["Accord (%)"].values[::-1],
        color=colors[::-1],
        edgecolor="black",
        linewidth=0.5
    )

    ax.axvline(90, color="green", ls="--", lw=1, alpha=0.7, label="Seuil 90%")
    ax.axvline(80, color="orange", ls="--", lw=1, alpha=0.7, label="Seuil 80%")
    ax.set_xlabel("Accord (%)")
    ax.set_xlim(0, 105)
    ax.set_title(f"Accord Inter-Annotateurs par Émotion\n({lg} vs {lp})")
    ax.legend(loc="lower right")

    for bar, val in zip(bars, df_stats["Accord (%)"].values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accord_par_emotion.png"), dpi=150)

    # Figure 2: Kappa par émotion
    fig, ax = plt.subplots(figsize=(10, 6))
    kappa_vals = []
    for k in df_stats["Kappa"]:
        if k == "N/A":
            kappa_vals.append(0)
        else:
            kappa_vals.append(float(k))

    colors_kappa = []
    for k in kappa_vals:
        if k >= 0.8:
            colors_kappa.append("#2ecc71")  # Excellent
        elif k >= 0.6:
            colors_kappa.append("#3498db")  # Bon
        elif k >= 0.4:
            colors_kappa.append("#f39c12")  # Modéré
        else:
            colors_kappa.append("#e74c3c")  # Faible

    bars = ax.barh(
        [e for e in reversed(emotions_common)],
        kappa_vals[::-1],
        color=colors_kappa[::-1],
        edgecolor="black",
        linewidth=0.5
    )

    ax.axvline(0.8, color="green", ls="--", lw=1, alpha=0.7, label="κ=0.8 (excellent)")
    ax.axvline(0.6, color="blue", ls="--", lw=1, alpha=0.7, label="κ=0.6 (bon)")
    ax.axvline(0.4, color="orange", ls="--", lw=1, alpha=0.7, label="κ=0.4 (modéré)")
    ax.set_xlabel("Cohen's Kappa (κ)")
    ax.set_xlim(-0.1, 1.05)
    ax.set_title(f"Cohen's Kappa par Émotion\n({lg} vs {lp})")
    ax.legend(loc="lower right", fontsize=8)

    for bar, val in zip(bars, kappa_vals[::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kappa_par_emotion.png"), dpi=150)

    # Figure 3: Divergences directionnelles
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(emotions_common))
    w = 0.35

    fn_vals = df_stats[f"{lg} seul (FN)"].values
    fp_vals = df_stats[f"{lp} seul (FP)"].values

    ax.barh(x + w/2, fn_vals, w, label=f"Faux Négatifs ({lg} seul)", color="#3498db")
    ax.barh(x - w/2, fp_vals, w, label=f"Faux Positifs ({lp} seul)", color="#e67e22")

    ax.set_yticks(x)
    ax.set_yticklabels(emotions_common)
    ax.set_xlabel("Nombre de messages")
    ax.set_title(f"Divergences Directionnelles\n({lg} vs {lp})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "divergences_direction.png"), dpi=150)

    # Figure 4: Distribution des divergences
    fig, ax = plt.subplots(figsize=(8, 5))
    max_div = max(n_divergences_list) if n_divergences_list else len(emotions_common)

    ax.hist(
        n_divergences_list,
        bins=range(0, max_div + 2),
        align="left",
        color="#9b59b6",
        edgecolor="black",
        linewidth=0.5
    )
    ax.set_xlabel("Nombre d'émotions divergentes")
    ax.set_ylabel("Nombre de messages")
    ax.set_title(f"Distribution du nombre de divergences par message\n({lg} vs {lp})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_divergences.png"), dpi=150)

    plt.close("all")
    print(f"✓ Figures sauvegardées dans {out_dir}/")

    # ── Export XLSX des résultats ──
    print(f"\nExport des résultats...")

    # Préparer les exemples de divergences
    df_div = merged[merged["n_divergences"] > 0].sort_values("n_divergences", ascending=False)

    rows_exp = []
    for idx_row, row in df_div.iterrows():
        r = {
            "idx": row.get(id_col, idx_row) if id_col else idx_row,
            "n_divergences": row["n_divergences"]
        }

        # Ajouter le texte si disponible
        if text_col and text_col in row:
            r["text"] = str(row[text_col])[:500]  # Limiter la longueur

        # Ajouter les valeurs pour chaque émotion
        for e in emotions_common:
            col_gold = f"{e}_gold" if f"{e}_gold" in merged.columns else e
            col_pred = f"{e}_pred" if f"{e}_pred" in merged.columns else e

            v1 = int(pd.to_numeric(row[col_gold], errors='coerce') or 0)
            v2 = int(pd.to_numeric(row[col_pred], errors='coerce') or 0)

            r[f"{e}_{lg}"] = v1
            r[f"{e}_{lp}"] = v2
            r[f"{e}_match"] = "✓" if v1 == v2 else "✗"

        rows_exp.append(r)

    out_xlsx = os.path.join(out_dir, "resultats_aia.xlsx")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # Feuille 1: Résumé
        resume_data = {
            "Métrique": [
                "Messages comparés",
                "Exact match",
                "Exact match (%)",
                "Messages avec divergences",
                "Divergences (%)",
                "Accord moyen (%)",
                "Kappa moyen"
            ],
            "Valeur": [
                N,
                n_exact,
                round(n_exact/N*100, 1),
                N - n_exact,
                round((N-n_exact)/N*100, 1),
                round(np.mean(accord_values), 1),
                round(np.mean(kappa_values), 3) if kappa_values else "N/A"
            ]
        }
        pd.DataFrame(resume_data).to_excel(writer, sheet_name="Résumé", index=False)

        # Feuille 2: Accord par émotion
        df_stats.to_excel(writer, sheet_name="Accord par émotion", index=False)

        # Feuille 3: Divergences détaillées
        if rows_exp:
            pd.DataFrame(rows_exp).to_excel(writer, sheet_name="Divergences", index=False)

        # Feuille 4: Tous les messages
        cols_export = [id_col] if id_col else []
        cols_export += ["exact_match", "n_divergences"]
        for e in emotions_common:
            col_gold = f"{e}_gold" if f"{e}_gold" in merged.columns else e
            col_pred = f"{e}_pred" if f"{e}_pred" in merged.columns else e
            cols_export.extend([col_gold, col_pred])

        cols_export = [c for c in cols_export if c in merged.columns]
        merged[cols_export].to_excel(writer, sheet_name="Tous messages", index=False)

    print(f"✓ XLSX exporté → {out_xlsx}")

    # ── Affichage des exemples de divergences ──
    print(f"\n{'='*70}")
    print(f"  EXEMPLES DE DIVERGENCES (top 10)")
    print(f"{'='*70}")

    for i, (idx_row, row) in enumerate(df_div.head(10).iterrows()):
        idx_val = row.get(id_col, idx_row) if id_col else idx_row
        print(f"\n  [{i+1}] idx={idx_val}, divergences={row['n_divergences']}")

        if text_col and text_col in row:
            text = str(row[text_col])[:100]
            print(f"      Texte: \"{text}...\"")

        for e in emotions_common:
            col_gold = f"{e}_gold" if f"{e}_gold" in merged.columns else e
            col_pred = f"{e}_pred" if f"{e}_pred" in merged.columns else e

            v1 = int(pd.to_numeric(row[col_gold], errors='coerce') or 0)
            v2 = int(pd.to_numeric(row[col_pred], errors='coerce') or 0)

            if v1 != v2:
                print(f"      {e:15s}  {lg}={v1}  {lp}={v2}")

    print(f"\n{'='*70}")
    print(f"  ANALYSE TERMINÉE")
    print(f"  Résultats disponibles dans: {out_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
```
```output
Étiquettes émotionnelles communes trouvées (11) :
['admiration', 'colere', 'culpabilite', 'degout', 'embarras', 'fierte', 'jalousie', 'joie', 'peur', 'surprise', 'tristesse']

Pas de colonne d'alignement : on suppose le même ordre de lignes.
Nombre d'exemples comparés : 101

===========================================================================
Étiquette             Accuracy     Kappa        F1  Precision    Recall
---------------------------------------------------------------------------
admiration              0.8218    0.0000    0.0000     0.0000    0.0000
colere                  0.5545    0.1591    0.5714     0.7500    0.4615
culpabilite             1.0000       nan    0.0000     0.0000    0.0000
degout                  0.4059    0.0000    0.0000     0.0000    0.0000
embarras                0.9505    0.0000    0.0000     0.0000    0.0000
fierte                  0.9802    0.0000    0.0000     0.0000    0.0000
jalousie                1.0000       nan    0.0000     0.0000    0.0000
joie                    0.9109    0.1544    0.1818     0.5000    0.1111
peur                    1.0000       nan    0.0000     0.0000    0.0000
surprise                0.9604   -0.0151    0.0000     0.0000    0.0000
tristesse               0.8713   -0.0186    0.0000     0.0000    0.0000
---------------------------------------------------------------------------
MACRO-MOYENNE           0.8596       nan    0.0685
MICRO-F1                                    0.2844
EXACT MATCH             0.1188
===========================================================================
```
