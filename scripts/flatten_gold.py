#!/usr/bin/env python3
"""
Aplatissement des annotations SitEmo span-level → gold labels phrase-level.

Lit un fichier XLSX validé (issu de supervise.py) contenant des annotations
span-level (spans_json, span*_cat, span*_mode) et produit un nouveau XLSX
avec des colonnes binaires agrégées via OR logique :

  · 12 colonnes émotions  : Colère, Dégoût, Joie, Peur, Surprise, Tristesse,
                            Admiration, Culpabilité, Embarras, Fierté, Jalousie, Autre
  · 4 colonnes modes      : Comportementale, Désignée, Montrée, Suggérée
  · 1 colonne Emo         : caractère émotionnel (1 si ≥1 span, 0 sinon)
  · 2 colonnes type       : Base, Complexe

Le fichier produit est directement utilisable par emotyc_predict.py pour
évaluer les prédictions EMOTYC sur les émotions ET les modes d'expression.

Usage :
    python scripts/flatten_gold.py \\
        --input /home/gwen/Downloads/annotations_validees.xlsx \\
        --output outputs/homophobie/annotations_gold_flat.xlsx
"""

import argparse
import json
import math
import os
import sys

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

# 12 catégories émotionnelles (ordre canonique)
EMOTION_ORDER = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie", "Autre",
]

# 4 modes d'expression (noms avec accents pour lisibilité)
MODE_ORDER = ["Comportementale", "Désignée", "Montrée", "Suggérée"]

# Émotions de base vs complexes (cf. schéma d'annotation)
BASE_EMOTIONS = {"Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse"}
COMPLEX_EMOTIONS = {"Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie"}
# "Autre" n'est ni Base ni Complexe → pas activé pour Base/Complexe

# Mapping des émotions fines → catégories canoniques (cf. Annexe 1 du schéma)
# Les 12 catégories canoniques se mappent à elles-mêmes ; les labels fins
# sont résolus vers leur catégorie parente.
FINE_TO_CANONICAL = {
    # Colère
    "Agacement": "Colère", "Contestation": "Colère", "Désaccord": "Colère",
    "Désapprobation": "Colère", "Énervement": "Colère", "Fureur": "Colère",
    "Rage": "Colère", "Indignation": "Colère", "Insatisfaction": "Colère",
    "Irritation": "Colère", "Mécontentement": "Colère", "Réprobation": "Colère",
    "Révolte": "Colère",
    # Dégoût
    "Lassitude": "Dégoût", "Répulsion": "Dégoût",
    # Joie
    "Amusement": "Joie", "Enthousiasme": "Joie", "Exaltation": "Joie",
    "Plaisir": "Joie",
    # Peur
    "Angoisse": "Peur", "Appréhension": "Peur", "Effroi": "Peur",
    "Horreur": "Peur", "Inquiétude": "Peur", "Méfiance": "Peur",
    "Stress": "Peur",
    # Surprise
    "Étonnement": "Surprise", "Stupeur": "Surprise",
    # Tristesse
    "Blues": "Tristesse", "Chagrin": "Tristesse", "Déception": "Tristesse",
    "Désespoir": "Tristesse", "Peine": "Tristesse", "Souffrance": "Tristesse",
    # Embarras
    "Gêne": "Embarras", "Honte": "Embarras", "Humiliation": "Embarras",
    # Fierté
    "Orgueil": "Fierté",
    # Timidité → Peur ET Embarras (double mapping)
    "Timidité": "Peur",  # also Embarras, handled specially below
    # Autre
    "Amour": "Autre", "Courage": "Autre", "Curiosité": "Autre",
    "Désir": "Autre", "Détermination": "Autre", "Envie": "Autre",
    "Espoir": "Autre", "Haine": "Autre", "Impuissance": "Autre",
    "Mépris": "Autre", "Soulagement": "Autre",
}
# Labels qui activent deux catégories à la fois
FINE_DUAL_MAP = {
    "Timidité": ["Peur", "Embarras"],
}


def _resolve_category(cat):
    """Résout un label d'émotion (fin ou canonique) vers la/les catégorie(s) canonique(s)."""
    if cat in dict.fromkeys(EMOTION_ORDER):
        return [cat]
    if cat in FINE_DUAL_MAP:
        return FINE_DUAL_MAP[cat]
    canonical = FINE_TO_CANONICAL.get(cat)
    if canonical:
        return [canonical]
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  LOGIQUE DE FLATTENING
# ═══════════════════════════════════════════════════════════════════════════

def flatten_row(spans_json_str, n_spans):
    """
    Aplatit les annotations span-level d'une ligne en vecteurs binaires
    phrase-level via OR logique.

    Arguments :
        spans_json_str  — contenu de la colonne spans_json (str ou NaN)
        n_spans         — nombre de spans (int ou NaN)

    Retourne :
        emotions  — dict {nom_émotion: 0|1}
        modes     — dict {nom_mode: 0|1}
        emo       — 0|1 (caractère émotionnel)
        base      — 0|1 (contient au moins une émotion de base)
        complexe  — 0|1 (contient au moins une émotion complexe)
    """
    emotions = {e: 0 for e in EMOTION_ORDER}
    modes = {m: 0 for m in MODE_ORDER}

    # Pas de spans → tout à 0
    ns = 0 if (n_spans is None or (isinstance(n_spans, float) and math.isnan(n_spans))) else int(n_spans)
    if ns == 0 or not isinstance(spans_json_str, str) or not spans_json_str.strip():
        return emotions, modes, 0, 0, 0

    try:
        spans = json.loads(spans_json_str)
    except (json.JSONDecodeError, TypeError):
        print(f"  ⚠ Impossible de parser spans_json : {spans_json_str[:80]}…",
              file=sys.stderr)
        return emotions, modes, 0, 0, 0

    if not isinstance(spans, list) or len(spans) == 0:
        return emotions, modes, 0, 0, 0

    unresolved = set()
    for span in spans:
        # ── OR sur les catégories émotionnelles ──
        for cat_key in ("categorie", "categorie2"):
            cat = span.get(cat_key)
            if not cat:
                continue
            resolved = _resolve_category(cat)
            if resolved:
                for r in resolved:
                    emotions[r] = 1
            else:
                unresolved.add(cat)

        # ── OR sur les modes ──
        mode = span.get("mode")
        if mode and mode in modes:
            modes[mode] = 1

    if unresolved:
        print(f"  ⚠ Catégorie(s) non résolue(s) : {unresolved}", file=sys.stderr)

    # ── Emo : 1 si au moins un span ──
    emo = 1

    # ── Base / Complexe ──
    base = 1 if any(emotions[e] for e in BASE_EMOTIONS) else 0
    complexe = 1 if any(emotions[e] for e in COMPLEX_EMOTIONS) else 0

    return emotions, modes, emo, base, complexe


def flatten_dataframe(df):
    """
    Applique le flattening sur tout le DataFrame.
    Met à jour les colonnes émotions existantes et ajoute les colonnes
    manquantes (modes, Emo, Base, Complexe).
    """
    # Préparer les colonnes de sortie
    all_emotions = {e: [] for e in EMOTION_ORDER}
    all_modes = {m: [] for m in MODE_ORDER}
    all_emo = []
    all_base = []
    all_complexe = []

    for i in range(len(df)):
        spans_json = df.iloc[i].get("spans_json")
        n_spans = df.iloc[i].get("n_spans", 0)

        emotions, modes, emo, base, complexe = flatten_row(spans_json, n_spans)

        for e in EMOTION_ORDER:
            all_emotions[e].append(emotions[e])
        for m in MODE_ORDER:
            all_modes[m].append(modes[m])
        all_emo.append(emo)
        all_base.append(base)
        all_complexe.append(complexe)

    # Mettre à jour / créer les colonnes
    for e in EMOTION_ORDER:
        df[e] = all_emotions[e]
    for m in MODE_ORDER:
        df[m] = all_modes[m]
    df["Emo"] = all_emo
    df["Base"] = all_base
    df["Complexe"] = all_complexe

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Aplatissement SitEmo span-level → gold labels phrase-level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True,
                    help="Fichier XLSX validé (issu de supervise.py)")
    p.add_argument("--output", required=True,
                    help="Fichier XLSX de sortie avec gold labels aplatis")
    p.add_argument("--keep-run-columns", action="store_true",
                    help="Conserver les colonnes *_run1/*_run2 dans la sortie")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Chargement ─────────────────────────────────────────────────
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"✗ Fichier introuvable : {input_path}")
        sys.exit(1)

    df = pd.read_excel(input_path)
    print(f"✓ Chargé : {input_path}")
    print(f"  {len(df)} lignes, {len(df.columns)} colonnes")

    # Vérifier les colonnes nécessaires
    required = ["spans_json", "n_spans"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"✗ Colonnes manquantes : {missing}")
        sys.exit(1)

    # ── 2. Flattening ─────────────────────────────────────────────────
    print("\n▸ Aplatissement des annotations span-level → phrase-level…")
    df = flatten_dataframe(df)

    # ── 3. Statistiques ──────────────────────────────────────────────
    print("\n  Statistiques après aplatissement :")
    print(f"  {'Label':<20s} {'#1':>5s} {'#0':>5s} {'%':>7s}")
    print(f"  {'-'*40}")

    for label in ["Emo"] + EMOTION_ORDER + MODE_ORDER + ["Base", "Complexe"]:
        n1 = int(df[label].sum())
        n0 = len(df) - n1
        pct = n1 / len(df) * 100
        print(f"  {label:<20s} {n1:>5d} {n0:>5d} {pct:>6.1f}%")

    # ── 4. Suppression optionnelle des colonnes run ───────────────────
    if not args.keep_run_columns:
        run_cols = [c for c in df.columns if c.endswith("_run1") or c.endswith("_run2")]
        if run_cols:
            df = df.drop(columns=run_cols)
            print(f"\n  Supprimé {len(run_cols)} colonnes *_run1/*_run2")

    # ── 5. Export ────────────────────────────────────────────────────
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\n✓ Exporté : {output_path}")
    print(f"  {len(df)} lignes, {len(df.columns)} colonnes")

    # ── 6. Résumé des colonnes ───────────────────────────────────────
    print(f"\n  Colonnes du fichier produit :")
    for i, col in enumerate(df.columns):
        print(f"    [{i:>2d}] {col}")


if __name__ == "__main__":
    main()
