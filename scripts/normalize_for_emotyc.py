#!/usr/bin/env python3
"""
Script intermédiaire — Normalise un XLSX agrégé pour le rendre compatible
avec emotyc_predict.py.

Le fichier agrégé (issu de aggregate.py) utilise des noms de colonnes sans
accents (Colere, Degout, Culpabilite, Fierte, Designee, Montree, Suggeree)
tandis qu'emotyc_predict.py attend les noms avec accents (Colère, Dégoût,
Culpabilité, Fierté, Désignée, Montrée, Suggérée).

Ce script :
  1. Renomme les colonnes pour correspondre aux conventions d'emotyc_predict.py
  2. Vérifie que toutes les colonnes requises sont présentes après renommage
  3. Exporte un nouveau XLSX prêt pour l'inférence

Usage :
    python scripts/normalize_for_emotyc.py \
        --input outputs/racisme/claude_racisme_aggregated.xlsx \
        --output outputs/racisme/claude_racisme_normalized.xlsx
"""

import argparse
import os
import sys

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  MAPPING DE RENOMMAGE
# ═══════════════════════════════════════════════════════════════════════════

# Colonnes sans accents → avec accents (convention attendue par emotyc_predict.py)
RENAME_MAP = {
    # Émotions
    "Colere":      "Colère",
    "Degout":      "Dégoût",
    "Culpabilite": "Culpabilité",
    "Fierte":      "Fierté",
    # Modes d'expression
    "Designee":    "Désignée",
    "Montree":     "Montrée",
    "Suggeree":    "Suggérée",
}

# Colonnes d'émotions attendues par emotyc_predict.py
REQUIRED_EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
]

# Colonnes optionnelles mais utiles pour l'évaluation complète
OPTIONAL_COLS = {
    "modes": ["Comportementale", "Désignée", "Montrée", "Suggérée"],
    "emo":   ["Emo"],
    "type":  ["Base", "Complexe"],
}


def normalize(input_path, output_path):
    df = pd.read_excel(input_path)
    print(f"✓ Chargé : {len(df)} lignes depuis {os.path.basename(input_path)}")
    print(f"  Colonnes d'origine : {list(df.columns)}")

    # ── Renommage ─────────────────────────────────────────────────────
    applied = {old: new for old, new in RENAME_MAP.items() if old in df.columns}
    skipped = {old: new for old, new in RENAME_MAP.items() if old not in df.columns}

    if applied:
        df = df.rename(columns=applied)
        print(f"\n▸ Colonnes renommées ({len(applied)}) :")
        for old, new in applied.items():
            print(f"    {old} → {new}")
    else:
        print("\n▸ Aucune colonne à renommer (déjà aux bons noms ?)")

    if skipped:
        print(f"  (colonnes absentes, non renommées : {list(skipped.keys())})")

    # ── Vérification colonne texte ────────────────────────────────────
    text_col = None
    for candidate in ("TEXT", "text", "sentence"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        print("\n✗ Colonne texte non trouvée (TEXT/text/sentence). Abandon.")
        sys.exit(1)
    print(f"\n▸ Colonne texte : '{text_col}' ✓")

    # ── Vérification colonnes émotions ────────────────────────────────
    missing_emo = [e for e in REQUIRED_EMOTIONS if e not in df.columns]
    if missing_emo:
        print(f"\n✗ Colonnes d'émotions manquantes après renommage : {missing_emo}")
        print("  Vérifiez que le fichier d'entrée contient bien ces colonnes.")
        sys.exit(1)
    print(f"▸ 11 colonnes d'émotions : ✓")

    # ── Vérification colonnes optionnelles ────────────────────────────
    for group, cols in OPTIONAL_COLS.items():
        present = [c for c in cols if c in df.columns]
        absent = [c for c in cols if c not in df.columns]
        if present:
            print(f"▸ Colonnes {group} : {present} ✓")
        if absent:
            print(f"  (colonnes {group} absentes : {absent})")

    # ── Export ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\n✓ Fichier normalisé exporté : {output_path}")
    print(f"  {len(df)} lignes, {len(df.columns)} colonnes")


def parse_args():
    p = argparse.ArgumentParser(
        description="Normalise un XLSX agrégé pour emotyc_predict.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True,
                    help="Chemin vers le fichier XLSX agrégé à normaliser")
    p.add_argument("--output", default=None,
                    help="Chemin de sortie (défaut: <input>_normalized.xlsx)")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_normalized{ext}"

    if not os.path.isfile(input_path):
        print(f"✗ Fichier introuvable : {input_path}")
        sys.exit(1)

    normalize(input_path, output_path)


if __name__ == "__main__":
    main()
