#!/usr/bin/env python3
"""
Agrégation SitEmo → vecteur 19 labels binaires (compatible EMOTYC).

Prend en entrée un JSONL d'annotations SitEmo (produit par annotate.py)
et produit un fichier XLSX directement utilisable par emotyc_predict.py
(noms de colonnes avec accents).

L'agrégation applique un OU logique sur toutes les SitEmo de chaque phrase :

  Emo = 1                si ≥1 SitEmo
  Comportementale = 1    si ∃ SitEmo avec mode="Comportementale"
  Désignée = 1           si ∃ SitEmo avec mode="Désignée"
  Montrée = 1            si ∃ SitEmo avec mode="Montrée"
  Suggérée = 1           si ∃ SitEmo avec mode="Suggérée"
  Base = 1               si ∃ SitEmo avec catégorie ∈ {émotions de base}
  Complexe = 1           si ∃ SitEmo avec catégorie ∈ {émotions complexes}
  <Émotion> = 1          si ∃ SitEmo avec categorie ou categorie2 = <Émotion>

Usage :
    python scripts/aggregate.py \\
        --input outputs/homophobie/run001.jsonl \\
        --output outputs/homophobie/run001_aggregated.xlsx
"""

import argparse
import json
import os
import sys
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES — correspondance exacte avec id2label d'EMOTYC
# ═══════════════════════════════════════════════════════════════════════════

# Ordre des 19 labels dans id2label d'EMOTYC (noms avec accents)
VECTOR_ORDER = [
    "Emo", "Comportementale", "Désignée", "Montrée", "Suggérée",
    "Base", "Complexe",
    "Admiration", "Autre", "Colère", "Culpabilité", "Dégoût",
    "Embarras", "Fierté", "Jalousie", "Joie", "Peur", "Surprise", "Tristesse",
]

# Émotions de base
EMOTIONS_BASE = {"Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse"}

# Émotions complexes
EMOTIONS_COMPLEXES = {"Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie"}

# Mapping catégorie → nom dans le vecteur (identité, avec accents)
CAT_TO_VECTOR = {
    "Admiration":   "Admiration",
    "Autre":        "Autre",
    "Colère":       "Colère",
    "Culpabilité":  "Culpabilité",
    "Dégoût":       "Dégoût",
    "Embarras":     "Embarras",
    "Fierté":       "Fierté",
    "Jalousie":     "Jalousie",
    "Joie":         "Joie",
    "Peur":         "Peur",
    "Surprise":     "Surprise",
    "Tristesse":    "Tristesse",
}

# Mapping mode → nom dans le vecteur (avec accents)
MODE_TO_VECTOR = {
    "Désignée":         "Désignée",
    "Comportementale":  "Comportementale",
    "Suggérée":         "Suggérée",
    "Montrée":          "Montrée",
}


# ═══════════════════════════════════════════════════════════════════════════
#  SPAN MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def strip_accents(s: str) -> str:
    """Supprime les accents d'une chaîne."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def find_span_positions(span_text: str, text: str) -> List[Tuple[int, int]]:
    """
    Retrouve les positions (start, end) du span dans le texte.
    Stratégie : recherche exacte → lowercase → strip accents + lowercase.
    Retourne une liste de (start, end) ; vide si non trouvé.
    """
    positions = []

    # 1. Recherche exacte
    start = 0
    while True:
        idx = text.find(span_text, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(span_text)))
        start = idx + 1
    if positions:
        return positions

    # 2. Recherche insensible à la casse
    text_low = text.lower()
    span_low = span_text.lower()
    start = 0
    while True:
        idx = text_low.find(span_low, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(span_text)))
        start = idx + 1
    if positions:
        return positions

    # 3. Recherche avec normalisation des accents
    text_norm = strip_accents(text.lower())
    span_norm = strip_accents(span_text.lower())
    start = 0
    while True:
        idx = text_norm.find(span_norm, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(span_text)))
        start = idx + 1
    if positions:
        return positions

    # 4. Fallback : difflib SequenceMatcher (recherche floue)
    sm = SequenceMatcher(None, text_low, span_low)
    match = sm.find_longest_match(0, len(text_low), 0, len(span_low))
    # On accepte si ≥ 70% du span est retrouvé
    if match.size >= len(span_low) * 0.7:
        positions.append((match.a, match.a + match.size))

    return positions


# ═══════════════════════════════════════════════════════════════════════════
#  AGRÉGATION
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_sitemo_to_vector(
    sitemo_units: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Agrège une liste d'unités SitEmo en un vecteur binaire de 19 composantes.
    Applique un OU logique sur les catégories et modes.
    """
    vec = {label: 0 for label in VECTOR_ORDER}

    if not sitemo_units:
        return vec

    # Emo = 1 dès qu'il y a ≥ 1 unité
    vec["Emo"] = 1

    for unit in sitemo_units:
        # Mode
        mode = unit.get("mode", "")
        mode_vec = MODE_TO_VECTOR.get(mode)
        if mode_vec:
            vec[mode_vec] = 1

        # Catégories (principale + secondaire)
        for cat_field in ("categorie", "categorie2"):
            cat = unit.get(cat_field)
            if not cat:
                continue

            cat_vec = CAT_TO_VECTOR.get(cat)
            if cat_vec:
                vec[cat_vec] = 1

            # Base / Complexe
            if cat in EMOTIONS_BASE:
                vec["Base"] = 1
            elif cat in EMOTIONS_COMPLEXES:
                vec["Complexe"] = 1

    return vec


# ═══════════════════════════════════════════════════════════════════════════
#  CHARGEMENT JSONL
# ═══════════════════════════════════════════════════════════════════════════

def load_annotation_jsonl(path: str) -> List[Dict[str, Any]]:
    """Charge un JSONL d'annotations et extrait les records avec parsed_json."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"  ⚠ Ligne {line_num} : JSON invalide — {e}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Agrégation SitEmo → vecteur 19 labels (compatible EMOTYC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True,
                   help="Chemin vers le JSONL d'annotations SitEmo")
    p.add_argument("--output", required=True,
                   help="Chemin de sortie XLSX")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Chargement ─────────────────────────────────────────────────
    input_path = os.path.abspath(args.input)
    records = load_annotation_jsonl(input_path)
    print(f"✓ {len(records)} lignes chargées depuis {os.path.basename(input_path)}")

    if not records:
        print("⚠ Aucun record à traiter.")
        sys.exit(1)

    # ── 2. Agrégation ─────────────────────────────────────────────────
    rows = []
    n_ok = 0
    n_no_json = 0
    n_old_format = 0

    for rec in records:
        idx = rec.get("idx", "?")
        row_id = rec.get("row_id", idx)
        json_ok = rec.get("json_ok", False)
        pj = rec.get("parsed_json")

        # Colonnes textuelles de base
        row = {"idx": idx, "row_id": row_id}

        # Extraire le texte depuis le prompt (chercher le TARGET)
        prompt = rec.get("prompt", "")
        text = ""
        if "meta" in rec and isinstance(rec["meta"], dict):
            row["thematique"] = rec["meta"].get("thematique", "")
            row["model"] = rec["meta"].get("model", "")

        # Extraire le texte TARGET du prompt
        for line in prompt.split("\n"):
            if line.startswith('TARGET:'):
                # Format: TARGET: [Name] (role=...) "text"
                quote_start = line.find('"')
                quote_end = line.rfind('"')
                if quote_start != -1 and quote_end > quote_start:
                    text = line[quote_start + 1:quote_end]
                break

        row["TEXT"] = text
        row["ID"] = row_id

        # Obtenir les SitEmo
        sitemo_units = []
        if json_ok and isinstance(pj, dict):
            if "sitemo_units" in pj:
                sitemo_units = pj.get("sitemo_units", [])
                n_ok += 1
            elif "emotions" in pj:
                # Ancien format — on ne peut pas agréger en modes, skip
                n_old_format += 1
                continue
            else:
                n_no_json += 1
                continue
        else:
            n_no_json += 1
            continue

        # Agréger en vecteur 19 labels
        vec = aggregate_sitemo_to_vector(sitemo_units)
        row.update(vec)

        # Span matching pour traçabilité
        span_positions = []
        for unit in sitemo_units:
            span_text = unit.get("span_text", "")
            positions = find_span_positions(span_text, text) if text else []
            span_positions.append({
                "span_text": span_text,
                "mode": unit.get("mode"),
                "categorie": unit.get("categorie"),
                "categorie2": unit.get("categorie2"),
                "positions": positions,
            })
        row["_span_details"] = json.dumps(span_positions, ensure_ascii=False)

        rows.append(row)

    # ── 3. Export XLSX ────────────────────────────────────────────────
    if not rows:
        print("⚠ Aucune ligne convertible en vecteur 19 labels.")
        sys.exit(1)

    df_out = pd.DataFrame(rows)

    # Ordonner les colonnes : textuelles d'abord, puis les 19 labels, puis détails
    text_cols = [c for c in ["idx", "row_id", "ID", "TEXT", "thematique", "model"]
                 if c in df_out.columns]
    label_cols = [c for c in VECTOR_ORDER if c in df_out.columns]
    extra_cols = [c for c in df_out.columns if c not in text_cols + label_cols]
    df_out = df_out[text_cols + label_cols + extra_cols]

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_excel(output_path, index=False, engine="openpyxl")

    # ── 4. Validation compatibilité emotyc_predict.py ─────────────────
    required_emotions = [
        "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
        "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
    ]
    missing_emo = [e for e in required_emotions if e not in df_out.columns]
    if missing_emo:
        print(f"\n⚠ Colonnes d'émotions manquantes pour emotyc_predict.py : {missing_emo}")
    else:
        print(f"\n▸ Fichier directement compatible avec emotyc_predict.py ✓")

    # ── 5. Résumé ─────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  AGRÉGATION TERMINÉE")
    print(f"{'═' * 60}")
    print(f"  Records traités   : {len(records)}")
    print(f"  Lignes agrégées   : {len(rows)}")
    print(f"  JSON invalides    : {n_no_json}")
    print(f"  Ancien format     : {n_old_format} (ignorés)")
    print(f"  Sortie XLSX       : {output_path}")

    # Distribution rapide
    if rows:
        print(f"\n  Distribution des labels :")
        for label in VECTOR_ORDER:
            if label in df_out.columns:
                n_pos = int(df_out[label].sum())
                pct = n_pos / len(df_out) * 100
                print(f"    {label:<20s} : {n_pos:>4d} ({pct:>5.1f}%)")

    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
