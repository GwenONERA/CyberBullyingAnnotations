#!/usr/bin/env python3
"""
Script 3/3 — Rapport statistique de l'évaluation EMOTYC.

Lit les JSONL produits par emotyc_predict.py et emotyc_llm_judge.py
et génère un rapport synthétique : métriques, distribution des erreurs,
cohérence du juge LLM, analyses par émotion.

Usage :
    python scripts/emotyc_report.py \
        --eval_dir outputs/homophobie/emotyc_eval

    # Export CSV + figures
    python scripts/emotyc_report.py \
        --eval_dir outputs/homophobie/emotyc_eval \
        --export
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

EMOTION_ORDER = [
    "Admiration", "Colère", "Culpabilité", "Dégoût", "Embarras",
    "Fierté", "Jalousie", "Joie", "Peur", "Surprise", "Tristesse",
]

MODE_ORDER = ["Comportementale", "Designee", "Montree", "Suggeree"]

TYPE_ORDER = ["Base", "Complexe"]

ERROR_TAXONOMY = [
    "lexical_argot", "ironie_polarity", "pragmatic_confusion",
    "seuil_limite", "erreur_humain", "contexte_manquant", "autre",
]

# ── Distribution du corpus Étienne (fine-tuning) ─────────────────────────
# Valeurs tirées de la thèse d'Étienne, p.118 et suivantes.
# Pourcentages calculés sur le corpus d'entraînement d'EMOTYC.
CORPUS_ETIENNE = {
    # Prévalence au niveau phrase
    "Emo":              19.5,  # % des phrases contenant ≥ 1 SitEmo
    # Distribution des modes (% des SitEmo)
    "Suggérée":         33.1,
    "Désignée":         24.7,
    "Comportementale":  21.2,
    "Montrée":          21.0,
    # Distribution des catégories (% des SitEmo)
    "Colère":           25.2,
    "Peur":             21.8,
    "Joie":             16.1,
    "Tristesse":        14.3,
    "Dégoût":           8.7,
    "Surprise":         5.1,
    "Admiration":       3.0,
    "Fierté":           2.4,
    "Culpabilité":      1.7,
    "Embarras":         1.0,
    "Jalousie":         0.7,
    # Type
    "Base":             90.2,
    "Complexe":         9.8,
}


# ═══════════════════════════════════════════════════════════════════════════
#  CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    """Charge un fichier JSONL et retourne une liste de dicts."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 : MÉTRIQUES CLASSIQUES (depuis predictions)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_section_metrics(predictions, label_names, label_key_prefix=""):
    """
    Reconstruit les matrices gold/pred et calcule les métriques
    pour un ensemble de labels donné.

    Arguments :
        predictions      — liste de records JSONL
        label_names      — liste de noms de labels
        label_key_prefix — préfixe des clés dans le record ("" pour émotions,
                           "_mode" pour modes, etc.)
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        cohen_kappa_score,
    )

    N = len(predictions)
    gold_key = f"golds{label_key_prefix}" if label_key_prefix else "golds"
    pred_key = f"preds{label_key_prefix}" if label_key_prefix else "preds"

    gold_mat = np.zeros((N, len(label_names)), dtype=int)
    pred_mat = np.zeros((N, len(label_names)), dtype=int)

    for i, rec in enumerate(predictions):
        golds = rec.get(gold_key, {})
        preds = rec.get(pred_key, {})
        for j, label in enumerate(label_names):
            gold_mat[i, j] = golds.get(label, 0)
            pred_mat[i, j] = preds.get(label, 0)

    results = []
    f1s = []
    for j, label in enumerate(label_names):
        g, p = gold_mat[:, j], pred_mat[:, j]
        acc = accuracy_score(g, p)
        try:
            kappa = cohen_kappa_score(g, p, labels=[0, 1])
        except Exception:
            kappa = float("nan")
        f1 = f1_score(g, p, zero_division=0)
        prec = precision_score(g, p, zero_division=0)
        rec_val = recall_score(g, p, zero_division=0)
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        prev = g.sum() / N * 100
        f1s.append(f1)

        results.append({
            "label": label, "accuracy": acc, "kappa": kappa,
            "f1": f1, "precision": prec, "recall": rec_val,
            "fp": fp, "fn": fn, "prevalence": prev,
        })

    macro_f1 = np.mean(f1s) if f1s else 0
    micro_f1 = f1_score(gold_mat.ravel(), pred_mat.ravel(), zero_division=0)
    exact_match = np.all(gold_mat == pred_mat, axis=1).mean()

    return results, {
        "macro_f1": macro_f1, "micro_f1": micro_f1,
        "exact_match": exact_match, "n_samples": N,
    }, gold_mat, pred_mat


def _print_metrics_section(title, results, global_metrics, N):
    """Affiche une section de métriques formatée."""
    print(f"\n{'═' * 75}")
    print(f"  {title}  ({N} phrases)")
    print(f"{'═' * 75}")
    print(f"\n  {'Label':<20s} {'Acc':>7s} {'κ':>7s} {'F1':>7s} "
          f"{'Prec':>7s} {'Rec':>7s} {'FP':>5s} {'FN':>5s} {'Prev%':>6s}")
    print(f"  {'-' * 72}")

    for r in results:
        k_str = f"{r['kappa']:.3f}" if not np.isnan(r['kappa']) else "  N/A"
        print(f"  {r['label']:<20s} {r['accuracy']:>7.3f} {k_str:>7s} "
              f"{r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['fp']:>5d} {r['fn']:>5d} {r['prevalence']:>5.1f}%")

    print(f"  {'-' * 72}")
    print(f"  Macro-F1      : {global_metrics['macro_f1']:.4f}")
    print(f"  Micro-F1      : {global_metrics['micro_f1']:.4f}")
    print(f"  Exact Match   : {global_metrics['exact_match']:.4f} "
          f"({int(global_metrics['exact_match'] * N)}/{N})")


def report_predictions(predictions):
    """Affiche les métriques issues des prédictions EMOTYC vs gold."""
    if not predictions:
        print("  ⚠ Aucune prédiction chargée.")
        return

    N = len(predictions)

    # ── Section 1a : Métriques par émotion ──
    emo_results, emo_global, gold_mat, pred_mat = _compute_section_metrics(
        predictions, EMOTION_ORDER
    )
    _print_metrics_section("SECTION 1a — MÉTRIQUES EMOTYC vs GOLD (émotions)", emo_results, emo_global, N)

    n_div = sum(1 for r in predictions if r.get("n_divergences", 0) > 0)
    print(f"  Lignes diverg.: {n_div}/{N} ({n_div/N*100:.1f}%)")

    stats = {
        "n_samples": N,
        "n_divergent": n_div,
        "macro_f1": emo_global["macro_f1"],
        "micro_f1": emo_global["micro_f1"],
        "exact_match": emo_global["exact_match"],
    }

    # ── Section 1b : Métriques par mode (si disponibles) ──
    has_mode_golds = any("golds_mode" in r for r in predictions)
    has_mode_preds = any("preds_mode" in r for r in predictions)

    if has_mode_preds and has_mode_golds:
        mode_results, mode_global, _, _ = _compute_section_metrics(
            predictions, MODE_ORDER, label_key_prefix="_mode"
        )
        _print_metrics_section("SECTION 1b — MÉTRIQUES PAR MODE D'EXPRESSION", mode_results, mode_global, N)

    # ── Section 1c : Métrique Emo (caractère émotionnel, si gold dispo) ──
    has_emo_gold = any("gold_emo" in r for r in predictions)
    has_emo_pred = any("pred_emo" in r for r in predictions)

    if has_emo_pred and has_emo_gold:
        from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
        golds_emo = np.array([r.get("gold_emo", 0) for r in predictions])
        preds_emo = np.array([r.get("pred_emo", 0) for r in predictions])
        acc = accuracy_score(golds_emo, preds_emo)
        try:
            kappa = cohen_kappa_score(golds_emo, preds_emo, labels=[0, 1])
        except Exception:
            kappa = float("nan")
        f1 = f1_score(golds_emo, preds_emo, zero_division=0)
        prev = golds_emo.sum() / N * 100

        print(f"\n{'═' * 75}")
        print(f"  SECTION 1c — CARACTÈRE ÉMOTIONNEL (Emo)  ({N} phrases)")
        print(f"{'═' * 75}")
        k_str = f"{kappa:.3f}" if not np.isnan(kappa) else "  N/A"
        print(f"  Acc={acc:.3f}  κ={k_str}  F1={f1:.3f}  Prévalence={prev:.1f}%")

    # ── Section 1d : Métriques Type (Base/Complexe, si gold dispo) ──
    has_type_golds = any("golds_type" in r for r in predictions)
    has_type_preds = any("preds_type" in r for r in predictions)

    if has_type_preds and has_type_golds:
        type_results, type_global, _, _ = _compute_section_metrics(
            predictions, TYPE_ORDER, label_key_prefix="_type"
        )
        _print_metrics_section("SECTION 1d — MÉTRIQUES TYPE (Base/Complexe)", type_results, type_global, N)

    # ── Section : Distribution comparée corpus Étienne ──
    _report_distribution_comparison(predictions, gold_mat)

    return stats


def _report_distribution_comparison(predictions, gold_mat):
    """Affiche un tableau comparatif des distributions avec le corpus Étienne."""
    N = len(predictions)

    print(f"\n{'═' * 75}")
    print(f"  DISTRIBUTION COMPARÉE — Corpus Étienne (fine-tuning) vs Corpus actuel (gold)")
    print(f"{'═' * 75}")
    print(f"\n  {'':30s} {'Corpus Étienne':>18s}  {'Corpus actuel':>16s}")
    print(f"  {'-' * 70}")

    # Emo : % des phrases avec ≥ 1 émotion
    n_emo = int(gold_mat.sum(axis=1).clip(max=1).sum())
    pct_emo = n_emo / N * 100 if N > 0 else 0
    etienne_emo = CORPUS_ETIENNE.get("Emo", 0)
    print(f"  {'Emo (≥1 émotion)':<30s} {etienne_emo:>17.1f}%  {pct_emo:>15.1f}%")

    # Modes — calculer la prévalence sur les phrases gold
    mode_mapping = {
        "Comportementale": "Comportementale",
        "Désignée":        "Designee",
        "Montrée":         "Montree",
        "Suggérée":        "Suggeree",
    }

    for display_name, col_name in [
        ("Suggérée", "Suggeree"), ("Désignée", "Designee"),
        ("Comportementale", "Comportementale"), ("Montrée", "Montree")
    ]:
        etienne_val = CORPUS_ETIENNE.get(display_name, 0)
        # Essayer de calculer la prévalence du mode dans le gold
        # Chercher dans les records la clé golds_mode
        n_mode = 0
        has_data = False
        for rec in predictions:
            golds_mode = rec.get("golds_mode", {})
            if golds_mode and col_name in golds_mode:
                has_data = True
                if golds_mode[col_name] == 1:
                    n_mode += 1
        if has_data:
            pct_mode = n_mode / N * 100
            print(f"  {display_name:<30s} {etienne_val:>17.1f}%  {pct_mode:>15.1f}%")
        else:
            print(f"  {display_name:<30s} {etienne_val:>17.1f}%  {'N/A':>16s}")

    # Émotions
    for j, emo in enumerate(EMOTION_ORDER):
        etienne_val = CORPUS_ETIENNE.get(emo, 0)
        n_pos = int(gold_mat[:, j].sum())
        pct_pos = n_pos / N * 100 if N > 0 else 0
        print(f"  {emo:<30s} {etienne_val:>17.1f}%  {pct_pos:>15.1f}%")

    # Type
    for t in TYPE_ORDER:
        etienne_val = CORPUS_ETIENNE.get(t, 0)
        n_type = 0
        has_data = False
        for rec in predictions:
            golds_type = rec.get("golds_type", {})
            if golds_type and t in golds_type:
                has_data = True
                if golds_type[t] == 1:
                    n_type += 1
        if has_data:
            pct_type = n_type / N * 100
            print(f"  {t:<30s} {etienne_val:>17.1f}%  {pct_type:>15.1f}%")
        else:
            print(f"  {t:<30s} {etienne_val:>17.1f}%  {'N/A':>16s}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 : DOUBLE-BLIND (cohérence du juge LLM)
# ═══════════════════════════════════════════════════════════════════════════

def report_blind(blind_records):
    """Analyse les résultats de la passe double-blind."""
    if not blind_records:
        print("\n  ⚠ Aucun résultat de passe double-blind.")
        return

    valid = [r for r in blind_records if r.get("json_ok") and r.get("parsed_json")]

    print(f"\n{'═' * 75}")
    print(f"  SECTION 2 — PASSE DOUBLE-BLIND  ({len(valid)}/{len(blind_records)} réponses valides)")
    print(f"{'═' * 75}")

    if not valid:
        print("  ⚠ Aucune réponse valide à analyser.")
        return

    # Comptage des verdicts
    verdict_a_counts = Counter()
    verdict_b_counts = Counter()
    error_type_counts = Counter()
    argot_count = 0
    llm_agrees_gold = 0  # quand le LLM donne raison au gold

    for rec in valid:
        pj = rec["parsed_json"]
        gold_is_a = rec.get("gold_is_a", True)

        va = pj.get("verdict_A", "?")
        vb = pj.get("verdict_B", "?")
        verdict_a_counts[va] += 1
        verdict_b_counts[vb] += 1

        err = pj.get("type_erreur_constatee", "?")
        error_type_counts[err] += 1

        if pj.get("argot_present"):
            argot_count += 1

        # Le LLM donne-t-il raison au gold ?
        gold_verdict = va if gold_is_a else vb
        if gold_verdict == "Correct":
            llm_agrees_gold += 1

    print(f"\n  Verdicts Annotateur A :")
    for k, v in verdict_a_counts.most_common():
        print(f"    {k:<25s} : {v:>3d} ({v/len(valid)*100:.1f}%)")

    print(f"\n  Verdicts Annotateur B :")
    for k, v in verdict_b_counts.most_common():
        print(f"    {k:<25s} : {v:>3d} ({v/len(valid)*100:.1f}%)")

    print(f"\n  Types d'erreurs constatées :")
    for k, v in error_type_counts.most_common():
        print(f"    {k:<30s} : {v:>3d} ({v/len(valid)*100:.1f}%)")

    print(f"\n  Argot détecté : {argot_count}/{len(valid)} ({argot_count/len(valid)*100:.1f}%)")
    print(f"  LLM donne raison au gold : {llm_agrees_gold}/{len(valid)} "
          f"({llm_agrees_gold/len(valid)*100:.1f}%)")

    return {
        "n_valid": len(valid),
        "n_total": len(blind_records),
        "llm_agrees_gold_pct": llm_agrees_gold / len(valid) * 100 if valid else 0,
        "argot_pct": argot_count / len(valid) * 100 if valid else 0,
        "error_types": dict(error_type_counts),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 : DIAGNOSTIC (analyse fine des erreurs)
# ═══════════════════════════════════════════════════════════════════════════

def report_diagnostic(diag_records):
    """Analyse les résultats de la passe diagnostic."""
    if not diag_records:
        print("\n  ⚠ Aucun résultat de passe diagnostic.")
        return

    valid = [r for r in diag_records if r.get("json_ok") and r.get("parsed_json")]

    print(f"\n{'═' * 75}")
    print(f"  SECTION 3 — PASSE DIAGNOSTIC  ({len(valid)}/{len(diag_records)} réponses valides)")
    print(f"{'═' * 75}")

    if not valid:
        print("  ⚠ Aucune réponse valide à analyser.")
        return

    # Collecter tous les verdicts individuels (par émotion)
    all_verdicts = []
    for rec in valid:
        pj = rec["parsed_json"]
        verdicts = pj.get("verdicts", [])
        for v in verdicts:
            v["_idx"] = rec["idx"]
            all_verdicts.append(v)

    if not all_verdicts:
        print("  ⚠ Aucun verdict individuel trouvé.")
        return

    print(f"\n  Total verdicts par émotion : {len(all_verdicts)}")

    # Distribution des types d'erreurs
    error_counts = Counter()
    for v in all_verdicts:
        error_counts[v.get("type_erreur", "?")] += 1

    print(f"\n  Distribution des types d'erreurs :")
    for k, c in error_counts.most_common():
        print(f"    {k:<30s} : {c:>3d} ({c/len(all_verdicts)*100:.1f}%)")

    # Distribution qui_a_raison
    raison_counts = Counter()
    for v in all_verdicts:
        raison_counts[v.get("qui_a_raison", "?")] += 1

    print(f"\n  Qui a raison :")
    for k, c in raison_counts.most_common():
        print(f"    {k:<20s} : {c:>3d} ({c/len(all_verdicts)*100:.1f}%)")

    # Distribution axe pragmatique
    axe_counts = Counter()
    for v in all_verdicts:
        axe_counts[v.get("axe_pragmatique", "?")] += 1

    print(f"\n  Axes pragmatiques :")
    for k, c in axe_counts.most_common():
        print(f"    {k:<20s} : {c:>3d} ({c/len(all_verdicts)*100:.1f}%)")

    # Par émotion
    emo_errors = defaultdict(list)
    for v in all_verdicts:
        emo = v.get("emotion", "?")
        emo_errors[emo].append(v)

    print(f"\n  Erreurs par émotion :")
    print(f"    {'Émotion':<15s} {'#Div':>5s} {'#Gold✓':>7s} {'#EMOTYC✓':>9s} "
          f"{'#Indéc':>7s} {'Top erreur':<25s}")
    print(f"    {'-' * 70}")

    for emo in EMOTION_ORDER:
        vlist = emo_errors.get(emo, [])
        if not vlist:
            continue
        n = len(vlist)
        n_gold = sum(1 for v in vlist if v.get("qui_a_raison") == "gold")
        n_emotyc = sum(1 for v in vlist if v.get("qui_a_raison") == "emotyc")
        n_indec = sum(1 for v in vlist if v.get("qui_a_raison") == "indecidable")
        err_counter = Counter(v.get("type_erreur", "?") for v in vlist)
        top_err = err_counter.most_common(1)[0][0] if err_counter else "?"
        print(f"    {emo:<15s} {n:>5d} {n_gold:>7d} {n_emotyc:>9d} "
              f"{n_indec:>7d} {top_err:<25s}")

    # FP vs FN breakdown
    fp_count = sum(1 for v in all_verdicts if v.get("type_divergence") == "faux_positif")
    fn_count = sum(1 for v in all_verdicts if v.get("type_divergence") == "faux_negatif")
    print(f"\n  Faux positifs : {fp_count}  |  Faux négatifs : {fn_count}")

    # ── Exemples de phrases par type d'erreur ──
    text_by_idx = {rec["idx"]: rec.get("text", "") for rec in valid}

    examples_by_type = defaultdict(list)
    for v in all_verdicts:
        err_type = v.get("type_erreur", "?")
        examples_by_type[err_type].append(v)

    print(f"\n  {'─' * 70}")
    print(f"  EXEMPLES PAR TYPE D'ERREUR")
    print(f"  {'─' * 70}")

    MAX_EXAMPLES = 3
    for err_type, verdicts_list in sorted(examples_by_type.items(),
                                          key=lambda x: -len(x[1])):
        print(f"\n  ▸ {err_type} ({len(verdicts_list)} occurrences) :")
        shown = 0
        for v in verdicts_list:
            if shown >= MAX_EXAMPLES:
                break
            idx = v.get("_idx", "?")
            text = text_by_idx.get(idx, "(texte non disponible)")
            text_display = text[:80] + ("…" if len(text) > 80 else "")
            emo = v.get("emotion", "?")
            div_type = v.get("type_divergence", "?")
            justif = v.get("justification", "")[:120]
            raison = v.get("qui_a_raison", "?")
            print(f"    [{div_type}] {emo} — « {text_display} »")
            print(f"       → raison: {raison} | {justif}")
            shown += 1

    return {
        "n_valid": len(valid),
        "n_verdicts": len(all_verdicts),
        "error_types": dict(error_counts),
        "qui_a_raison": dict(raison_counts),
        "fp_count": fp_count,
        "fn_count": fn_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_csv(eval_dir, predictions, diag_records):
    """Exporte un CSV de synthèse combinant prédictions et diagnostics."""
    if not predictions:
        return

    # Base : une ligne par prédiction
    rows = []
    for rec in predictions:
        row = {
            "idx": rec["idx"],
            "text": rec["text"][:100],
            "n_divergences": rec["n_divergences"],
        }
        for emo in EMOTION_ORDER:
            row[f"{emo}_gold"] = rec["golds"].get(emo, 0)
            row[f"{emo}_pred"] = rec["preds"].get(emo, 0)
            row[f"{emo}_proba"] = rec["probas"].get(emo, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(eval_dir, "emotyc_eval_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV exporté : {csv_path}")

    # CSV des verdicts diagnostic
    if diag_records:
        valid = [r for r in diag_records if r.get("json_ok") and r.get("parsed_json")]
        verdict_rows = []
        for rec in valid:
            for v in rec["parsed_json"].get("verdicts", []):
                verdict_rows.append({
                    "idx": rec["idx"],
                    "text": rec["text"][:80],
                    "emotion": v.get("emotion"),
                    "gold": v.get("gold"),
                    "pred_binaire": v.get("pred_binaire"),
                    "pred_proba": v.get("pred_proba"),
                    "type_divergence": v.get("type_divergence"),
                    "axe_pragmatique": v.get("axe_pragmatique"),
                    "type_erreur": v.get("type_erreur"),
                    "qui_a_raison": v.get("qui_a_raison"),
                    "justification": v.get("justification", "")[:200],
                })
        if verdict_rows:
            df_v = pd.DataFrame(verdict_rows)
            csv_v_path = os.path.join(eval_dir, "emotyc_verdicts_detail.csv")
            df_v.to_csv(csv_v_path, index=False)
            print(f"✓ CSV verdicts : {csv_v_path}")


def export_figures(eval_dir, predictions):
    """Génère des figures de synthèse."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.0)
    except ImportError:
        print("  ⚠ matplotlib/seaborn non installés — figures ignorées.")
        return

    if not predictions:
        return

    N = len(predictions)
    gold_mat = np.zeros((N, len(EMOTION_ORDER)), dtype=int)
    pred_mat = np.zeros((N, len(EMOTION_ORDER)), dtype=int)
    for i, rec in enumerate(predictions):
        for j, emo in enumerate(EMOTION_ORDER):
            gold_mat[i, j] = rec["golds"].get(emo, 0)
            pred_mat[i, j] = rec["preds"].get(emo, 0)

    from sklearn.metrics import f1_score

    # Figure 1 : F1 par émotion
    f1s = []
    for j, emo in enumerate(EMOTION_ORDER):
        f1s.append(f1_score(gold_mat[:, j], pred_mat[:, j], zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if v >= 0.5 else "#f39c12" if v >= 0.3 else "#e74c3c" for v in f1s]
    bars = ax.barh(
        list(reversed(EMOTION_ORDER)),
        list(reversed(f1s)),
        color=list(reversed(colors)),
        edgecolor="black", linewidth=0.5,
    )
    ax.axvline(0.5, color="green", ls="--", lw=1, alpha=0.7, label="F1=0.5")
    ax.set_xlabel("F1 Score")
    ax.set_xlim(0, 1.05)
    ax.set_title("F1 Score par Émotion — EMOTYC vs Gold")
    ax.legend(loc="lower right")
    for bar, val in zip(bars, reversed(f1s)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "f1_par_emotion.png"), dpi=150)

    # Figure 2 : FP vs FN par émotion
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(EMOTION_ORDER))
    w = 0.35
    fp_vals = [int(((gold_mat[:, j] == 0) & (pred_mat[:, j] == 1)).sum())
               for j in range(len(EMOTION_ORDER))]
    fn_vals = [int(((gold_mat[:, j] == 1) & (pred_mat[:, j] == 0)).sum())
               for j in range(len(EMOTION_ORDER))]
    ax.barh(x + w / 2, fp_vals, w, label="Faux Positifs", color="#e67e22")
    ax.barh(x - w / 2, fn_vals, w, label="Faux Négatifs", color="#3498db")
    ax.set_yticks(x)
    ax.set_yticklabels(EMOTION_ORDER)
    ax.set_xlabel("Nombre")
    ax.set_title("Faux Positifs vs Faux Négatifs par Émotion")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "fp_fn_par_emotion.png"), dpi=150)

    plt.close("all")
    print(f"✓ Figures sauvegardées dans {eval_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Rapport statistique de l'évaluation EMOTYC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--eval_dir", required=True,
                    help="Dossier contenant les JSONL de résultats")
    p.add_argument("--export", action="store_true",
                    help="Exporter les CSV et figures")
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = os.path.abspath(args.eval_dir)

    print(f"\n{'═' * 75}")
    print(f"  RAPPORT D'ÉVALUATION EMOTYC")
    print(f"  Dossier : {eval_dir}")
    print(f"{'═' * 75}")

    # Charger les fichiers
    pred_path = os.path.join(eval_dir, "emotyc_predictions.jsonl")
    blind_path = os.path.join(eval_dir, "emotyc_judge_blind.jsonl")
    diag_path = os.path.join(eval_dir, "emotyc_judge_diagnostic.jsonl")

    predictions = load_jsonl(pred_path)
    blind_records = load_jsonl(blind_path)
    diag_records = load_jsonl(diag_path)

    print(f"\n  Fichiers chargés :")
    print(f"    Prédictions  : {len(predictions)} lignes")
    print(f"    Double-blind : {len(blind_records)} lignes")
    print(f"    Diagnostic   : {len(diag_records)} lignes")

    # Section 1 : Métriques classiques (+ modes, emo, type, distribution Étienne)
    pred_stats = report_predictions(predictions)

    # Section 2 : Double-blind
    blind_stats = report_blind(blind_records)

    # Section 3 : Diagnostic
    diag_stats = report_diagnostic(diag_records)

    # Export
    if args.export:
        export_csv(eval_dir, predictions, diag_records)
        export_figures(eval_dir, predictions)

    print(f"\n{'═' * 75}")
    print(f"  RAPPORT TERMINÉ")
    print(f"{'═' * 75}\n")


if __name__ == "__main__":
    main()
