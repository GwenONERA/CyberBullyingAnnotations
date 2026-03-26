from __future__ import annotations
import glob
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import (
    ensure_dir, append_jsonl, safe_write_json, load_json, utc_now_iso,
)
from .prompt_utils import EMOTIONS, MODES

# ── Regex pour extraire le JSON d'un éventuel bloc ```json … ``` ───────────
_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)

# ── Émotions et modes attendus (utilisés pour la validation) ───────────────
_EXPECTED_EMOTIONS = frozenset(EMOTIONS)
_EXPECTED_MODES = frozenset(MODES)

# ── Ancien format (11 émotions binaires, rétro-compatibilité) ─────────────
_OLD_EMOTIONS = frozenset({
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
})


# ── Progress ───────────────────────────────────────────────────────────────

def load_progress(progress_path: str) -> Dict[str, Any]:
    prog = load_json(progress_path)
    if prog is None:
        return {"last_completed_idx": -1}
    return prog


def save_progress(progress_path: str, last_completed_idx: int) -> None:
    safe_write_json(progress_path, {
        "last_completed_idx": last_completed_idx,
        "updated_at": utc_now_iso(),
    })


# ── JSON parsing robuste ──────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Retire un éventuel bloc markdown ``` autour du JSON."""
    text = text.strip()
    m = _CODEBLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Cas où il n'y a que des ``` en début/fin sans contenu capturé
    if text.startswith("```"):
        lines = text.splitlines()
        # retirer première et dernière ligne si elles sont des ```
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        return "\n".join(lines).strip()
    return text


def try_parse_json(text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Parse la réponse LLM en JSON, en tolérant un wrapper markdown."""
    cleaned = _strip_markdown(text)
    try:
        obj = json.loads(cleaned)
        return True, obj, None
    except Exception as exc:
        return False, None, str(exc)


# ── Détection de format ───────────────────────────────────────────────────

def _is_old_format(obj: Dict[str, Any]) -> bool:
    """Détecte si l'objet JSON est au format ancien (émotions binaires)."""
    return "emotions" in obj and "sitemo_units" not in obj


# ── Validation structurelle ────────────────────────────────────────────────

def validate_annotation(
    obj: Dict[str, Any],
    target_text: Optional[str] = None,
) -> List[str]:
    """
    Vérifie que le JSON parsé respecte le schéma attendu.
    Supporte le nouveau format SitEmo et l'ancien format émotions binaires.
    Retourne une liste de warnings (vide = tout est OK).

    Arguments :
        obj         — le JSON parsé de la réponse LLM
        target_text — le texte TARGET original, pour vérification du span
    """
    warnings: List[str] = []

    if not isinstance(obj, dict):
        return ["root is not a dict"]

    # ── Rétro-compatibilité : ancien format émotions binaires ──
    if _is_old_format(obj):
        emotions = obj.get("emotions", {})
        present = set(emotions.keys())
        missing = _OLD_EMOTIONS - present
        extra = present - _OLD_EMOTIONS - {"Autre"}
        if missing:
            warnings.append(f"missing emotions: {sorted(missing)}")
        if extra:
            warnings.append(f"unexpected emotions: {sorted(extra)}")
        for k, v in emotions.items():
            if v not in (0, 1):
                warnings.append(f"'{k}' has non-binary value: {v}")
        return warnings

    # ── Nouveau format SitEmo ──
    units = obj.get("sitemo_units")
    if units is None:
        warnings.append("missing 'sitemo_units'")
        return warnings

    if not isinstance(units, list):
        warnings.append(f"'sitemo_units' is not a list: {type(units).__name__}")
        return warnings

    for i, unit in enumerate(units):
        prefix = f"sitemo_units[{i}]"

        if not isinstance(unit, dict):
            warnings.append(f"{prefix}: not a dict")
            continue

        # span_text
        span = unit.get("span_text")
        if not span or not isinstance(span, str) or not span.strip():
            warnings.append(f"{prefix}: 'span_text' manquant ou vide")
        elif target_text:
            # Vérification que le span se retrouve dans le texte TARGET
            if span not in target_text and span.lower() not in target_text.lower():
                warnings.append(
                    f"{prefix}: span_text '{span[:50]}' non trouvé dans le TARGET"
                )

        # mode
        mode = unit.get("mode")
        if mode not in _EXPECTED_MODES:
            warnings.append(f"{prefix}: mode invalide '{mode}' "
                            f"(attendu: {sorted(_EXPECTED_MODES)})")

        # categorie
        cat = unit.get("categorie")
        if cat not in _EXPECTED_EMOTIONS:
            warnings.append(f"{prefix}: categorie invalide '{cat}' "
                            f"(attendu: {sorted(_EXPECTED_EMOTIONS)})")

        # categorie2 (optionnel : null ou ∈ EMOTIONS)
        cat2 = unit.get("categorie2")
        if cat2 is not None and cat2 not in _EXPECTED_EMOTIONS:
            warnings.append(f"{prefix}: categorie2 invalide '{cat2}' "
                            f"(attendu: null ou {sorted(_EXPECTED_EMOTIONS)})")

    return warnings


# ── Persistance d'une itération ────────────────────────────────────────────

def persist_iteration(
    out_dir: str,
    run_id: str,
    idx: int,
    row_id: Any,
    prompt: str,
    raw_text: str,
    llm_result: Dict[str, Any],
    parsed_json: Optional[Dict[str, Any]],
    json_ok: bool,
    json_error: Optional[str],
    validation_warnings: Optional[List[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(out_dir)

    record: Dict[str, Any] = {
        "run_id":               run_id,
        "idx":                  idx,
        "row_id":               row_id,
        "timestamp_utc":        utc_now_iso(),
        "json_ok":              json_ok,
        "json_error":           json_error,
        "validation_warnings":  validation_warnings or [],
        "raw_text":             raw_text,
        "parsed_json":          parsed_json,
        "llm_result":           llm_result,
        "prompt":               prompt,
    }
    if extra_meta:
        record["meta"] = extra_meta

    # 1) Log JSONL append-only (artefact principal)
    append_jsonl(os.path.join(out_dir, f"{run_id}.jsonl"), record)

    # 2) Snapshot individuel temporaire (debug / reprise)
    item_dir = os.path.join(out_dir, "items")
    ensure_dir(item_dir)
    safe_write_json(
        os.path.join(item_dir, f"{run_id}__idx{idx:05d}__id{row_id}.json"),
        record,
    )


def cleanup_items_dir(out_dir: str, run_id: str) -> int:
    """
    Supprime le dossier items/ après génération du JSONL final.
    Retourne le nombre de fichiers supprimés.
    """
    item_dir = os.path.join(out_dir, "items")
    if not os.path.isdir(item_dir):
        return 0
    pattern = os.path.join(item_dir, f"{run_id}__*.json")
    files = glob.glob(pattern)
    count = len(files)
    for f in files:
        os.remove(f)
    # Supprimer le dossier s'il est vide
    try:
        os.rmdir(item_dir)
    except OSError:
        pass  # pas vide (d'autres runs y sont)
    return count
