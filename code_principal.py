#@title Code principal - démarrage de l'annotation

import os, sys, json, time, logging
import pandas as pd

logging.basicConfig(level=logging.WARNING)

# ═══ 1. REPO & IMPORTS ═════════════════════════════════════════════════════
chemin_repo = os.path.abspath("")
sys.path.insert(0, os.path.join(chemin_repo, "src"))

from cyberagg_llm_annot.bedrock_claude import (
    make_bedrock_client, invoke_claude, extract_text, check_stop_reason,
)
from cyberagg_llm_annot.context import get_message_window, minimal_msg_repr
from cyberagg_llm_annot.parsing import extract_row_labels
from cyberagg_llm_annot.prompt_utils import (
    SYSTEM_PROMPT, DEFAULT_LABEL_COLS,
    build_annotations_block, build_user_message,
)
from cyberagg_llm_annot.runner import (
    load_progress, save_progress, try_parse_json,
    validate_annotation, persist_iteration,
)
from cyberagg_llm_annot.io_utils import ensure_dir

# ═══ 2. CHARGEMENT DES DONNÉES ═════════════════════════════════════════════
XLSX_PATH = "/content/a/data/homophobie_scenario_julie.xlsx"
df = pd.read_excel('/content/a/data/homophobie_scenario_julie.xlsx')
print(f"✓ {len(df)} lignes chargées depuis {os.path.basename(XLSX_PATH)}")

# ═══ 3. CONFIGURATION DU RUN ═══════════════════════════════════════════════
THEMATIQUE       = "homophobie"
RUN_ID           = "homophobie_scenario_julie_run001"
OUT_DIR          = os.path.join(chemin_repo, "outputs", THEMATIQUE)
PROGRESS_PATH    = os.path.join(OUT_DIR, f"{RUN_ID}__progress.json")
INTER_CALL_DELAY = 1.0   # secondes entre appels (anti-throttle)
MAX_TOKENS       = 512
ensure_dir(OUT_DIR)

# ═══ 4. CLIENT BEDROCK ═════════════════════════════════════════════════════
client = make_bedrock_client(region_name="eu-north-1")

# ═══ 5. REPRISE ════════════════════════════════════════════════════════════
progress  = load_progress(PROGRESS_PATH)
start_idx = progress["last_completed_idx"] + 1
total     = len(df)
print(f"▸ Reprise à idx={start_idx} / {total}")

# ═══ 6. BOUCLE PRINCIPALE ══════════════════════════════════════════════════
errors = 0
t0 = time.time()

for idx in range(start_idx, total):
    # ── Fenêtre prev / target / next ──
    w = get_message_window(df, idx)

    prev_repr   = minimal_msg_repr(w["prev"])
    target_repr = minimal_msg_repr(w["target"])
    next_repr   = minimal_msg_repr(w["next"])

    row_dict = w["target"]

    # ── Annotations existantes (optionnel — mettre None pour run "sans") ──
    parsed_labels    = extract_row_labels(row_dict, DEFAULT_LABEL_COLS)
    annotations_block = build_annotations_block(parsed_labels)

    # ── Construction du message user ──
    user_message = build_user_message(
        thematique=THEMATIQUE,
        prev_repr=prev_repr,
        target_repr=target_repr,
        next_repr=next_repr,
        annotations_block=None,
    )

    # ── Appel LLM (retry intégré) ──
    bedrock_result = invoke_claude(
        client=client,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )

    raw_text = extract_text(bedrock_result)
    is_complete, stop_reason = check_stop_reason(bedrock_result)

    # ── Parse + validation ──
    json_ok, parsed_obj, json_error = try_parse_json(raw_text)

    validation_warnings = []
    if json_ok and parsed_obj:
        validation_warnings = validate_annotation(parsed_obj)
    if not is_complete:
        validation_warnings.append(f"stop_reason={stop_reason} (troncature probable)")

    # ── Persistance immédiate ──
    row_id = row_dict.get("ID", idx)
    full_prompt_log = f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_message}"

    persist_iteration(
        out_dir=OUT_DIR,
        run_id=RUN_ID,
        idx=idx,
        row_id=row_id,
        prompt=full_prompt_log,
        raw_text=raw_text,
        bedrock_result=bedrock_result,
        parsed_json=parsed_obj,
        json_ok=json_ok,
        json_error=json_error,
        validation_warnings=validation_warnings,
        extra_meta={
            "thematique":   THEMATIQUE,
            "stop_reason":  stop_reason,
            "target_role":  row_dict.get("ROLE"),
            "target_name":  row_dict.get("NAME"),
        },
    )

    # ── Progression (après sauvegarde !) ──
    save_progress(PROGRESS_PATH, last_completed_idx=idx)
    if not json_ok:
        errors += 1

    # ── Monitoring ──
    done   = idx - start_idx + 1
    remain = total - idx - 1
    elapsed = time.time() - t0
    avg    = elapsed / done
    eta    = avg * remain

    status = "✓" if (json_ok and not validation_warnings) else "⚠" if json_ok else "✗"
    print(
        f"[{status}] {idx}/{total-1}  ID={row_id}  "
        f"json={json_ok}  stop={stop_reason}  "
        f"warn={len(validation_warnings)}  err={errors}  "
        f"ETA={eta/60:.1f}min"
    )

    time.sleep(INTER_CALL_DELAY)

# ═══ 7. RÉSUMÉ ═════════════════════════════════════════════════════════════
elapsed_total = time.time() - t0
processed = total - start_idx
print(f"\n{'='*60}")
print(f"Terminé. {processed} items traités en {elapsed_total/60:.1f} min")
print(f"Erreurs JSON : {errors}/{processed}")
print(f"Sorties dans : {OUT_DIR}")
