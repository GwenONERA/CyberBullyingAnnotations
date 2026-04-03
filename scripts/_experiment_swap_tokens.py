"""Quick experiment: check tokenization and test no-swap vs swap."""
import os, sys

# Fix torch CUDA DLL loading on Windows
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ── Q1: Tokenization check ──────────────────────────────────────────
print("=" * 70)
print("  Q1: add_special_tokens=False vs True")
print("=" * 70)

tok = AutoTokenizer.from_pretrained("camembert-base")
text = "before:</s>current:Bonjour</s>after:</s>"

enc_no = tok(text, add_special_tokens=False)
enc_yes = tok(text, add_special_tokens=True)

tokens_no = tok.convert_ids_to_tokens(enc_no["input_ids"])
tokens_yes = tok.convert_ids_to_tokens(enc_yes["input_ids"])

print(f"\nadd_special_tokens=False → {len(tokens_no)} tokens")
print(f"  First 15: {tokens_no[:15]}")
print(f"  Position 0: '{tokens_no[0]}' (ID {enc_no['input_ids'][0]})")

print(f"\nadd_special_tokens=True → {len(tokens_yes)} tokens")
print(f"  First 15: {tokens_yes[:15]}")
print(f"  Position 0: '{tokens_yes[0]}' (ID {enc_yes['input_ids'][0]})")

print(f"\n  BOS token: '{tok.bos_token}' (ID {tok.bos_token_id})")
print(f"  EOS token: '{tok.eos_token}' (ID {tok.eos_token_id})")

# Check: Does CamemBERT use <s> as CLS for classification?
print(f"\n  → Avec False, le modèle classifie sur token[0] = '{tokens_no[0]}'")
print(f"  → Avec True,  le modèle classifie sur token[0] = '{tokens_yes[0]}' = <s> (CLS)")


# ── Q2: Swap vs No-swap ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Q2: Impact du swap Admiration ↔ Autre")
print("=" * 70)

import pandas as pd

MODEL_NAME = "TextToKids/CamemBERT-base-EmoTextToKids"
EMOTYC_LABEL2ID = {
    "Emo": 0, "Comportementale": 1, "Designee": 2, "Montree": 3,
    "Suggeree": 4, "Base": 5, "Complexe": 6, "Admiration": 7,
    "Autre": 8, "Colere": 9, "Culpabilite": 10, "Degout": 11,
    "Embarras": 12, "Fierte": 13, "Jalousie": 14, "Joie": 15,
    "Peur": 16, "Surprise": 17, "Tristesse": 18,
}

GOLD_TO_EMOTYC = {
    "Colère": "Colere", "Dégoût": "Degout", "Joie": "Joie",
    "Peur": "Peur", "Surprise": "Surprise", "Tristesse": "Tristesse",
    "Admiration": "Admiration", "Culpabilité": "Culpabilite",
    "Embarras": "Embarras", "Fierté": "Fierte", "Jalousie": "Jalousie",
}
EMOTION_ORDER = list(GOLD_TO_EMOTYC.keys())
EMOTION_INDICES = {g: EMOTYC_LABEL2ID[e] for g, e in GOLD_TO_EMOTYC.items()}

OPTIMIZED_THRESHOLDS = {
    "Admiration": 0.9531926895718311, "Colère": 0.28217218720548165,
    "Culpabilité": 0.12671495241969652, "Dégoût": 0.19269005632824862,
    "Embarras": 0.9548280448988165, "Fierté": 0.8002327448859459,
    "Jalousie": 0.017136900811277365, "Joie": 0.9155047132251537,
    "Peur": 0.9881862235180032, "Surprise": 0.9722425408373772,
    "Tristesse": 0.6984491339960737,
}

SWAP_PAIRS = [(EMOTYC_LABEL2ID["Admiration"], EMOTYC_LABEL2ID["Autre"])]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
print(f"\n✓ Modèle chargé sur {device}")

# Test on racisme corpus (with context, same as before)
xlsx_path = r"C:\Users\gtsang\Annotation_\outputs\racisme\claude_racisme_aggregated_normalized.xlsx"
df = pd.read_excel(xlsx_path)
sentences = df["TEXT"].astype(str).tolist()
N = len(sentences)

# Format with context
eos = tok.eos_token
formatted = []
for i in range(N):
    prev = sentences[i-1] if i > 0 else eos
    nxt = sentences[i+1] if i < N-1 else eos
    formatted.append(f"before:{prev}{eos}current:{sentences[i]}{eos}after:{nxt}{eos}")

# Inference
@torch.no_grad()
def run_inference(texts, do_swap=True):
    all_probs = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True,
                  max_length=512, add_special_tokens=False).to(device)
        logits = model(**enc).logits
        if do_swap:
            for a, b in SWAP_PAIRS:
                logits[:, [a, b]] = logits[:, [b, a]]
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

probs_swap = run_inference(formatted, do_swap=True)
probs_noswap = run_inference(formatted, do_swap=False)

# Extract gold
gold = np.zeros((N, len(EMOTION_ORDER)), dtype=int)
for j, emo in enumerate(EMOTION_ORDER):
    vals = pd.to_numeric(df[emo], errors="coerce").fillna(0)
    gold[:, j] = (vals >= 0.5).astype(int)

threshold_array = np.array([OPTIMIZED_THRESHOLDS[e] for e in EMOTION_ORDER])

# Compare for Admiration specifically
adm_idx = EMOTION_ORDER.index("Admiration")
adm_model_idx_swap = EMOTION_INDICES["Admiration"]  # index 7 after swap = what was Autre
adm_model_idx_noswap = EMOTION_INDICES["Admiration"]  # index 7 without swap = actual Admiration

print(f"\n--- Admiration (gold positives: {gold[:, adm_idx].sum()}/{N}) ---")
print(f"  Seuil optimisé: {OPTIMIZED_THRESHOLDS['Admiration']:.4f}")

# With swap
probs_adm_swap = probs_swap[:, adm_model_idx_swap]
preds_adm_swap = (probs_adm_swap >= OPTIMIZED_THRESHOLDS["Admiration"]).astype(int)
tp_s = ((gold[:, adm_idx] == 1) & (preds_adm_swap == 1)).sum()
fp_s = ((gold[:, adm_idx] == 0) & (preds_adm_swap == 1)).sum()
fn_s = ((gold[:, adm_idx] == 1) & (preds_adm_swap == 0)).sum()
print(f"\n  AVEC swap (Admiration ↔ Autre):")
print(f"    Proba mean: {probs_adm_swap.mean():.4f}, median: {np.median(probs_adm_swap):.4f}")
print(f"    Preds positives: {preds_adm_swap.sum()}, TP={tp_s}, FP={fp_s}, FN={fn_s}")

# Without swap
probs_adm_noswap = probs_noswap[:, adm_model_idx_noswap]
preds_adm_noswap = (probs_adm_noswap >= OPTIMIZED_THRESHOLDS["Admiration"]).astype(int)
tp_n = ((gold[:, adm_idx] == 1) & (preds_adm_noswap == 1)).sum()
fp_n = ((gold[:, adm_idx] == 0) & (preds_adm_noswap == 1)).sum()
fn_n = ((gold[:, adm_idx] == 1) & (preds_adm_noswap == 0)).sum()
print(f"\n  SANS swap:")
print(f"    Proba mean: {probs_adm_noswap.mean():.4f}, median: {np.median(probs_adm_noswap):.4f}")
print(f"    Preds positives: {preds_adm_noswap.sum()}, TP={tp_n}, FP={fp_n}, FN={fn_n}")

# Also check "Autre" raw index
autre_idx = EMOTYC_LABEL2ID["Autre"]  # 8
print(f"\n--- Logit brut index 7 (Admiration config) vs index 8 (Autre config) ---")
print(f"  Raw index 7 - mean proba: {torch.sigmoid(torch.tensor(probs_noswap[:, 7])).mean():.4f} (already sigmoid)")
print(f"  Raw index 8 - mean proba: {torch.sigmoid(torch.tensor(probs_noswap[:, 8])).mean():.4f} (already sigmoid)")
# probs_noswap is already sigmoid, so just use directly
print(f"\n  Position 7 (Admiration in config): mean={probs_noswap[:, 7].mean():.4f}, >0.5: {(probs_noswap[:, 7] > 0.5).sum()}")
print(f"  Position 8 (Autre in config):      mean={probs_noswap[:, 8].mean():.4f}, >0.5: {(probs_noswap[:, 8] > 0.5).sum()}")

# Full comparison: all 11 emotions, swap vs no-swap
from sklearn.metrics import f1_score

print(f"\n--- Comparaison globale (11 émotions, seuils optimisés) ---")
print(f"  {'Émotion':<15s} {'F1 swap':>8s} {'F1 no-swap':>10s} {'Δ':>8s}")
print(f"  {'-'*45}")

for j, emo in enumerate(EMOTION_ORDER):
    idx = EMOTION_INDICES[emo]
    p_s = probs_swap[:, idx]
    p_n = probs_noswap[:, idx]
    pred_s = (p_s >= OPTIMIZED_THRESHOLDS[emo]).astype(int)
    pred_n = (p_n >= OPTIMIZED_THRESHOLDS[emo]).astype(int)
    f1_s = f1_score(gold[:, j], pred_s, zero_division=0)
    f1_n = f1_score(gold[:, j], pred_n, zero_division=0)
    delta = f1_n - f1_s
    marker = " ◄" if abs(delta) > 0.01 else ""
    print(f"  {emo:<15s} {f1_s:>8.4f} {f1_n:>10.4f} {delta:>+8.4f}{marker}")

# Macro-F1
f1s_swap = [f1_score(gold[:, j], (probs_swap[:, EMOTION_INDICES[e]] >= OPTIMIZED_THRESHOLDS[e]).astype(int), zero_division=0) for j, e in enumerate(EMOTION_ORDER)]
f1s_noswap = [f1_score(gold[:, j], (probs_noswap[:, EMOTION_INDICES[e]] >= OPTIMIZED_THRESHOLDS[e]).astype(int), zero_division=0) for j, e in enumerate(EMOTION_ORDER)]
print(f"  {'-'*45}")
print(f"  {'Macro-F1':<15s} {np.mean(f1s_swap):>8.4f} {np.mean(f1s_noswap):>10.4f} {np.mean(f1s_noswap)-np.mean(f1s_swap):>+8.4f}")
