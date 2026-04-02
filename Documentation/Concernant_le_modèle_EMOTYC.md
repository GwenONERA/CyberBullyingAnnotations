Notebook


```python
#!/usr/bin/env python3
"""
H1 vs H2 hypothesis test for TextToKids/CamemBERT-base-EmoTextToKids

H1 : fine-tuning used add_special_tokens=True  → position 0 = <s>  (CLS/BOS)
H2 : fine-tuning used add_special_tokens=False → position 0 = 1er subword du texte

On évalue les deux sur le jeu de données de fine-tuning et on compare.
"""

import json, ast, unicodedata, time
from collections import OrderedDict

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════
BATCH  = 128
MAXLEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════════
#  MODÈLE & TOKENIZER
# ════════════════════════════════════════════════════════════════
print("=" * 72)
print(" CHARGEMENT MODÈLE & TOKENIZER")
print("=" * 72)

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = (
    AutoModelForSequenceClassification
    .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
    .to(DEVICE).eval()
)

NL     = model.config.num_labels                         # 19
LABELS = [model.config.id2label[i] for i in range(NL)]
BOS_ID = tokenizer.bos_token_id                          # 5
EOS_ID = tokenizer.eos_token_id                          # 6
BOS    = tokenizer.bos_token                             # "<s>"
EOS    = tokenizer.eos_token                             # "</s>"

print(f"Device       : {DEVICE}")
print(f"Labels ({NL}) : {LABELS}")
print(f"BOS={BOS} (id={BOS_ID})   EOS={EOS} (id={EOS_ID})")

# Démonstration du comportement du tokenizer
demo = "Il pleure."
for sp in (True, False):
    enc  = tokenizer(demo, add_special_tokens=sp)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][:8])
    print(f"  add_special_tokens={sp!s:5s} → {toks}")

# ════════════════════════════════════════════════════════════════
#  JEU DE DONNÉES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" CHARGEMENT DU JEU DE DONNÉES")
print("=" * 72)

ds = load_dataset("TextToKids/EmoTextToKids-sentences")
if hasattr(ds, "keys"):
    print(f"Splits disponibles : {list(ds.keys())}")
    all_ds = concatenate_datasets([ds[s] for s in ds.keys()])
else:
    all_ds = ds

print(f"Nombre de lignes : {len(all_ds)}")
print(f"Colonnes         : {all_ds.column_names}")
for i in range(min(2, len(all_ds))):
    ex = all_ds[i]
    print(f"\n  Exemple {i} :")
    for c in all_ds.column_names:
        print(f"    {c}: {repr(ex[c])[:120]}")

# ════════════════════════════════════════════════════════════════
#  PARSING DES LABELS → matrice (N, 19)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" PARSING DES LABELS")
print("=" * 72)


def _norm(s: str) -> str:
    """Minuscules + suppression des accents."""
    s = s.lower().strip()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# Index principal (français normalisé → indice)
label2idx = {_norm(l): i for i, l in enumerate(LABELS)}

# Alias anglais → français
_ALIAS = {
    "anger": "colere", "other": "autre", "admiration": "admiration",
    "guilt": "culpabilite", "disgust": "degout",
    "embarrassment": "embarras", "pride": "fierte",
    "jealousy": "jalousie", "joy": "joie", "fear": "peur",
    "surprise": "surprise", "sadness": "tristesse",
    "base": "base", "complex": "complexe",
    "behavioral": "comportementale", "designated": "designee",
    "shown": "montree", "suggested": "suggeree",
}
for en, fr in _ALIAS.items():
    nfr = _norm(fr)
    if nfr in label2idx:
        label2idx[en] = label2idx[nfr]


def parse_list(val) -> list[str]:
    """Parse robuste d'un champ liste (JSON, crochets, etc.)."""
    if val is None:
        return []
    if isinstance(val, list):
        return [_norm(str(v)) for v in val if v]
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan", "[]", "[ ]"):
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            r = parser(s)
            if isinstance(r, list):
                return [_norm(str(v)) for v in r if v]
        except Exception:
            pass
    if s.startswith("[") and s.endswith("]"):
        return [_norm(x.strip().strip('"').strip("'"))
                for x in s[1:-1].split(",") if x.strip()]
    return [_norm(s)]


def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "oui", "vrai")


def build_vector(example) -> np.ndarray:
    vec = np.zeros(NL, dtype=np.float32)
    if parse_bool(example.get("is_emotional", False)):
        vec[0] = 1.0
    for field in ("modes", "types", "categories"):
        for val in parse_list(example.get(field)):
            if val in label2idx:
                vec[label2idx[val]] = 1.0
    return vec


# Construction des tableaux
texts, prevs, nexts, vecs = [], [], [], []
skipped = 0

for i in range(len(all_ds)):
    ex = all_ds[i]
    t = str(ex.get("target_sentence", "")).strip()
    if not t or t.lower() in ("none", "nan", ""):
        skipped += 1
        continue

    p = str(ex.get("previous_sentence", "")).strip()
    n = str(ex.get("next_sentence", "")).strip()
    if p.lower() in ("none", "nan"):
        p = ""
    if n.lower() in ("none", "nan"):
        n = ""

    texts.append(t)
    prevs.append(p)
    nexts.append(n)
    vecs.append(build_vector(ex))

Y = np.array(vecs)
N, L = Y.shape
CELLS = N * L

print(f"Exemples parsés : {N}   (ignorés : {skipped})")
print(f"Matrice         : {N} × {L} = {CELLS} cellules\n")

print("Distribution des labels :")
for j in range(L):
    pos = int(Y[:, j].sum())
    print(f"  {LABELS[j]:<22s}: {pos:5d} / {N}  ({100 * pos / N:5.1f}%)")

zero = [LABELS[j] for j in range(L) if Y[:, j].sum() == 0]
if zero:
    print(f"\n  ⚠ Labels sans exemples positifs : {zero}")

# ════════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@torch.no_grad()
def get_logits(input_texts, add_sp, padding="longest"):
    """Inférence par batch → (N, 19) logits."""
    parts = []
    for s in range(0, len(input_texts), BATCH):
        enc = tokenizer(
            input_texts[s : s + BATCH],
            return_tensors="pt",
            truncation=True,
            max_length=MAXLEN,
            padding=padding,
            add_special_tokens=add_sp,
        ).to(DEVICE)
        parts.append(model(**enc).logits.cpu().numpy())
    return np.concatenate(parts)


def find_opt_thresholds(probs):
    """Seuil optimal par label (minimise les erreurs)."""
    thr = np.empty(L)
    per_err = np.zeros(L, dtype=int)
    for j in range(L):
        p, y = probs[:, j], Y[:, j]
        vals = np.sort(np.unique(p))
        cuts = np.concatenate(
            [[vals[0] - 1e-9], (vals[:-1] + vals[1:]) / 2, [vals[-1] + 1e-9]]
        )
        errs = np.array([int(((p >= c).astype(int) != y).sum()) for c in cuts])
        best = int(errs.argmin())
        thr[j] = cuts[best]
        per_err[j] = errs[best]
    return thr, int(per_err.sum()), per_err


def compute_f1(yt, yp):
    """Micro-F1 et Macro-F1 pour matrices (N, L)."""
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0
    re = tp / (tp + fn) if tp + fn else 0
    micro = 2 * pr * re / (pr + re) if pr + re else 0

    f1s = []
    for j in range(L):
        t = int(((yt[:, j] == 1) & (yp[:, j] == 1)).sum())
        f = int(((yt[:, j] == 0) & (yp[:, j] == 1)).sum())
        n = int(((yt[:, j] == 1) & (yp[:, j] == 0)).sum())
        pj = t / (t + f) if t + f else 0
        rj = t / (t + n) if t + n else 0
        f1s.append(2 * pj * rj / (pj + rj) if pj + rj else 0)
    return micro, float(np.mean(f1s))


def evaluate(lg):
    """Évaluation complète : seuil 0.5 + seuils optimaux."""
    P = sigmoid(lg)

    # Seuil 0.5
    p5 = (P >= 0.5).astype(int)
    e5 = int((p5 != Y).sum())
    mif5, maf5 = compute_f1(Y, p5)

    # Seuils optimaux
    thr, eo, peo = find_opt_thresholds(P)
    po = (P >= thr).astype(int)
    mifo, mafo = compute_f1(Y, po)

    pe5 = np.array(
        [int(((P[:, j] >= 0.5).astype(int) != Y[:, j]).sum()) for j in range(L)]
    )

    # Séparation des logits (positifs vs négatifs)
    seps = np.zeros(L)
    for j in range(L):
        mask = Y[:, j] == 1
        if mask.sum() and (~mask).sum():
            seps[j] = lg[mask, j].mean() - lg[~mask, j].mean()

    return dict(
        e5=e5,
        a5=float((p5 == Y).mean()),
        x5=float((p5 == Y).all(axis=1).mean()),
        mif5=mif5,
        maf5=maf5,
        eo=eo,
        ao=float((po == Y).mean()),
        xo=float((po == Y).all(axis=1).mean()),
        mifo=mifo,
        mafo=mafo,
        thr=thr,
        pe5=pe5,
        peo=peo,
        seps=seps,
        msep=float(seps.mean()),
    )


# ════════════════════════════════════════════════════════════════
#  DÉFINITION DES EXPÉRIENCES
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" EXPÉRIENCES")
print("=" * 72)

# Variantes de texte
raw = texts

bca_ctx = []
for i in range(N):
    p = prevs[i] if prevs[i] else ""
    n = nexts[i] if nexts[i] else ""
    bca_ctx.append(f"before:{p}{EOS}current:{texts[i]}{EOS}after:{n}{EOS}")

bca_empty = [f"before:{EOS}current:{s}{EOS}after:{EOS}" for s in texts]

EXPS = OrderedDict(
    [
        # ── H1 : add_special_tokens=True ──
        (
            "H1_raw",
            dict(
                texts=raw, sp=True, pad="longest",
                desc="phrase brute, sp=True",
            ),
        ),
        (
            "H1_raw_maxlen",
            dict(
                texts=raw, sp=True, pad="max_length",
                desc="phrase brute, sp=True, pad=max_length",
            ),
        ),
        (
            "H1_bca_ctx",
            dict(
                texts=bca_ctx, sp=True, pad="longest",
                desc="template BCA + contexte, sp=True",
            ),
        ),
        (
            "H1_bca_empty",
            dict(
                texts=bca_empty, sp=True, pad="longest",
                desc="template BCA (sans contexte), sp=True",
            ),
        ),
        # ── H2 : add_special_tokens=False ──
        (
            "H2_raw",
            dict(
                texts=raw, sp=False, pad="longest",
                desc="phrase brute, sp=False",
            ),
        ),
        (
            "H2_raw_maxlen",
            dict(
                texts=raw, sp=False, pad="max_length",
                desc="phrase brute, sp=False, pad=max_length (≈code officiel)",
            ),
        ),
        (
            "H2_bca_ctx",
            dict(
                texts=bca_ctx, sp=False, pad="longest",
                desc="template BCA + contexte, sp=False",
            ),
        ),
        (
            "H2_bca_empty",
            dict(
                texts=bca_empty, sp=False, pad="longest",
                desc="template BCA (sans contexte), sp=False",
            ),
        ),
    ]
)

# ════════════════════════════════════════════════════════════════
#  EXÉCUTION
# ════════════════════════════════════════════════════════════════
results = OrderedDict()

for name, cfg in EXPS.items():
    print(f"\n{'─' * 72}")
    print(f"  {name} : {cfg['desc']}")
    print(f"{'─' * 72}")

    # Diagnostic : tokenisation du premier exemple
    enc0 = tokenizer(
        cfg["texts"][0],
        add_special_tokens=cfg["sp"],
        truncation=True,
        max_length=MAXLEN,
    )
    ids0 = enc0["input_ids"][:10]
    tok0 = tokenizer.convert_ids_to_tokens(ids0)
    is_bos = ids0[0] == BOS_ID
    print(f"  texte[:70] = {cfg['texts'][0][:70]}")
    print(f"  tokens[:10]= {tok0}")
    print(f"  ids[:10]   = {ids0}")
    print(f"  pos-0 est <s> ? {'OUI' if is_bos else 'NON'}  (id={ids0[0]})")

    # Inférence
    t0 = time.time()
    lg = get_logits(cfg["texts"], cfg["sp"], cfg["pad"])
    dt = time.time() - t0
    print(f"  inférence  : {dt:.1f}s")

    # Évaluation
    res = evaluate(lg)
    res["hyp"] = "H1" if cfg["sp"] else "H2"
    results[name] = res

    print(f"\n  ┌── @seuil 0.5 ─────────────────────────────────────┐")
    print(
        f"  │ Erreurs: {res['e5']:5d}/{CELLS}  "
        f"Acc: {res['a5']:.4%}  µF1: {res['mif5']:.4f}  MF1: {res['maf5']:.4f} │"
    )
    print(f"  └──────────────────────────────────────────────────────┘")
    print(f"  ┌── @seuils optimaux ──────────────────────────────────┐")
    print(
        f"  │ Erreurs: {res['eo']:5d}/{CELLS}  "
        f"Acc: {res['ao']:.4%}  µF1: {res['mifo']:.4f}  MF1: {res['mafo']:.4f} │"
    )
    print(f"  └──────────────────────────────────────────────────────┘")
    print(f"  Séparation moyenne des logits : {res['msep']:+.2f}")

    # Détail par label
    for j in range(L):
        e5j = res["pe5"][j]
        eoj = res["peo"][j]
        tj  = res["thr"][j]
        sj  = res["seps"][j]
        st  = "✓" if eoj == 0 else f"✗{eoj}"
        print(
            f"    {LABELS[j]:<20s}  e@.5={e5j:4d}  e@opt={eoj:4d}  "
            f"thr={tj:.4f}  sep={sj:+6.2f}  {st}"
        )

# ════════════════════════════════════════════════════════════════
#  TABLEAU COMPARATIF
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" TABLEAU COMPARATIF (trié par erreurs optimales)")
print("=" * 72)

ranked = sorted(results, key=lambda n: results[n]["eo"])

hdr = (
    f"{'Expérience':<22s} {'H':>2s} {'err@.5':>7s} {'acc@.5':>8s} "
    f"{'err@opt':>8s} {'acc@opt':>8s} {'µF1@opt':>8s} {'sep':>6s}"
)
print(hdr)
print("─" * len(hdr))
for n in ranked:
    r = results[n]
    print(
        f"{n:<22s} {r['hyp']:>2s} {r['e5']:7d} {r['a5']:8.4%} "
        f"{r['eo']:8d} {r['ao']:8.4%} {r['mifo']:8.4f} {r['msep']:+6.2f}"
    )

# ════════════════════════════════════════════════════════════════
#  FACE-À-FACE  H1 vs H2
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" FACE-À-FACE : H1 vs H2  (même texte, sp différent)")
print("=" * 72)

pairs = [
    ("H1_raw", "H2_raw"),
    ("H1_raw_maxlen", "H2_raw_maxlen"),
    ("H1_bca_ctx", "H2_bca_ctx"),
    ("H1_bca_empty", "H2_bca_empty"),
]

h1w = h2w = 0
for h1n, h2n in pairs:
    r1, r2 = results[h1n], results[h2n]
    d5 = r1["e5"] - r2["e5"]
    do = r1["eo"] - r2["eo"]
    dsep = r1["msep"] - r2["msep"]
    if do > 0:
        w = "→ H2 gagne"
        h2w += 1
    elif do < 0:
        w = "→ H1 gagne"
        h1w += 1
    else:
        w = "→ ÉGALITÉ"
    print(f"\n  {h1n} vs {h2n}")
    print(f"    err@0.5 : H1={r1['e5']:5d}  H2={r2['e5']:5d}  (Δ={d5:+d})")
    print(f"    err@opt : H1={r1['eo']:5d}  H2={r2['eo']:5d}  (Δ={do:+d})")
    print(f"    sep moy : H1={r1['msep']:+.2f}  H2={r2['msep']:+.2f}  (Δ={dsep:+.2f})")
    print(f"    {w}")

print(f"\n  Score global : H1 gagne {h1w} / H2 gagne {h2w}")

# ════════════════════════════════════════════════════════════════
#  DIAGNOSTIC : norme du hidden state en position 0
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" DIAGNOSTIC : norme L2 du dernier hidden state en position 0")
print("=" * 72)

n_diag = min(200, N)

for sp, tag in [(True, "H1 (avec <s>)"), (False, "H2 (sans <s>)")]:
    norms = []
    for s in range(0, n_diag, BATCH):
        batch = raw[s : s + BATCH]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAXLEN,
            padding="longest",
            add_special_tokens=sp,
        ).to(DEVICE)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[-1]
        norms.extend(hs[:, 0, :].norm(dim=-1).cpu().tolist())
    norms = np.array(norms)
    cv = norms.std() / norms.mean() if norms.mean() > 0 else float("inf")
    print(
        f"  {tag:20s}  mean={norms.mean():.3f}  std={norms.std():.3f}  "
        f"CV={cv:.4f}  [{norms.min():.2f}, {norms.max():.2f}]"
    )

# ════════════════════════════════════════════════════════════════
#  VERDICT
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(" VERDICT")
print("=" * 72)

best_name = ranked[0]
best_r = results[best_name]

print(f"\n  Meilleure expérience : {best_name}  (hypothèse {best_r['hyp']})")
print(f"  Erreurs @optimal     : {best_r['eo']} / {CELLS}  ({best_r['ao']:.4%})")
print(f"  µ-F1 @optimal        : {best_r['mifo']:.4f}")

r_h1 = results.get("H1_raw", {})
r_h2 = results.get("H2_raw", {})
if r_h1 and r_h2:
    print(f"\n  Comparaison directe (phrase brute) :")
    print(f"    H1 err@0.5={r_h1['e5']}  err@opt={r_h1['eo']}  µF1@opt={r_h1['mifo']:.4f}")
    print(f"    H2 err@0.5={r_h2['e5']}  err@opt={r_h2['eo']}  µF1@opt={r_h2['mifo']:.4f}")

print(f"\n  Face-à-face : H1 gagne {h1w} — H2 gagne {h2w}")

# Rassembler les signaux
signals = []
signals.append(best_r["hyp"])  # meilleure expérience
if h2w > h1w:
    signals.append("H2")
elif h1w > h2w:
    signals.append("H1")
signals.append("H2")  # a priori fort : le code officiel utilise sp=False

h1c = signals.count("H1")
h2c = signals.count("H2")
print(f"  Signaux : H1={h1c}  H2={h2c}")

if h2c > h1c:
    verdict = (
        "H2 est l'hypothèse la plus probable.\n"
        "  Le modèle a été fine-tuné avec add_special_tokens=False.\n"
        "  La tête de classification lit le PREMIER SUBWORD TOKEN\n"
        "  du texte en entrée (position 0), PAS le token <s>."
    )
elif h1c > h2c:
    verdict = (
        "H1 est l'hypothèse la plus probable.\n"
        "  Le modèle a été fine-tuné avec add_special_tokens=True.\n"
        "  La tête de classification lit le token <s> (BOS/CLS)\n"
        "  en position 0."
    )
else:
    verdict = "INCONCLUSIF : les signaux sont équilibrés entre H1 et H2."

print()
print("  ╔══════════════════════════════════════════════════════════╗")
for line in verdict.split("\n"):
    print(f"  ║  {line:<56s}║")
print("  ╚══════════════════════════════════════════════════════════╝")
```

```output
========================================================================
 CHARGEMENT MODÈLE & TOKENIZER
========================================================================
Labels (19) : ['Emo', 'Comportementale', 'Designee', 'Montree', 'Suggeree', 'Base', 'Complexe', 'Admiration', 'Autre', 'Colere', 'Culpabilite', 'Degout', 'Embarras', 'Fierte', 'Jalousie', 'Joie', 'Peur', 'Surprise', 'Tristesse']
BOS=<s> (id=5)   EOS=</s> (id=6)
  add_special_tokens=True  → ['<s>', '▁Il', '▁pleure', '.', '</s>']
  add_special_tokens=False → ['▁Il', '▁pleure', '.']

========================================================================
 CHARGEMENT DU JEU DE DONNÉES
========================================================================
Splits disponibles : ['train', 'validation', 'test']
Nombre de lignes : 27911
Colonnes         : ['previous_sentence', 'types', 'modes', 'categories', 'next_sentence', 'target_sentence', 'is_emotional']

  Exemple 0 :
    previous_sentence: None
    types: ['basic']
    modes: ['behavioral']
    categories: ['anger', 'other']
    next_sentence: 'Des manifestations ont été organisées en France pour dire «non à l’antisémitisme» et au rejet de l’autre. '
    target_sentence: 'Ces dernières semaines, plusieurs actes antisémites, c’est-à-dire dirigés contre des personnes juives ou des symboles j
    is_emotional: True

  Exemple 1 :
    previous_sentence: 'Ces dernières semaines, plusieurs actes antisémites, c’est-à-dire dirigés contre des personnes juives ou des symboles j
    types: ['basic']
    modes: ['behavioral']
    categories: ['anger', 'other']
    next_sentence: 'Pourquoi des gens s’attaquent-ils aux juifs ? '
    target_sentence: 'Des manifestations ont été organisées en France pour dire «non à l’antisémitisme» et au rejet de l’autre. '
    is_emotional: True

========================================================================
 PARSING DES LABELS
========================================================================
Exemples parsés : 27911   (ignorés : 0)
Matrice         : 27911 × 19 = 530309 cellules

Distribution des labels :
  Emo                   :  5374 / 27911  ( 19.3%)
  Comportementale       :  1242 / 27911  (  4.4%)
  Designee              :     0 / 27911  (  0.0%)
  Montree               :     0 / 27911  (  0.0%)
  Suggeree              :  1909 / 27911  (  6.8%)
  Base                  :     0 / 27911  (  0.0%)
  Complexe              :   568 / 27911  (  2.0%)
  Admiration            :   211 / 27911  (  0.8%)
  Autre                 :  1270 / 27911  (  4.6%)
  Colere                :  1180 / 27911  (  4.2%)
  Culpabilite           :    19 / 27911  (  0.1%)
  Degout                :    52 / 27911  (  0.2%)
  Embarras              :     0 / 27911  (  0.0%)
  Fierte                :   202 / 27911  (  0.7%)
  Jalousie              :     7 / 27911  (  0.0%)
  Joie                  :   888 / 27911  (  3.2%)
  Peur                  :  1047 / 27911  (  3.8%)
  Surprise              :   824 / 27911  (  3.0%)
  Tristesse             :   673 / 27911  (  2.4%)

  ⚠ Labels sans exemples positifs : ['Designee', 'Montree', 'Base', 'Embarras']

========================================================================
 EXPÉRIENCES
========================================================================

────────────────────────────────────────────────────────────────────────
  H1_raw : phrase brute, sp=True
────────────────────────────────────────────────────────────────────────
  texte[:70] = Ces dernières semaines, plusieurs actes antisémites, c’est-à-dire diri
  tokens[:10]= ['<s>', '▁Ces', '▁dernières', '▁semaines', ',', '▁plusieurs', '▁actes', '▁antisémite', 's', ',']
  ids[:10]   = [5, 515, 1194, 1075, 7, 247, 3109, 23973, 10, 7]
  pos-0 est <s> ? OUI  (id=5)

  ┌── @seuil 0.5 ─────────────────────────────────────┐
  │ Erreurs: 17155/530309  Acc: 96.7651%  µF1: 0.2444  MF1: 0.0638 │
  └──────────────────────────────────────────────────────┘
  ┌── @seuils optimaux ──────────────────────────────────┐
  │ Erreurs: 14061/530309  Acc: 97.3485%  µF1: 0.2164  MF1: 0.0738 │
  └──────────────────────────────────────────────────────┘
  Séparation moyenne des logits : +1.31
    Emo                   e@.5=4322  e@opt=4224  thr=0.6558  sep= +2.73  ✗4224
    Comportementale       e@.5=1217  e@opt=1214  thr=0.3456  sep= +2.20  ✗1214
    Designee              e@.5=  56  e@opt=   1  thr=0.9984  sep= +0.00  ✗1
    Montree               e@.5=  81  e@opt=   1  thr=0.9997  sep= +0.00  ✗1
    Suggeree              e@.5=1889  e@opt=1878  thr=0.1234  sep= +1.23  ✗1878
    Base                  e@.5=2809  e@opt=   1  thr=0.9998  sep= +0.00  ✗1
    Complexe              e@.5= 540  e@opt= 531  thr=0.0010  sep= +1.80  ✗531
    Admiration            e@.5= 208  e@opt= 208  thr=0.1980  sep= +0.73  ✗208
    Autre                 e@.5=1257  e@opt=1256  thr=0.1843  sep= +0.67  ✗1256
    Colere                e@.5=1164  e@opt=1162  thr=0.2284  sep= +1.17  ✗1162
    Culpabilite           e@.5=  19  e@opt=  20  thr=0.0589  sep= +2.88  ✗20
    Degout                e@.5=  52  e@opt=  49  thr=0.0531  sep= +1.73  ✗49
    Embarras              e@.5=  15  e@opt=   1  thr=0.9474  sep= +0.00  ✗1
    Fierte                e@.5= 195  e@opt= 193  thr=0.1429  sep= +1.84  ✗193
    Jalousie              e@.5=   7  e@opt=   7  thr=0.0160  sep= +0.45  ✗7
    Joie                  e@.5= 878  e@opt= 877  thr=0.3905  sep= +1.31  ✗877
    Peur                  e@.5= 987  e@opt= 983  thr=0.3882  sep= +3.09  ✗983
    Surprise              e@.5= 815  e@opt= 813  thr=0.2737  sep= +0.82  ✗813
    Tristesse             e@.5= 644  e@opt= 642  thr=0.4418  sep= +2.25  ✗642

────────────────────────────────────────────────────────────────────────
  H1_raw_maxlen : phrase brute, sp=True, pad=max_length
────────────────────────────────────────────────────────────────────────
  texte[:70] = Ces dernières semaines, plusieurs actes antisémites, c’est-à-dire diri
  tokens[:10]= ['<s>', '▁Ces', '▁dernières', '▁semaines', ',', '▁plusieurs', '▁actes', '▁antisémite', 's', ',']
  ids[:10]   = [5, 515, 1194, 1075, 7, 247, 3109, 23973, 10, 7]
  pos-0 est <s> ? OUI  (id=5)
  inférence  : 828.7s

  ┌── @seuil 0.5 ─────────────────────────────────────┐
  │ Erreurs: 17155/530309  Acc: 96.7651%  µF1: 0.2444  MF1: 0.0638 │
  └──────────────────────────────────────────────────────┘
  ┌── @seuils optimaux ──────────────────────────────────┐
  │ Erreurs: 14061/530309  Acc: 97.3485%  µF1: 0.2164  MF1: 0.0738 │
  └──────────────────────────────────────────────────────┘
  Séparation moyenne des logits : +1.31
    Emo                   e@.5=4322  e@opt=4224  thr=0.6558  sep= +2.73  ✗4224
    Comportementale       e@.5=1217  e@opt=1214  thr=0.3456  sep= +2.20  ✗1214
    Designee              e@.5=  56  e@opt=   1  thr=0.9984  sep= +0.00  ✗1
    Montree               e@.5=  81  e@opt=   1  thr=0.9997  sep= +0.00  ✗1
    Suggeree              e@.5=1889  e@opt=1878  thr=0.1234  sep= +1.23  ✗1878
    Base                  e@.5=2809  e@opt=   1  thr=0.9998  sep= +0.00  ✗1
    Complexe              e@.5= 540  e@opt= 531  thr=0.0010  sep= +1.80  ✗531
    Admiration            e@.5= 208  e@opt= 208  thr=0.1980  sep= +0.73  ✗208
    Autre                 e@.5=1257  e@opt=1256  thr=0.1843  sep= +0.67  ✗1256
    Colere                e@.5=1164  e@opt=1162  thr=0.2284  sep= +1.17  ✗1162
    Culpabilite           e@.5=  19  e@opt=  20  thr=0.0589  sep= +2.88  ✗20
    Degout                e@.5=  52  e@opt=  49  thr=0.0531  sep= +1.73  ✗49
    Embarras              e@.5=  15  e@opt=   1  thr=0.9474  sep= +0.00  ✗1
    Fierte                e@.5= 195  e@opt= 193  thr=0.1429  sep= +1.84  ✗193
    Jalousie              e@.5=   7  e@opt=   7  thr=0.0160  sep= +0.45  ✗7
    Joie                  e@.5= 878  e@opt= 877  thr=0.3905  sep= +1.31  ✗877
    Peur                  e@.5= 987  e@opt= 983  thr=0.3882  sep= +3.09  ✗983
    Surprise              e@.5= 815  e@opt= 813  thr=0.2737  sep= +0.82  ✗813
    Tristesse             e@.5= 644  e@opt= 642  thr=0.4418  sep= +2.25  ✗642

────────────────────────────────────────────────────────────────────────
  H1_bca_ctx : template BCA + contexte, sp=True
────────────────────────────────────────────────────────────────────────
  texte[:70] = before:</s>current:Ces dernières semaines, plusieurs actes antisémites
  tokens[:10]= ['<s>', '▁be', 'for', 'e', ':', '</s>', '▁', 'current', ':', 'C']
  ids[:10]   = [5, 2446, 6270, 35, 92, 6, 21, 30340, 92, 228]
  pos-0 est <s> ? OUI  (id=5)
  inférence  : 264.5s

  ┌── @seuil 0.5 ─────────────────────────────────────┐
  │ Erreurs: 10140/530309  Acc: 98.0879%  µF1: 0.7278  MF1: 0.5451 │
  └──────────────────────────────────────────────────────┘
  ┌── @seuils optimaux ──────────────────────────────────┐
  │ Erreurs:  3421/530309  Acc: 99.3549%  µF1: 0.8876  MF1: 0.5700 │
  └──────────────────────────────────────────────────────┘
  Séparation moyenne des logits : +10.17
    Emo                   e@.5= 783  e@opt= 774  thr=0.6404  sep=+14.21  ✗774
    Comportementale       e@.5= 279  e@opt= 269  thr=0.7568  sep=+14.48  ✗269
    Designee              e@.5=1502  e@opt=   1  thr=1.0000  sep= +0.00  ✗1
    Montree               e@.5= 928  e@opt=   1  thr=1.0000  sep= +0.00  ✗1
    Suggeree              e@.5= 706  e@opt= 698  thr=0.5891  sep=+10.86  ✗698
    Base                  e@.5=4071  e@opt=   1  thr=1.0000  sep= +0.00  ✗1
    Complexe              e@.5= 166  e@opt= 161  thr=0.7158  sep=+13.80  ✗161
    Admiration            e@.5=  98  e@opt=  98  thr=0.4974  sep=+11.12  ✗98
    Autre                 e@.5= 259  e@opt= 248  thr=0.6826  sep=+15.24  ✗248
    Colere                e@.5= 213  e@opt= 212  thr=0.5061  sep=+15.06  ✗212
    Culpabilite           e@.5=  19  e@opt=  20  thr=0.2097  sep= +9.67  ✗20
    Degout                e@.5=  51  e@opt=  38  thr=0.1468  sep= +8.91  ✗38
    Embarras              e@.5= 152  e@opt=   1  thr=0.9766  sep= +0.00  ✗1
    Fierte                e@.5=  76  e@opt=  68  thr=0.7579  sep=+13.31  ✗68
    Jalousie              e@.5=   7  e@opt=   8  thr=0.0843  sep= +9.72  ✗8
    Joie                  e@.5= 220  e@opt= 220  thr=0.5030  sep=+14.19  ✗220
    Peur                  e@.5= 230  e@opt= 229  thr=0.4940  sep=+15.03  ✗229
    Surprise              e@.5= 164  e@opt= 161  thr=0.4299  sep=+14.33  ✗161
    Tristesse             e@.5= 216  e@opt= 213  thr=0.5405  sep=+13.36  ✗213

────────────────────────────────────────────────────────────────────────
  H1_bca_empty : template BCA (sans contexte), sp=True
────────────────────────────────────────────────────────────────────────
  texte[:70] = before:</s>current:Ces dernières semaines, plusieurs actes antisémites
  tokens[:10]= ['<s>', '▁be', 'for', 'e', ':', '</s>', '▁', 'current', ':', 'C']
  ids[:10]   = [5, 2446, 6270, 35, 92, 6, 21, 30340, 92, 228]
  pos-0 est <s> ? OUI  (id=5)
  inférence  : 142.3s

  ┌── @seuil 0.5 ─────────────────────────────────────┐
  │ Erreurs: 11435/530309  Acc: 97.8437%  µF1: 0.6909  MF1: 0.5082 │
  └──────────────────────────────────────────────────────┘
  ┌── @seuils optimaux ──────────────────────────────────┐
  │ Erreurs:  4875/530309  Acc: 99.0807%  µF1: 0.8396  MF1: 0.5354 │
  └──────────────────────────────────────────────────────┘
  Séparation moyenne des logits : +9.45
    Emo                   e@.5=1171  e@opt=1169  thr=0.4754  sep=+13.50  ✗1169
    Comportementale       e@.5= 351  e@opt= 346  thr=0.3884  sep=+14.09  ✗346
    Designee              e@.5=1518  e@opt=   1  thr=1.0000  sep= +0.00  ✗1
    Montree               e@.5= 812  e@opt=   1  thr=0.9999  sep= +0.00  ✗1
    Suggeree              e@.5= 982  e@opt= 963  thr=0.6666  sep= +9.99  ✗963
    Base                  e@.5=4058  e@opt=   1  thr=1.0000  sep= +0.00  ✗1
    Complexe              e@.5= 241  e@opt= 233  thr=0.1700  sep=+12.43  ✗233
    Admiration            e@.5= 116  e@opt= 112  thr=0.6849  sep=+10.47  ✗112
    Autre                 e@.5= 347  e@opt= 345  thr=0.5147  sep=+14.70  ✗345
… [output truncated]
```
