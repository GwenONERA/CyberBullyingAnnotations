#!/usr/bin/env python3
"""
Comparative distribution analysis of the 19 EMOTYC labels across three datasets:
  1. data/emotexttokids_gold_flat.xlsx  (EmoTextToKids — training data)
  2. outputs/homophobie/homophobie_annotations_gold_flat.xlsx
  3. outputs/obésité/obésité_annotations_gold_flat.xlsx

Outputs:
  - Per-label distribution tables
  - Density analysis (number of active labels per instance)
  - Co-occurrence matrices (emotion×emotion, emotion×mode)
  - Profile rarity analysis
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

# ════════════════════════════════════════════════════════════════
#  CONSTANTS — 19 labels in canonical order
# ════════════════════════════════════════════════════════════════
LABELS_19 = [
    "Emo",
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie", "Autre",
    "Comportementale", "Désignée", "Montrée", "Suggérée",
    "Base", "Complexe",
]
EMOTIONS_12 = LABELS_19[1:13]
MODES_4 = LABELS_19[13:17]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_xlsx(rel_path, split_filter=None):
    """Load an XLSX and return the (N, 19) binary label matrix."""
    df = pd.read_excel(os.path.join(ROOT, rel_path))
    if split_filter and "split" in df.columns:
        df = df[df["split"] == split_filter].reset_index(drop=True)
    mat = np.zeros((len(df), len(LABELS_19)), dtype=np.float32)
    for j, label in enumerate(LABELS_19):
        if label in df.columns:
            mat[:, j] = df[label].fillna(0).astype(int).values
    return mat


# ════════════════════════════════════════════════════════════════
#  LOAD DATASETS
# ════════════════════════════════════════════════════════════════
print("=" * 80)
print(" LOADING DATASETS")
print("=" * 80)

Y_hf       = load_xlsx(r"data\emotexttokids_gold_flat.xlsx")
Y_hf_train = load_xlsx(r"data\emotexttokids_gold_flat.xlsx", split_filter="train")
Y_homo     = load_xlsx(r"outputs\homophobie\homophobie_annotations_gold_flat.xlsx")
Y_obes     = load_xlsx(r"outputs\obésité\obésité_annotations_gold_flat.xlsx")

for name, Y in [("EmoTextToKids (all)", Y_hf), ("EmoTextToKids (train)", Y_hf_train),
                ("Homophobie", Y_homo), ("Obésité", Y_obes)]:
    print(f"  {name}: {Y.shape[0]} instances × {Y.shape[1]} labels")

datasets = {
    "EmoTextToKids (all)": Y_hf,
    "EmoTextToKids (train)": Y_hf_train,
    "Homophobie": Y_homo,
    "Obésité": Y_obes,
}


# ════════════════════════════════════════════════════════════════
#  1. PER-LABEL DISTRIBUTION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 1. PER-LABEL DISTRIBUTION (prevalence %)")
print("=" * 80)

header = f"{'Label':<20s}"
for name in datasets:
    header += f" {name:>22s}"
print(header)
print("-" * len(header))

for j, label in enumerate(LABELS_19):
    row = f"{label:<20s}"
    for name, Y in datasets.items():
        n = Y.shape[0]
        pos = int(Y[:, j].sum())
        pct = 100 * pos / n if n > 0 else 0
        row += f" {pos:>6d}/{n:<6d}({pct:5.1f}%)"
    print(row)


# ════════════════════════════════════════════════════════════════
#  2. DENSITY ANALYSIS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 2. DENSITY ANALYSIS (labels actifs par instance)")
print("=" * 80)

print(f"\n{'Metric':<35s}", end="")
for name in datasets:
    print(f" {name:>22s}", end="")
print()
print("-" * 120)

for metric_name, func in [
    ("Mean active labels (all 19)",     lambda Y: Y.sum(axis=1).mean()),
    ("Median active labels (all 19)",   lambda Y: np.median(Y.sum(axis=1))),
    ("Std active labels (all 19)",      lambda Y: Y.sum(axis=1).std()),
    ("Mean active emotions (12)",       lambda Y: Y[:, 1:13].sum(axis=1).mean()),
    ("Mean active modes (4)",           lambda Y: Y[:, 13:17].sum(axis=1).mean()),
    ("% instances with 0 active labels",lambda Y: 100 * (Y.sum(axis=1) == 0).mean()),
    ("% instances with ≥3 active emot.",lambda Y: 100 * (Y[:, 1:13].sum(axis=1) >= 3).mean()),
    ("% instances with ≥2 modes",       lambda Y: 100 * (Y[:, 13:17].sum(axis=1) >= 2).mean()),
    ("Max active labels seen",          lambda Y: int(Y.sum(axis=1).max())),
]:
    row = f"{metric_name:<35s}"
    for name, Y in datasets.items():
        val = func(Y)
        if isinstance(val, int):
            row += f" {val:>22d}"
        else:
            row += f" {val:>22.3f}"
    print(row)

# Density distribution (histogram)
print(f"\nDensity distribution (# active labels → # instances):")
for name, Y in datasets.items():
    totals = Y.sum(axis=1).astype(int)
    counts = Counter(totals)
    n = Y.shape[0]
    print(f"\n  {name} (n={n}):")
    for k in sorted(counts.keys()):
        bar = "█" * int(50 * counts[k] / n)
        print(f"    {k:>2d} labels: {counts[k]:>5d} ({100*counts[k]/n:5.1f}%) {bar}")

# Statistical test: density comparison
print(f"\n  Mann-Whitney U tests on density (all 19 labels):")
for name in ("Homophobie", "Obésité"):
    d = datasets[name].sum(axis=1)
    d_hf = Y_hf.sum(axis=1)
    U, p = stats.mannwhitneyu(d, d_hf, alternative="two-sided")
    effect_r = 1 - (2 * U) / (len(d) * len(d_hf))
    print(f"    {name} vs EmoTextToKids: U={U:.0f}, p={p:.2e}, effect_r={effect_r:.3f}")
    print(f"      Median {name}={np.median(d):.1f}, Median EmoTextToKids={np.median(d_hf):.1f}")


# ════════════════════════════════════════════════════════════════
#  3. CO-OCCURRENCE ANALYSIS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 3. CO-OCCURRENCE ANALYSIS")
print("=" * 80)


def compute_cooccurrence(Y, col_indices, col_names, normalize=True):
    """Compute co-occurrence matrix for given column indices."""
    sub = Y[:, col_indices]
    n = len(col_indices)
    cooc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cooc[i, j] = ((sub[:, i] == 1) & (sub[:, j] == 1)).sum()
    if normalize and Y.shape[0] > 0:
        cooc = cooc / Y.shape[0]
    return cooc


def print_cooc_matrix(cooc, names, title):
    print(f"\n  {title}")
    header = f"  {'':>15s}" + "".join(f" {n[:8]:>8s}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"  {name[:15]:>15s}"
        for j in range(len(names)):
            row += f" {cooc[i,j]:>8.4f}"
        print(row)


# Emotion × Emotion co-occurrences
emo_indices = list(range(1, 13))  # indices 1-12 in LABELS_19
for name, Y in datasets.items():
    cooc = compute_cooccurrence(Y, emo_indices, EMOTIONS_12)
    print_cooc_matrix(cooc, EMOTIONS_12, f"Emotion×Emotion co-occurrence (normalized) — {name}")

# Emotion × Mode co-occurrences
mode_indices = list(range(13, 17))  # indices 13-16 in LABELS_19
for name, Y in datasets.items():
    # Cross co-occurrence: emotion (rows) × mode (cols)
    n_emo = 12
    n_mode = 4
    cross = np.zeros((n_emo, n_mode))
    for i in range(n_emo):
        for j in range(n_mode):
            cross[i, j] = ((Y[:, 1 + i] == 1) & (Y[:, 13 + j] == 1)).sum()
    if Y.shape[0] > 0:
        cross = cross / Y.shape[0]
    print(f"\n  Emotion×Mode cross co-occurrence (normalized) — {name}")
    header = f"  {'':>15s}" + "".join(f" {m[:12]:>12s}" for m in MODES_4)
    print(header)
    for i, emo in enumerate(EMOTIONS_12):
        row = f"  {emo[:15]:>15s}"
        for j in range(n_mode):
            row += f" {cross[i,j]:>12.4f}"
        print(row)


# ════════════════════════════════════════════════════════════════
#  4. PROFILE ANALYSIS — identify rare/frequent patterns
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 4. PROFILE (LABEL CONFIGURATION) ANALYSIS")
print("=" * 80)


def get_profiles(Y):
    """Convert each row to a tuple for profile comparison."""
    return [tuple(int(x) for x in row) for row in Y]


profiles = {name: Counter(get_profiles(Y)) for name, Y in datasets.items()}

print(f"\n  Unique profiles: " + ", ".join(f"{n}={len(p)}" for n, p in profiles.items()))

profiles_train = profiles["EmoTextToKids (train)"]
profiles_all = profiles["EmoTextToKids (all)"]

for xlsx_name in ("Homophobie", "Obésité"):
    xlsx_profiles = profiles[xlsx_name]
    n_xlsx = datasets[xlsx_name].shape[0]
    print(f"\n  --- {xlsx_name} profiles not seen in EmoTextToKids training ---")
    unseen = {p: c for p, c in xlsx_profiles.items() if p not in profiles_train}
    total_unseen = sum(unseen.values())
    print(f"  {len(unseen)} unseen profiles covering {total_unseen}/{n_xlsx} "
          f"instances ({100*total_unseen/n_xlsx:.1f}%)")

    if unseen:
        print(f"  Top unseen profiles (by frequency in {xlsx_name}):")
        for profile, count in sorted(unseen.items(), key=lambda x: -x[1])[:10]:
            active_labels = [LABELS_19[i] for i, v in enumerate(profile) if v == 1]
            hf_all_count = profiles_all.get(profile, 0)
            print(f"    Count={count:>3d}, density={sum(profile)}, "
                  f"hf_all={hf_all_count}, labels={active_labels}")

    rare = {p: c for p, c in xlsx_profiles.items()
            if p in profiles_train and profiles_train[p] <= 5}
    total_rare = sum(rare.values())
    print(f"\n  {len(rare)} profiles seen ≤5 times in training, "
          f"covering {total_rare}/{n_xlsx} instances ({100*total_rare/n_xlsx:.1f}%)")


# ════════════════════════════════════════════════════════════════
#  5. CHI-SQUARED TEST PER LABEL
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 5. CHI-SQUARED TESTS: XLSX vs EmoTextToKids (per label)")
print("=" * 80)

for xlsx_name in ("Homophobie", "Obésité"):
    Y_xlsx = datasets[xlsx_name]
    print(f"\n  {xlsx_name} vs EmoTextToKids (all splits):")
    print(f"  {'Label':<20s} {'%XLSX':>8s} {'%HF':>8s} {'χ²':>10s} {'p-value':>12s} {'Cramér V':>10s}")
    print(f"  {'-'*72}")
    for j, label in enumerate(LABELS_19):
        n_xlsx = Y_xlsx.shape[0]
        n_hf = Y_hf.shape[0]
        pos_xlsx = int(Y_xlsx[:, j].sum())
        pos_hf = int(Y_hf[:, j].sum())
        neg_xlsx = n_xlsx - pos_xlsx
        neg_hf = n_hf - pos_hf
        table = np.array([[pos_xlsx, neg_xlsx], [pos_hf, neg_hf]])
        if (pos_xlsx + pos_hf == 0) or (neg_xlsx + neg_hf == 0):
            pct_xlsx = 100 * pos_xlsx / n_xlsx if n_xlsx > 0 else 0
            pct_hf = 100 * pos_hf / n_hf if n_hf > 0 else 0
            print(f"  {label:<20s} {pct_xlsx:>7.1f}% {pct_hf:>7.1f}% {'N/A':>10s} {'N/A':>12s} {'N/A':>10s}")
            continue
        try:
            chi2, p, dof, expected = stats.chi2_contingency(table, correction=True)
        except ValueError:
            pct_xlsx = 100 * pos_xlsx / n_xlsx if n_xlsx > 0 else 0
            pct_hf = 100 * pos_hf / n_hf if n_hf > 0 else 0
            print(f"  {label:<20s} {pct_xlsx:>7.1f}% {pct_hf:>7.1f}% {'N/A':>10s} {'N/A':>12s} {'N/A':>10s}")
            continue
        n_total = n_xlsx + n_hf
        cramer_v = np.sqrt(chi2 / n_total) if n_total > 0 else 0
        pct_xlsx = 100 * pos_xlsx / n_xlsx if n_xlsx > 0 else 0
        pct_hf = 100 * pos_hf / n_hf if n_hf > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<20s} {pct_xlsx:>7.1f}% {pct_hf:>7.1f}% {chi2:>10.2f} {p:>12.2e} {cramer_v:>9.4f} {sig}")


# ════════════════════════════════════════════════════════════════
#  6. JENSEN-SHANNON DIVERGENCE between distributions
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 6. JENSEN-SHANNON DIVERGENCE (label prevalence distributions)")
print("=" * 80)


def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (stats.entropy(p, m, base=2) + stats.entropy(q, m, base=2))


prev = {name: Y.mean(axis=0) for name, Y in datasets.items()}

for a, b in [("Homophobie", "EmoTextToKids (all)"),
             ("Obésité", "EmoTextToKids (all)"),
             ("Homophobie", "Obésité")]:
    print(f"  JSD({a}, {b}) = {js_divergence(prev[a], prev[b]):.6f}")


# ════════════════════════════════════════════════════════════════
#  7. SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 7. SUMMARY")
print("=" * 80)

for name, Y in datasets.items():
    density = Y.sum(axis=1).mean()
    emo_count = Y[:, 1:13].sum(axis=1).mean()
    print(f"  {name:<25s}  n={Y.shape[0]:>5d}  density={density:.2f}  emo_count={emo_count:.2f}")

print("\nAnalysis complete.")
