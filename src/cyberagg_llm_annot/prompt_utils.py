from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── Colonnes du corpus existant (annotations d'experts) ────────────────────
DEFAULT_LABEL_COLS = [
    "ROLE", "HATE", "TARGET", "VERBAL_ABUSE",
    "INTENTION", "CONTEXT", "SENTIMENT",
]

# ── 12 catégories émotionnelles ────────────────────────────────────────────
EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie", "Autre",
]

# ── 4 modes d'expression ──────────────────────────────────────────────────
MODES = ["Désignée", "Comportementale", "Suggérée", "Montrée"]

# ── System prompt (injecté via le paramètre "system" de Bedrock) ───────────
SYSTEM_PROMPT = """\
Tu es un annotateur expert en linguistique des émotions. Ta tâche : identifier et annoter les unités émotionnelles (SitEmo) dans UN SEUL message cible (TARGET).

═══ CADRE THÉORIQUE ═══

Une unité SitEmo est un triplet (Span ; Catégorie ; Mode) :
• Span — le segment textuel précis qui exprime l'émotion. Tu dois CITER les mots exacts.
• Catégorie — l'émotion exprimée parmi les 12 catégories ci-dessous.
• Mode — la MANIÈRE dont l'émotion est exprimée, parmi les 4 modes ci-dessous.

═══ LES 12 CATÉGORIES ÉMOTIONNELLES ═══

Émotions de base : Colère, Dégoût, Joie, Peur, Surprise, Tristesse
Émotions complexes : Admiration, Culpabilité, Embarras, Fierté, Jalousie
Autre : toute émotion ne rentrant dans aucune des 11 catégories précédentes (amour, haine, mépris, espoir, courage, soulagement, etc.)

Chaque catégorie est un ensemble large. Par exemple, Colère inclut : agacement, agressivité, irritation, indignation, rage, mécontentement, etc.

═══ LES 4 MODES D'EXPRESSION ═══

▸ Désignée — l'émotion est NOMMÉE explicitement par un terme du lexique émotionnel.
  Critère : le segment contient un mot dont la définition lexicographique renvoie à une émotion/sentiment/affect.
  Exemples : "heureux" → Désignée/Joie, "en colère" → Désignée/Colère, "honte" → Désignée/Embarras, "en danger" → Désignée/Peur.
  Cas des négations/modalisations : même si l'émotion est niée ("pas content") ou hypothétique ("aurait peur"), le terme émotionnel est annoté Désignée.

▸ Comportementale — l'émotion est inférée d'une MANIFESTATION PHYSIQUE ou d'une ATTITUDE DISCURSIVE.
  Critère : le segment décrit un comportement physiologique (pleurer, rougir, sursauter), comportemental (taper du poing, bloquer, manifester) ou discursif (accuser, reprocher, protester, rétorquer).
  Exemples : "éclata en sanglots" → Comportementale/Tristesse, "sourit" → Comportementale/Joie, "protestent" → Comportementale/Colère.
  Règle : le span englobe l'ensemble du syntagme verbal pertinent (verbe + arguments), en excluant les circonstants temporels.
  ATTENTION : si un même segment combine un terme du lexique émotionnel ET un comportement (ex: "saute de joie"), créer DEUX SitEmo : une Comportementale ("saute de joie") et une Désignée ("joie").

▸ Suggérée — l'émotion est DÉDUITE par le lecteur à partir de la description d'une SITUATION prototypiquement associée à un ressenti émotionnel.
  Critère : la situation décrite est conventionnellement/socio-culturellement associée à un ressenti. C'est souvent l'événement déclencheur qui est mis en mots.
  Exemples : "a gagné la course" → Suggérée/Joie, "a volé de l'argent" → Suggérée/Colère, "victimes d'injustices" → Suggérée/Colère.
  ATTENTION : n'annoter comme Suggérée que les situations PROTOTYPIQUEMENT associées à une émotion, pas celles dont l'interprétation émotionnelle ne fonctionne que dans le contexte du texte.

▸ Montrée — l'émotion TRANSPARAÎT au travers des caractéristiques FORMELLES de l'énoncé.
  Critère : l'émotion est visible par des marques lexicales (interjections : "Hélas !", "Merde !"), syntaxiques (phrases exclamatives, averbales, répétitions, clivées), typographiques (points d'exclamation, points de suspension), ou connotatives (insultes, termes péjoratifs à forte charge affective).
  Exemples : "!" → Montrée/Joie ou Surprise (selon contexte), "connard" → Montrée/Colère, "t'es dégeulasse" → Montrée/Colère.
  Règle émojis : les émojis sont des marqueurs d'émotion Montrée. Le span est l'émoji seul (ex: 😂 → Montrée/Joie, 🤮 → Montrée/Dégoût).
  Règle abréviations argotiques : décoder les abréviations — "ftg" = "ferme ta gueule" → Montrée/Colère, "tg" → Montrée/Colère.
  Règle "mdr"/"ptdr" : marqueur Montrée/Joie UNIQUEMENT si un affect positif est réellement exprimé (pas si c'est un simple ponctuant conversationnel).
  Règle ironie figée : "tu m'étonnes" est une locution figée signifiant "évidemment" → ne PAS annoter Surprise.
  Règle questions rhétoriques : une question rhétorique qui exprime de l'indignation/du mépris → Montrée/Colère (pas Surprise).

═══ RÈGLES GÉNÉRALES ═══

1. Annote UNIQUEMENT le message TARGET. PREV et NEXT sont du contexte pour comprendre la phrase, rien de plus.
2. Annote les émotions EXPRIMÉES par l'auteur du TARGET (pas celles de la victime, ni l'effet sur le lecteur).
3. NE SUR-ANNOTE PAS. Annote uniquement les unités émotionnelles clairement identifiables et prototypiques. En cas de forte hésitation, n'annote pas et signale l'ambiguïté.
4. Un span peut recevoir au plus DEUX catégories émotionnelles (champs "categorie" et éventuellement "categorie2").
5. Un span n'a qu'UN SEUL mode.
6. Les spans peuvent se chevaucher entre SitEmo différentes.
7. Le champ "span_text" doit citer les mots EXACTS tels qu'ils apparaissent dans le message TARGET (y compris les fautes d'orthographe, l'argot, les abréviations).
8. Une phrase peut ne contenir AUCUNE SitEmo — renvoie alors une liste vide.
9. Le champ "justification" doit être une phrase UNIQUE et CONCISE expliquant pourquoi ce mode et cette catégorie.

═══ FORMAT DE SORTIE — JSON STRICT ═══

Renvoie UNIQUEMENT un objet JSON brut, sans markdown, sans commentaire, avec cette structure exacte :

{
  "sitemo_units": [
    {
      "span_text": "<mots exacts du segment>",
      "mode": "<Désignée|Comportementale|Suggérée|Montrée>",
      "categorie": "<une des 12 catégories>",
      "categorie2": null ou "<catégorie secondaire si applicable>",
      "justification": "<une phrase concise>"
    }
  ],
  "ambiguities": ["<éventuels cas ambigus, max 3>"]
}

Si la phrase ne contient aucune émotion :
{
  "sitemo_units": [],
  "ambiguities": []
}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def build_annotations_block(
    parsed_labels: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Transforme les cellules parsées en bloc structuré pour le prompt."""
    block: Dict[str, Any] = {}
    for k, v in parsed_labels.items():
        if v["status"] == "value":
            block[k] = {"status": "value", "value": v["value"]}
        elif v["status"] == "no_consensus":
            block[k] = {"status": "no_consensus", "raw": v["raw"]}
        else:
            block[k] = {"status": "missing"}
    return block


def _is_block_empty(block: Dict[str, Any]) -> bool:
    """Vrai si toutes les annotations sont missing."""
    return all(v["status"] == "missing" for v in block.values())


def _fmt_msg(label: str, msg_repr: Optional[Dict[str, Any]]) -> str:
    if msg_repr is None:
        return f"{label}: (aucun message)"
    name = msg_repr.get("NAME", "?")
    role = msg_repr.get("ROLE", "")
    time = msg_repr.get("TIME", "")
    text = msg_repr.get("TEXT", "")
    header = f"[{name}]"
    if role:
        header += f" (role={role})"
    if time:
        header += f" (time={time})"
    return f'{label}: {header} "{text}"'


def build_user_message(
    thematique: str,
    prev_repr: Optional[Dict[str, Any]],
    target_repr: Dict[str, Any],
    next_repr: Optional[Dict[str, Any]],
    annotations_block: Optional[Dict[str, Any]] = None,
) -> str:
    """Construit le message *user* injecté dans la requête LLM."""
    lines: List[str] = [f"THÉMATIQUE: {thematique}", ""]

    # ── Fenêtre de contexte ──
    lines.append("<CONTEXT>")
    lines.append(_fmt_msg("PREV",   prev_repr))
    lines.append(_fmt_msg("TARGET", target_repr))
    lines.append(_fmt_msg("NEXT",   next_repr))
    lines.append("</CONTEXT>")

    lines += [
        "",
        "Annote les unités SitEmo du message TARGET. "
        "Renvoie UNIQUEMENT le JSON demandé, sans markdown ni commentaire.",
    ]
    return "\n".join(lines)
