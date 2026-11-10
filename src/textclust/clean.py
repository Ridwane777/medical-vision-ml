"""Text normalization, stdlib language detection and corpus filtering."""
from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

log = logging.getLogger(__name__)

FRENCH_STOP_WORDS = frozenset("""
a ai aie aient ais ait alors as au aucun aucune aupres auquel aura aurai auraient aurais aurait
aussi autre autres aux auxquelles auxquels avaient avais avait avant avec avez aviez avions avoir
avons ayant bien c ca car ce ceci cela celle celles celui cependant certain certaines certains ces
cet cette ceux chaque chez ci comme comment d dans de des deja depuis donc dont du elle elles en
encore entre es est et etaient etais etait ete etes etre eu eux fait faire fois font hors ici il
ils j je jusqu l la le les leur leurs lors lui m ma mais me meme mes moi moins mon n ne ni non nos
notre nous on ont ou par parce pas peu peut plus pour pourquoi puis qu quand que quel quelle
quelles quels qui quoi s sa sans se sera serai serait ses si sien soi soit son sont sous suis sur
t ta tandis te tes toi ton tous tout toute toutes tres tu un une unes uns vers voici voila vos
votre vous y à ça été être déjà même très où était étaient étais été êtes aprés après
""".split())

# Fragments left by splitting English contractions on the apostrophe ("I've" -> "i ve").
ENGLISH_CONTRACTION_FRAGMENTS = frozenset("s t d m ve ll re don didn doesn isn wasn aren couldn wouldn shouldn haven hasn won can".split())

LANGUAGE_STOP_WORDS = {"en": frozenset(ENGLISH_STOP_WORDS) | ENGLISH_CONTRACTION_FRAGMENTS, "fr": FRENCH_STOP_WORDS}

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_NON_LETTER = re.compile(r"[^\w\s]|[\d_]")
_SPACES = re.compile(r"\s+")


def normalize(text) -> str:
    """NFKC-normalize, lowercase, remove URLs, digits and punctuation; accented letters are kept."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _URL.sub(" ", t)
    t = t.replace("'", " ").replace("’", " ")
    t = _NON_LETTER.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def detect_language(texts, sample: int = 2000) -> tuple[str, dict]:
    """Pick the language whose stop-word list covers the most tokens (stop-word overlap)."""
    tokens = [tok for txt in list(texts)[:sample] for tok in txt.split()]
    if not tokens:
        return "unknown", {}
    shares = {lang: sum(tok in sw for tok in tokens) / len(tokens) for lang, sw in LANGUAGE_STOP_WORDS.items()}
    best = max(shares, key=shares.get)
    if shares[best] < 0.05:
        return "unknown", shares
    return best, shares


def clean_corpus(df: pd.DataFrame, text_column: str, min_tokens: int = 2):
    """Return (clean frame with `clean_text`, report dict). Drops empty, too short and duplicate texts."""
    out = df.copy()
    out["clean_text"] = out[text_column].map(normalize)
    n0 = len(out)
    empty = out["clean_text"].eq("")
    out = out[~empty]
    short = out["clean_text"].str.split().str.len() < min_tokens
    out = out[~short]
    dup = out["clean_text"].duplicated()
    out = out[~dup].reset_index(drop=True)
    lang, shares = detect_language(out["clean_text"])
    report = {
        "input_rows": n0,
        "dropped_empty": int(empty.sum()),
        "dropped_too_short": int(short.sum()),
        "dropped_duplicates": int(dup.sum()),
        "kept": len(out),
        "min_tokens": min_tokens,
        "language": lang,
        "stop_word_share": {k: round(v, 4) for k, v in shares.items()},
    }
    log.info("cleaning: kept %d/%d (empty %d, short %d, duplicates %d), language=%s",
             len(out), n0, report["dropped_empty"], report["dropped_too_short"],
             report["dropped_duplicates"], lang)
    return out, report


def stop_words_for(language: str) -> list[str] | None:
    sw = LANGUAGE_STOP_WORDS.get(language)
    return sorted(sw) if sw else None
