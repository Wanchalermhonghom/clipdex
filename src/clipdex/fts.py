"""Build SQLite FTS5 queries from natural language (German and English).

The transcript has three full-text indexes. Search tries them in this order:
  word     the spoken words, with prefix match ("lauf" finds "läuft")
  stem     Snowball word stems in the file's language ("Bahnhöfe" finds "Bahnhof")
  trigram  parts of words, for compounds ("maschine" finds "Kaffeemaschine")
"""

from __future__ import annotations

import functools
import re
import unicodedata

_STOP_DE = """
aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes auch auf aus
bei bin bis bist da damit dann das dass dein deine dem den denn der des dessen deshalb die dies diese
diesem diesen dieser dieses doch dort du durch ein eine einem einen einer eines er es etwas euch euer
für gegen gewesen hab habe haben hat hatte hätte hier hin hinter ich ihm ihn ihnen ihr ihre im in indem
ins ist ja jede jedem jeden jeder jedes jetzt kann kein keine können man manche mein meine mich mir mit
muss nach nicht nichts noch nun nur ob oder ohne sehr sein seine sich sie sind so solche soll sollte sondern
über um und uns unser unter vom von vor war waren warum was weil welche wenn wer werde werden wie wieder
will wir wird wo wurde zu zum zur zwar zwischen video clip szene stelle moment wo rede rede spreche sage
sagt gesagt redet spricht erzähle erzählt geht
"""
_STOP_EN = """
a about above after again all am an and any are as at be because been before being below between both but
by can did do does doing down during each few for from further had has have having he her here hers him his
how i if in into is it its just me more most my no nor not now of off on once only or other our out over
own same she should so some such than that the their them then there these they this those through to too
under until up very was we were what when where which while who whom why will with you your video clip scene
moment part talk talks talking say says said speak speaks
"""
STOPWORDS = set(_STOP_DE.split()) | set(_STOP_EN.split())

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_PHRASE_RE = re.compile(r'"([^"]+)"')


def fold(text: str) -> str:
    """Lower-case and strip diacritics, like FTS5 'unicode61 remove_diacritics 2'."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def parse(query: str) -> tuple[list[str], list[str]]:
    """Return (phrases, terms). Text in double quotes becomes an exact phrase."""
    phrases = [p.strip() for p in _PHRASE_RE.findall(query) if p.strip()]
    rest = _PHRASE_RE.sub(" ", query)
    tokens = [t for t in _TOKEN_RE.findall(rest) if not t.isdigit() or len(t) > 1]
    terms = [t for t in tokens if t.casefold() not in STOPWORDS]
    if not terms and not phrases:
        terms = tokens  # query made only of stop words: keep them
    seen: set[str] = set()
    unique = []
    for t in terms:
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return phrases, unique


def _phrase(text: str) -> str | None:
    words = _TOKEN_RE.findall(text)
    return '"' + " ".join(words) + '"' if words else None


def _term(t: str) -> str:
    return f'"{t}"*' if len(t) >= 3 else f'"{t}"'


def build(query: str) -> list[str]:
    """FTS5 MATCH expressions to try in order: all terms (AND), then any term (OR)."""
    phrases, terms = parse(query)
    parts = [p for p in (_phrase(x) for x in phrases) if p] + [_term(t) for t in terms]
    if not parts:
        return []
    out = [" AND ".join(parts)]
    if len(parts) > 1 and not phrases:
        out.append(" OR ".join(parts))
    return out


# Whisper language code -> Snowball stemmer name
_SNOWBALL = {
    "de": "german", "en": "english", "fr": "french", "es": "spanish", "it": "italian", "nl": "dutch",
    "pt": "portuguese", "sv": "swedish", "da": "danish", "no": "norwegian", "nn": "norwegian",
    "fi": "finnish", "ru": "russian", "tr": "turkish", "hu": "hungarian", "ro": "romanian",
}
TRIGRAM_MIN = 4  # shorter parts match too much ("rot" is in "Brot")


@functools.cache
def _stemmer(language: str | None):
    name = _SNOWBALL.get((language or "").lower())
    if name is None:
        return None
    import snowballstemmer

    return snowballstemmer.stemmer(name)


def stem_word(word: str, language: str | None) -> str:
    stemmer = _stemmer(language)
    word = word.casefold()
    return fold(stemmer.stemWord(word) if stemmer else word)


def stem_text(text: str, language: str | None) -> str:
    """The text as word stems in its language, for the stem index."""
    return " ".join(stem_word(w, language) for w in _TOKEN_RE.findall(text))


def trigram_text(text: str) -> str:
    """The text for the trigram index: lower case, without diacritics."""
    return fold(text)


def build_stem(query: str, languages: list[str]) -> list[str]:
    """Like build(), but over word stems. Each term is stemmed in every language of the index."""
    phrases, terms = parse(query)
    langs = [lang for lang in dict.fromkeys(languages) if _stemmer(lang)] or [None]
    parts = []
    for p in phrases:
        for lang in langs[:1]:  # a phrase is stemmed in the first language only
            words = [stem_word(w, lang) for w in _TOKEN_RE.findall(p)]
            if words:
                parts.append('"' + " ".join(words) + '"')
    for t in terms:
        stems = list(dict.fromkeys(stem_word(t, lang) for lang in langs))
        alts = [f'"{s}"*' if len(s) >= 3 else f'"{s}"' for s in stems if s]
        if alts:
            parts.append(alts[0] if len(alts) == 1 else "(" + " OR ".join(alts) + ")")
    if not parts:
        return []
    out = [" AND ".join(parts)]
    if len(parts) > 1 and not phrases:
        out.append(" OR ".join(parts))
    return out


def build_trigram(query: str) -> list[str]:
    """Like build(), but finds terms inside longer words. Terms shorter than TRIGRAM_MIN are left out."""
    phrases, terms = parse(query)
    parts = ['"' + fold(" ".join(_TOKEN_RE.findall(p))) + '"' for p in phrases if _TOKEN_RE.findall(p)]
    long_terms = [t for t in terms if len(fold(t)) >= TRIGRAM_MIN]
    parts += [f'"{fold(t)}"' for t in long_terms]
    if not parts:
        return []
    # "All terms" only when no term was left out; otherwise only the "any term" level.
    out = [" AND ".join(parts) if len(long_terms) == len(terms) else ""]
    if (len(parts) > 1 or len(long_terms) < len(terms)) and not phrases:
        out.append(" OR ".join(parts))
    return out


def match_terms(query: str) -> list[str]:
    """Folded words used to find the exact spoken moment inside a segment."""
    phrases, terms = parse(query)
    words = [w for p in phrases for w in _TOKEN_RE.findall(p)][:1] + terms
    return [fold(w) for w in words]


def first_word_time(words: list, terms: list[str], language: str | None = None) -> float | None:
    """Start time of the first spoken word that matches one of the terms.

    A word matches when it begins with the term, has the same stem, or contains the term
    (compounds, terms of TRIGRAM_MIN or more letters).
    """
    if not terms:
        return None
    stems = {stem_word(t, language) for t in terms}
    for exact in (True, False):  # prefer a word that begins with the term
        for item in words:
            word, start = item[0], item[1]
            token = fold(word).strip(" .,!?;:\"'()[]{}…-–—")
            if not token:
                continue
            if exact:
                if any(token.startswith(t) for t in terms):
                    return float(start)
            elif stem_word(token, language) in stems or any(len(t) >= TRIGRAM_MIN and t in token for t in terms):
                return float(start)
    return None
