"""
Utterance-level language detection for the "auto" output-language policy.

Design goals
------------
* Deterministic and dependency-free, so it runs in well under a millisecond
  and is fully unit-testable without network or model downloads.
* Two signals, combined in a fixed priority order:

    1. Script.  Any Devanagari in the text is decisive for Hindi.
    2. Lexicon. Romanised Hindi ("Hinglish") has no Devanagari, so a
       small high-precision lexicon of function words and common verbs
       is used.  Two or more hits, or one hit in a short utterance,
       classify as Hinglish.
    3. STT hint. Deepgram's per-word language field (language=multi)
       is used only to break ties when the text itself is ambiguous.

* Hinglish resolves to a Hindi *reply*.  This matches the evaluation
  utterance U3 ("Order ORD124 cancel karna hai" -> expected reply hi)
  and real collections calls, where a code-mixed caller expects Hindi.

The detector never touches session state; the voice runtime owns the
decision to switch and logs it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TOKEN_RE = re.compile(r"[a-zA-Z\u0900-\u097F]+")

# High-precision romanised Hindi tokens. Deliberately excludes words that
# are also common English ("me", "to", "the", "is", "so", "us", "do",
# "main", "par", "char").
HINGLISH_LEXICON: frozenset[str] = frozenset(
    {
        # pronouns / determiners
        "mera", "meri", "mere", "mujhe", "mujhko", "hum", "humko", "hamara",
        "hamari", "aap", "aapka", "aapki", "aapke", "apna", "apni", "apne",
        "tum", "tumhara", "tumhari", "yeh", "woh", "iska", "uska", "kya",
        "kyu", "kyun", "kyon", "kaun", "kab", "kahan", "kaise", "kitna",
        "kitne", "kitni", "kaunsa", "kaunsi",
        # verbs / auxiliaries
        "hai", "hain", "hoon", "hun", "tha", "thi", "hoga", "hogi", "honge", "ho",
        "hua", "hui", "hue", "karna", "karo", "kar", "karunga", "karungi",
        "karenge", "kiya", "kiye", "karke", "kardo", "kardunga", "chahiye",
        "chahta", "chahti", "chahte", "sakta", "sakti", "sakte", "raha",
        "rahi", "rahe", "gaya", "gayi", "gaye", "dena", "dedo", "diya",
        "lena", "liya", "batao", "bataiye", "bata", "batana", "bolo",
        "boliye", "sunao", "dekho", "milega", "milegi", "paisa", "paise",
        "bhejo", "bhej", "bhejdo",
        # particles / adverbs / conjunctions
        "nahi", "nahin", "mat", "abhi", "kal", "aaj", "parso", "baad",
        "pehle", "phir", "fir", "lekin", "magar", "aur", "bhi",
        "sirf", "bas", "thoda", "zyada", "jyada", "kam", "bilkul", "haan",
        "ji", "theek", "thik", "accha", "acha", "achha", "sahi", "galat",
        "wala", "wali", "wale", "liye", "waste", "tak", "se", "ka", "ki",
        "ke", "ko", "ne", "pe", "mein", "ek", "teen",
        "paanch", "hazaar", "hazar", "lakh", "rupaye", "rupay", "rupees",
        # domain
        "bhugtan", "kisht", "karz", "udhar", "bakaya", "khata",
        "samasya", "madad", "shikayat", "galti",
    }
)

Language = str  # "english" | "hindi" | "hinglish"


@dataclass(frozen=True)
class LanguageDetection:
    detected: Language
    output_language: str  # "english" | "hindi"
    reason: str
    hinglish_hits: int = 0
    devanagari_chars: int = 0


def output_language_for(detected: Language) -> str:
    return "english" if detected == "english" else "hindi"


class LanguageDetector:
    """
    Stateless classifier. One instance per session is fine but not
    required.
    """

    def __init__(
        self,
        *,
        min_hinglish_hits: int = 2,
        short_utterance_tokens: int = 4,
    ) -> None:
        self.min_hinglish_hits = min_hinglish_hits
        self.short_utterance_tokens = short_utterance_tokens

    def detect(
        self,
        text: str,
        *,
        stt_hint: str | None = None,
    ) -> LanguageDetection:
        text = str(text or "")
        devanagari = len(DEVANAGARI_RE.findall(text))

        if devanagari:
            return LanguageDetection(
                detected="hindi",
                output_language="hindi",
                reason="devanagari_script",
                devanagari_chars=devanagari,
            )

        tokens = [t.lower() for t in TOKEN_RE.findall(text)]
        hits = sum(1 for t in tokens if t in HINGLISH_LEXICON)

        if not tokens:
            hint = _normalise_hint(stt_hint)
            return LanguageDetection(
                detected=hint or "english",
                output_language=output_language_for(hint or "english"),
                reason="empty_text_hint" if hint else "empty_text_default",
            )

        threshold = (
            1
            if len(tokens) <= self.short_utterance_tokens
            else self.min_hinglish_hits
        )

        if hits >= threshold:
            return LanguageDetection(
                detected="hinglish",
                output_language="hindi",
                reason=f"lexicon_hits>={threshold}",
                hinglish_hits=hits,
            )

        hint = _normalise_hint(stt_hint)

        # Text is plain Latin with no Hindi function words. Only trust the
        # STT hint when it says Hindi AND there is at least one weak hit;
        # a bare hint on clean English text is usually a mis-tag.
        if hint == "hindi" and hits >= 1:
            return LanguageDetection(
                detected="hinglish",
                output_language="hindi",
                reason="stt_hint_plus_weak_lexicon",
                hinglish_hits=hits,
            )

        return LanguageDetection(
            detected="english",
            output_language="english",
            reason="latin_no_hindi_markers",
            hinglish_hits=hits,
        )


def _normalise_hint(hint: str | None) -> Language | None:
    if not hint:
        return None
    h = str(hint).strip().lower()
    if h in {"hi", "hi-in", "hindi", "hin"}:
        return "hindi"
    if h in {"en", "en-in", "en-us", "en-gb", "english", "eng"}:
        return "english"
    return None


def majority_word_language(words: list[dict]) -> str | None:
    """
    Reduce Deepgram per-word `language` tags (present with language=multi)
    to a single hint. Returns None when no tags are present.
    """
    counts: dict[str, int] = {}
    for w in words or []:
        lang = _normalise_hint(w.get("language"))
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return None
    # Hindi wins ties: a code-mixed utterance should be treated as Hindi.
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == "hindi"))
    return best[0]
