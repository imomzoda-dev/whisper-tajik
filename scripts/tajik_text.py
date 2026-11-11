"""Tajik text cleanup.

fix_text()          -> cleans a training label but keeps case and punctuation
                       (so the model learns to write properly).
normalize_for_wer() -> lowercase, no punctuation; used only for scoring WER/CER.
"""
import re
import unicodedata

# Wrong code points that look like Tajik letters (from Kazakh, Belarusian,
# Macedonian, Ukrainian keyboards, or Latin homoglyphs).
_CHAR_MAP = {
    "һ": "ҳ", "Һ": "Ҳ",          # Bashkir/Kazakh shha -> Tajik ha
    "ӌ": "ҷ", "Ӌ": "Ҷ",          # Khakas che -> Tajik che with descender
    "ҹ": "ҷ", "Ҹ": "Ҷ",          # Abkhaz che -> Tajik
    "ќ": "қ", "Ќ": "Қ",          # Macedonian kje -> qof
    "ҝ": "қ", "Ҝ": "Қ",
    "ў": "ӯ", "Ў": "Ӯ",          # Belarusian short u -> Tajik long u
    "ӻ": "ғ", "Ӻ": "Ғ", "ґ": "ғ", "Ґ": "Ғ",
    "ї": "ӣ",
    " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",
    "‘": "'", "’": "'", "ʼ": "'", "`": "'",
    "“": "\"", "”": "\"", "„": "\"",
    "–": "-", "—": "-", "‑": "-",
    "…": "...",
}
_CHAR_RE = re.compile("|".join(map(re.escape, _CHAR_MAP)))

# Latin letters that look identical to Cyrillic ones; replaced only inside
# words that are otherwise Cyrillic (e.g. "дaр" typed with Latin "a").
_LAT2CYR = str.maketrans("aceopxyACEHKMOPTXBk", "асеорхуАСЕНКМОРТХВк")
_CYR = r"Ѐ-ӿ"
_MIXED_WORD = re.compile(rf"\b(?=\w*[{_CYR}])(?=\w*[A-Za-z])\w+\b")


def fix_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _CHAR_RE.sub(lambda m: _CHAR_MAP[m.group(0)], s)
    s = _MIXED_WORD.sub(lambda m: m.group(0).translate(_LAT2CYR), s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalize_for_wer(s: str) -> str:
    s = fix_text(s).lower()
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


if __name__ == "__main__":
    for t in ["Шаҳри романтикӣ, ки баъди нақли Лорд Байрон!",
              "Дaр шаҳр һамин ўзбек ќишлоқ",
              "Ӯ гуфт: «Ман омадам…»"]:
        print(repr(fix_text(t)), "|", repr(normalize_for_wer(t)))
