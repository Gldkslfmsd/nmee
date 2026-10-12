"""Regex-based punctuation fixes for ASR output (Luxembourgish).

Usage:
    from fix_punct import fix_punct
    fix_punct("am Sechs -Minuten -Takt") -> "am Sechs-Minuten-Takt"
"""
import re

# Each rule is (compiled pattern, replacement), applied in order.
RULES = [
    # 1) "Sechs -Minuten -Takt" -> "Sechs-Minuten-Takt", "Rheinland -Pfalz" -> "Rheinland-Pfalz".
    #    Only a space followed by a dash glued to the next word; a free-standing " - " is kept.
    (re.compile(r"(?<=\w) +-(?=\w)"), "-"),
    # 2) "d' Sitzung" -> "d'Sitzung" (also "D' " at sentence start, and the ’ apostrophe).
    (re.compile(r"\b([dD]['’]) +(?=\w)"), r"\1"),
    # 3) Decimal numbers: "12 ,22" -> "12,22", "14 ,08" -> "14,08" (also "12 .5" -> "12.5").
    #    Only when the separator is glued to the following digit, so lists like "12, 22" are kept.
    (re.compile(r"(?<=\d) +([,.])(?=\d)"), r"\1"),
    # 4) Shared second part of a compound: "Familljen - an Integratioun" -> "Familljen- an Integratioun",
    #    "Natur - a Forstverwaltung" -> "Natur- a Forstverwaltung".
    (re.compile(r"(?<=\w) +- +(?=(?:a|an|oder|respektiv|resp\.|bzw\.)\s)"), "- "),
    # 5) French elisions: "l 'égalité" -> "l'égalité", "qu 'il" -> "qu'il", "n 'accepte" -> "n'accepte".
    #    The apostrophe character is kept as it is.
    (re.compile(r"\b([cdjlmnstCDJLMNST]|[Qq]u|[Jj]usqu|[Ll]orsqu|[Pp]uisqu) +(['’])(?=\w)"), r"\1\2"),
    # 6) Web domains: "infolux .lu" -> "infolux.lu".
    (re.compile(r"(?<=\w) +\.(?=(?:lu|com|org|net|eu|de|fr|be)\b)"), "."),
    # 7) "Park &Ride" -> "Park & Ride".
    (re.compile(r"(?<=\w) +&(?=\w)"), " & "),
    # 8) ASR writes "Milliounen"/"Milliarden" as digits: "800 1000000 Euro" -> "800 Milliounen Euro",
    #    "1 1000000000" -> "1 Milliard". The MT misreads these, e.g. "400 1000000" -> "400,100,000".
    #    Must run after rule 3, so "2 ,3 1000000" is already "2,3 1000000".
    (re.compile(r"(?<![\d,.])1 1000000000\b"), "1 Milliard"),
    (re.compile(r"(?<=\d) 1000000000\b"), " Milliarden"),
    (re.compile(r"(?<![\d,.])1 1000000\b"), "1 Millioun"),
    (re.compile(r"(?<=\d) 1000000\b"), " Milliounen"),
]


def fix_punct(text: str) -> str:
    for pattern, repl in RULES:
        text = pattern.sub(repl, text)
    return text


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        sys.stdout.write(fix_punct(line))