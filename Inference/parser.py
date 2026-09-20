"""
Inference/parser.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import re
from typing import Dict, Optional


ANSWER_CLASSES = ("yes", "no", "abstain", "truncated", "empty", "unparsed")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_STRIP = ".,!?:;*\"'`"


def clean_text(text: Optional[str], reasoning: bool) -> str:
    text = "" if text is None else str(text)
    text = text.replace("Ġ", " ").replace("Ċ", "\n")
    if reasoning:
        text = _THINK.sub("", text)
    return text.strip()


def _tokens(pcfg: Dict, key: str):
    toks = pcfg[key]
    if not all(isinstance(t, str) for t in toks):
        raise TypeError(f"parser.{key} holds a non-string token {toks}; quote yes, no, true, and false in config.yaml")
    return toks


def _word_class(word: str, pcfg: Dict) -> int:
    w = word.lower().rstrip(_STRIP).lstrip(_STRIP)
    if w in _tokens(pcfg, "affirmative_tokens"):
        return 1
    if w in _tokens(pcfg, "negative_tokens"):
        return 0
    return -1


def parse_fixed(text: Optional[str], reasoning: bool, pcfg: Dict) -> int:
    text = clean_text(text, reasoning)
    if not text:
        return -1
    if reasoning:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            c = _word_class(lines[-1], pcfg)
            if c != -1:
                return c
    c = _word_class(text.split()[0], pcfg)
    if c != -1:
        return c
    head = text.lower()[: int(pcfg["head_chars"])]
    has_yes, has_no = re.search(r"\byes\b", head) is not None, re.search(r"\bno\b", head) is not None
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return -1


def parse_permissive(text: Optional[str], reasoning: bool, pcfg: Dict) -> int:
    fixed = parse_fixed(text, reasoning, pcfg)
    if fixed != -1:
        return fixed
    text = clean_text(text, reasoning)
    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if m is None:
        return -1
    return 1 if m.group(1).lower() == "yes" else 0


def is_abstention(text: Optional[str], pcfg: Dict) -> bool:
    low = clean_text(text, False).lower()
    return any(p in low for p in pcfg["abstention_phrases"])


def classify(raw: Optional[str], parsed: int, reasoning: bool,
             finish_reason: Optional[str], pcfg: Dict) -> str:
    if parsed == 1:
        return "yes"
    if parsed == 0:
        return "no"
    text = clean_text(raw, reasoning)
    if not clean_text(raw, False):
        return "empty"
    if reasoning:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines and is_abstention(lines[-1], pcfg) and finish_reason != "length":
            return "abstain"
        return "truncated"
    if is_abstention(raw, pcfg):
        return "abstain"
    if finish_reason == "length":
        return "truncated"
    return "unparsed"
