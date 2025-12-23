import math
import re
from typing import List


WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+")


def count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def sentence_count(text: str) -> int:
    return max(1, len([s for s in SENT_RE.split(text) if s.strip()]))


def flesch_kincaid_grade(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    sentences = sentence_count(text)
    words_count = len(words)
    return 0.39 * (words_count / sentences) + 11.8 * (syllables / words_count) - 15.59


def avg_sentence_length(text: str) -> float:
    words = tokenize_words(text)
    return len(words) / sentence_count(text)


def long_word_ratio(text: str, min_len: int = 8) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    long_words = [w for w in words if len(w) >= min_len]
    return len(long_words) / len(words)


def jargon_rate(text: str, jargon: List[str]) -> float:
    words = [w.lower() for w in tokenize_words(text)]
    if not words:
        return 0.0
    jargon_set = {j.lower() for j in jargon}
    count = sum(1 for w in words if w in jargon_set)
    return 100.0 * count / len(words)


def code_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_chars = sum(1 for ch in text if ch in "`{}[]()=<>:+-*/_#\\")
    return code_chars / len(text)


def technicality_components(text: str, jargon: List[str]) -> dict:
    return {
        "readability_grade": flesch_kincaid_grade(text),
        "avg_sentence_len": avg_sentence_length(text),
        "long_word_ratio": long_word_ratio(text),
        "jargon_rate": jargon_rate(text, jargon),
        "code_ratio": code_ratio(text),
    }
