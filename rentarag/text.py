"""Sentence splitting and within-sentence word windows.

Both functions reproduce the reference implementation used in the paper
(``reproduce/experiments/watermark_dirbucket.py``) exactly; the provider and the
auditor must segment text identically for buckets to line up.
"""

from __future__ import annotations

import re

__all__ = ["split_sentences", "sentence_windows", "text_windows"]

_SENT_RE = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences.

    A regex split on ``. ? !`` followed by whitespace (with guards for common
    abbreviations), after which fragments shorter than 40 characters are
    merged into the preceding sentence to avoid unstable micro-sentences.
    """
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    buf = ""
    for part in _SENT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) < 40 and buf:
            buf += " " + part
        else:
            if buf:
                out.append(buf)
            buf = part
    if buf:
        out.append(buf)
    return out


def sentence_windows(sentence: str, win_words: int = 28, stride_words: int = 14) -> list[str]:
    """Overlapping word windows inside one sentence.

    Sentences with at most ``win_words`` words yield a single window. Longer
    sentences yield windows every ``stride_words`` words plus a tail window so
    the last word is always covered. Windows never cross sentence boundaries.
    """
    toks = sentence.strip().split()
    n = len(toks)
    if n == 0:
        return []
    if n <= win_words:
        return [" ".join(toks)]
    out = [" ".join(toks[s : s + win_words]) for s in range(0, n - win_words + 1, max(1, stride_words))]
    tail = " ".join(toks[n - win_words : n])
    if out[-1] != tail:
        out.append(tail)
    return out


def text_windows(text: str, win_words: int = 28, stride_words: int = 14) -> list[str]:
    """All within-sentence windows of ``text``, in reading order."""
    out: list[str] = []
    for sent in split_sentences(text):
        out.extend(sentence_windows(sent, win_words, stride_words))
    return out
