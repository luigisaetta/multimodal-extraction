"""
Author: Luigi Saetta
Date last modified: 2026-02-01
Python Version: 3.11
License: MIT

Description:
    This script processes text extracted, using docling, from documents to remove noise lines.

    PATCH (2026-02-01):
    - Preserve fenced code blocks (``` / ~~~) verbatim.
      Cleanup is applied ONLY outside fenced code blocks to avoid breaking Markdown fences
      and to prevent removing/altering code content.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Caption patterns (keep them!)
_CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:fig(?:ure)?|fig\.)\s*\d+"
    r"|(?:tab(?:le)?|tab\.)\s*\d+"
    r"|tabella\s*\d+"
    r")"
    r"\s*[:.\-)]?\s+",
    re.IGNORECASE,
)

# Also keep lines that look like "Figure 1" without trailing text
_CAPTION_BARE_RE = re.compile(
    r"^\s*(?:fig(?:ure)?|fig\.|tab(?:le)?|tab\.|tabella)\s*\d+\s*$",
    re.IGNORECASE,
)


def _is_caption_line(line: str) -> bool:
    s = line.strip()
    return bool(_CAPTION_RE.match(s) or _CAPTION_BARE_RE.match(s))


def _looks_like_markdown_structure(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith(("##", "###", "- ", "* ", "> ")):
        return True
    if s.startswith("|") and s.endswith("|"):
        return True
    return False


def _is_noise_label_line(line: str, max_chars: int = 20, max_words: int = 3) -> bool:
    """
    Heuristic: detect tiny "diagram label" lines (often extracted from figures).
    Conservative: only drop when it strongly looks like noise.
    """
    s = line.strip()
    if not s:
        return False

    if _is_caption_line(s):
        return False

    if _looks_like_markdown_structure(s):
        return False

    words = s.split()
    if len(s) > max_chars or len(words) > max_words:
        return False

    # Too many non-letter chars => likely axis/diagram artifacts
    letters = sum(ch.isalpha() for ch in s)
    non_letters = len(s) - letters
    non_letter_ratio = non_letters / max(1, len(s))

    # Many uppercase tokens (e.g., "LLM", "FLOW", "NODE") in short lines
    alpha_tokens = [w for w in words if any(c.isalpha() for c in w)]
    upper_tokens = sum(w.isupper() for w in alpha_tokens)

    # Extremely short single token is often noise
    # (unless it ends with '.' and looks like sentence)
    looks_sentence_like = s.endswith((".", ":", ";")) and len(words) >= 2

    label_like = (non_letter_ratio >= 0.35) or (upper_tokens >= 2)
    very_short = len(words) == 1 and len(s) <= 8

    if looks_sentence_like:
        return False

    return bool(label_like or very_short)


def _split_by_fenced_code_blocks(text: str) -> List[tuple[bool, List[str]]]:
    """
    Split text into segments separated by fenced code blocks.

    Returns:
        List of segments (is_code, lines)
        - is_code=True  => segment is a fenced code block INCLUDING fence lines
        - is_code=False => segment is normal text (outside code fences)

    Supports fences started with ``` or ~~~. End fence must match the opening marker.
    """
    lines = text.splitlines()
    segments: List[tuple[bool, List[str]]] = []

    in_code = False
    fence_marker: Optional[str] = None
    buf: List[str] = []

    def flush(current_is_code: bool) -> None:
        nonlocal buf
        if buf:
            segments.append((current_is_code, buf))
            buf = []

    for line in lines:
        stripped = line.lstrip()

        is_fence = stripped.startswith("```") or stripped.startswith("~~~")
        if is_fence:
            marker = "```" if stripped.startswith("```") else "~~~"

            if not in_code:
                # entering code block:
                # flush any pending non-code lines, then start code block buffer
                flush(False)
                in_code = True
                fence_marker = marker
                buf.append(line)
            else:
                # closing code block only if marker matches opening fence
                if fence_marker == marker:
                    buf.append(line)
                    flush(True)
                    in_code = False
                    fence_marker = None
                else:
                    # fence-like line inside code block but different marker: keep as code content
                    buf.append(line)
            continue

        buf.append(line)

    # flush remaining lines
    flush(in_code)

    return segments


def _cleanup_outside_code_segment(
    lines: List[str],
    collapse_runs: bool,
    run_min_lines: int,
) -> List[str]:
    """
    Apply your existing noise removal logic to a list of lines (outside code fences).
    Returns cleaned lines (still outside code fences).
    """
    out: List[str] = []

    noise_run = 0
    for line in lines:
        if _is_noise_label_line(line):
            noise_run += 1
            continue

        # Close run
        if noise_run and collapse_runs and noise_run >= run_min_lines:
            out.append("[FIGURE LABELS REMOVED]")
        noise_run = 0

        out.append(line)

    # End run at EOF
    if noise_run and collapse_runs and noise_run >= run_min_lines:
        out.append("[FIGURE LABELS REMOVED]")

    # Cleanup: avoid many consecutive blank lines (optional)
    cleaned: List[str] = []
    blank_streak = 0
    for line in out:
        if not line.strip():
            blank_streak += 1
            if blank_streak <= 2:
                cleaned.append("")
        else:
            blank_streak = 0
            cleaned.append(line)

    return cleaned


def cleanup_docling_text_keep_captions(
    text: str,
    collapse_runs: bool = True,
    run_min_lines: int = 6,
) -> str:
    """
    Remove 'diagram label' noise lines while keeping captions.

    IMPORTANT:
    - Cleanup is applied ONLY outside fenced code blocks (``` / ~~~).
      Code blocks are preserved verbatim to avoid breaking Markdown fences.

    collapse_runs:
      if True, when many consecutive noise lines appear, replace them with one marker.
      (Marker is optional; you can also skip adding it.)
    """
    segments = _split_by_fenced_code_blocks(text)

    rebuilt_lines: List[str] = []
    for is_code, seg_lines in segments:
        if is_code:
            # Preserve code blocks EXACTLY (including fences)
            rebuilt_lines.extend(seg_lines)
        else:
            cleaned = _cleanup_outside_code_segment(
                seg_lines,
                collapse_runs=collapse_runs,
                run_min_lines=run_min_lines,
            )
            rebuilt_lines.extend(cleaned)

    # Final join + trim outer whitespace; keep internal structure as-is
    return "\n".join(rebuilt_lines).strip()
