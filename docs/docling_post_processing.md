# `docling_post_processing.py` — Post-processing for Docling Markdown (Noise Cleanup)

## What this module provides

This module cleans up text extracted by **Docling** (typically exported as Markdown) by removing “noise” lines that often come from **figures/diagrams** (axis labels, tiny tokens, stray OCR-like artifacts), while preserving content that matters for RAG and readability.

### Public API used by other modules
- **`cleanup_docling_text_keep_captions(text: str, collapse_runs: bool = True, run_min_lines: int = 6) -> str`**
  - Main entry point.
  - Removes short “label-like” lines outside code blocks.
  - Preserves captions (e.g., `Figure 1: ...`, `Tabella 2 ...`).
  - Preserves fenced code blocks verbatim (important for Markdown integrity).

---

## Key features

- **Caption preservation**
  - Keeps lines that match common caption patterns:
    - `Figure 1: ...`, `Fig. 2 ...`
    - `Table 3 ...`, `Tab. 4 ...`
    - `Tabella 5 ...`
  - Also keeps “bare” captions like `Figure 1` with no trailing text.

- **Conservative noise removal**
  - Only drops lines that strongly look like “diagram label junk”:
    - very short single-token labels
    - short lines with a high ratio of non-letter characters
    - short lines with many uppercase tokens (common in diagram artifacts)
  - Avoids removing Markdown structure lines (headings, list items, blockquotes, tables).

- **Run collapsing**
  - When many consecutive noise lines are removed, it can insert a single marker:
    - `[FIGURE LABELS REMOVED]`
  - This preserves the *fact* that something was removed without flooding output.

- **Markdown safety: code blocks preserved**
  - Fenced blocks started with ``` or ~~~ are detected and preserved verbatim.
  - Cleanup is applied **only outside** fences to avoid breaking:
    - code content
    - Markdown fences alignment
    - embedded snippets produced by Docling or later tooling

---

## Key design decisions

### 1) “Don’t break Markdown”
Instead of operating on raw text globally, the module splits the document into segments:
- **code segments**: kept exactly as-is
- **non-code segments**: cleaned with heuristics

This avoids a common failure mode: post-processing that accidentally alters fences or code content.

### 2) Heuristics are intentionally conservative
The goal is not “perfect” filtering (which risks deleting real content), but removing the most frequent, low-value artifacts while:
- keeping captions (high retrieval value)
- keeping tables and headings (structure)
- keeping readable prose untouched

### 3) Structure-aware filtering
Before considering a line as noise, the code checks whether it looks like Markdown structure:
- headings (`##`, `###`)
- lists (`- `, `* `)
- blockquotes (`> `)
- tables (`| ... |`)

This reduces the risk of damaging Docling’s table rendering.

---

## Practical notes
- The cleanup is best applied **after** Docling exports Markdown (especially tables).
- If you see legitimate short lines being removed, tighten the heuristic by lowering aggressiveness:
  - reduce `non_letter_ratio` threshold
  - disable `collapse_runs`
  - increase `run_min_lines`
- If too much figure-label noise remains, you can make it slightly more aggressive:
  - lower `max_chars/max_words` limits in `_is_noise_label_line`
  - reduce `run_min_lines`
