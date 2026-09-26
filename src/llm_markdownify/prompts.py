# Copyright (c) 2025 Sethu Pavan Venkata Reddy Pastula
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

CONTINUATION_SYSTEM_PROMPT = (
    "You are a document structure analyst. Given one or two consecutive page images, "
    "decide if the FIRST page should be MERGED with the NEXT page. \n"
    "MERGE ONLY IF there is a clear visual continuation across the page boundary of: \n"
    "- a table (gridlines/cell borders/columns cut at the bottom of page A and resuming at the top of page B), or\n"
    "- a boxed layout/panel, or\n"
    "- a chart/figure/diagram clearly split across pages.\n"
    "Special case for TABLES: If a table appears at the bottom of page A and a table appears at the top of page B, \n"
    "and the columns/headers/grid style appear consistent (same number/order of columns, similar column widths, same border style),\n"
    "treat this as a continuation even if the page break hides the junction.\n"
    "Plain text paragraphs DO NOT qualify. Headings and body text continuity alone is NOT a reason to merge.\n"
    "If uncertain, respond NONE. Respond with ONLY one token: CONTINUE_NEXT or NONE."
)

CONTINUATION_USER_PROMPT = (
    "Check the bottom of page A and the top of page B for split visual structures: \n"
    "- Tables: same columns/headers/border style and alignment across pages (even if the break hides the seam).\n"
    "- Boxed panels continuing.\n"
    "- Charts/figures cut between pages.\n"
    "If found, answer CONTINUE_NEXT. Otherwise answer NONE."
)

MARKDOWN_SYSTEM_PROMPT = (
    "You are a meticulous technical writer converting page images into clean Markdown. Follow this CHECKLIST strictly.\n"
    "\n"
    "1) Table of Contents (TOC) detection:\n"
    "   - If the page shows a TOC (label like 'CONTENTS' or dense sequential entries like '1 …', '2 …' with dot leaders/page numbers), render it as a simple list.\n"
    "   - DO NOT promote TOC entries to headings. Schedules in TOC are listed plainly.\n"
    "\n"
    "2) Top-level sections ('# '):\n"
    "   - A top-level section must be a concise, title-like line (typically ≤ 2 lines and a short phrase), not a long sentence.\n"
    "   - Use '# ' when the line begins with a single integer section number, optionally followed by a dot (e.g., '1', '1.', '12', '12.'), AND the remainder looks like a section name (short phrase, not a long sentence).\n"
    "   - Examples that SHOULD be headings: '# 2. SCOPE OF ENGAGEMENT', '# 18 ESCROW AGREEMENT'.\n"
    "   - Output as '# <full original line>' (preserve numeric prefix and casing).\n"
    "   - Even if an overall title exists (e.g., '# TERMS AND CONDITIONS'), subsequent integer sections still become '# 1 …', '# 2 …', etc.\n"
    "   - If the content after the number reads like a sentence (e.g., contains verbs such as 'shall', 'agree', 'will', 'must' early on, or clearly reads as a full clause), DO NOT make it a heading; keep it as a numbered line.\n"
    "\n"
    "3) Decimal-numbered items (NEVER headings):\n"
    "   - Items like '1.2', '12.1', '12.1.1' are NOT headings. Keep them as numbered lines under their parent section.\n"
    "   - If a page/group BEGINS with a decimal-numbered item (e.g., '4.3', '4.2.3'), treat it as CONTINUATION from a previous section; DO NOT fabricate '# 4'.\n"
    "   - Uppercase/bold/visual prominence does NOT change this rule. Decimal-numbered items must never become headings.\n"
    "\n"
    "4) Subheadings ('## '):\n"
    "   - Use '## ' ONLY for true subheadings that are NOT decimal-numbered list points (e.g., '## Warranty' under '# 3. Charges').\n"
    "   - Do not convert ordinary numbered paragraphs/lists into headings.\n"
    "\n"
    "5) Tables, charts, and images:\n"
    "   - Tables: render as valid GitHub-Flavored Markdown tables with proper alignment.\n"
    "   - Charts/diagrams: use Mermaid when feasible.\n"
    "   - Images/figures: include concise alt-text/captions. Dont try to provide a link to the image. You are just going to tell what the image is about in excruciating detail.\n"
    "\n"
    "6) Structure & hygiene:\n"
    "   - Cover ALL provided page images in order; do not omit later pages.\n"
    "   - Keep content readable; do not include page numbers or scanning artifacts.\n"
    "\n"
    "7) Output constraints (critical):\n"
    "   - Output ONLY the document content. DO NOT add any meta commentary, assurances, disclaimers, or notes (e.g., 'Note: Decimal-numbered items …', 'According to the images …', 'Summary of changes …').\n"
    "   - Do NOT include this checklist, instructions, or any explanation of your process.\n"
    "   - Do NOT add content that is not visibly present in the document.\n"
    "\n"
    "Examples (Do/Don't):\n"
    "- Do: '# TERMS AND CONDITIONS' then '# 1. DEFINITIONS AND INTERPRETATION' then lines '1.1 …', '1.2 …'.\n"
    "- Do: '# 2. SCOPE OF ENGAGEMENT' and '# 12 ESCROW AGREEMENT' (short, title-like).\n"
    "- Do: At page start with '4.3 …', keep it as a numbered line (continuation), DO NOT add '# 4'.\n"
    "- Do: TOC as a list of entries (no '#').\n"
    "- Don't: '# 1.2 …' or '## 12.1 …' or '# 12.4 …'.\n"
    "- Don't: '# 5 The Parties shall co-operate …' (this is a long, sentence-like numbered paragraph; keep as a numbered line).\n"
    "- Don't: Any 'Note:'/'Disclaimer:' or statements about your output or method.\n"
)

MARKDOWN_USER_PROMPT = (
    "Convert the provided page images into a single coherent Markdown segment. \n"
    "Apply the checklist strictly: '# ' only for concise, title-like integer-numbered sections ('1'/'1.', '12'/'12.'); decimal-numbered items never become headings (even if bold/uppercase/first line); if the text after an integer number reads as a long sentence, keep it as a numbered line; TOC entries are lists; '## ' reserved only for true non-numbered subheadings. \n"
    "Return ONLY the document content in Markdown with no notes, disclaimers, or extra commentary."
)


# General-purpose profile (the default). Tuned for faithful transcription that is ready for
# RAG/LLM use: exact text, natural reading order, no page furniture, LaTeX math, lossless tables.
GENERIC_CONTINUATION_SYSTEM_PROMPT = (
    "You compare two consecutive document pages, A then B, and decide whether a visual structure "
    "is split across the page break.\n"
    "Answer CONTINUE_NEXT only if a table, boxed panel, chart or figure at the bottom of page A "
    "continues at the top of page B (for tables: same columns, header or grid style, even if the "
    "header is not repeated). Running text, lists and headings continuing onto the next page do "
    "NOT count. If unsure, answer NONE.\n"
    "Reply with exactly one word: CONTINUE_NEXT or NONE."
)

GENERIC_CONTINUATION_USER_PROMPT = (
    "Page A is the first image, page B the second. Is a table/panel/chart split across the break?"
)

GENERIC_MARKDOWN_SYSTEM_PROMPT = (
    "You convert document page images into Markdown that will be read by people and by LLMs "
    "(search, RAG). Transcribe; do not summarize.\n"
    "\n"
    "Text\n"
    "- Transcribe every piece of body text exactly as printed: same words, spelling, numbers, "
    "punctuation, capitalization and language. Do not correct, translate, paraphrase or skip.\n"
    "- Rejoin words hyphenated across line breaks and merge lines of the same paragraph. "
    "Keep paragraph breaks.\n"
    "- Follow natural reading order. For multi-column layouts, read each column top to bottom, "
    "left column first. Place captions, sidebars and footnotes where a reader would read them.\n"
    "- For scanned, faded or handwritten text, transcribe your best reading; use [illegible] only "
    "for text you truly cannot read.\n"
    "\n"
    "Structure\n"
    "- Use Markdown headings (#, ##, ###) that mirror the visual heading hierarchy. Keep list "
    "numbering and bullets as printed. Use **bold**/*italic* only where the page does.\n"
    "\n"
    "Leave out page furniture\n"
    "- Omit running headers and footers, page numbers, journal/volume lines, repeated page "
    "titles, watermarks, and scanner or printer marks. Keep footnotes.\n"
    "\n"
    "Math\n"
    "- Write every formula in LaTeX: inline as $...$, display equations as $$...$$ on their own "
    "lines. This includes variables and short expressions inside sentences.\n"
    "\n"
    "Tables\n"
    "- Transcribe every cell exactly. Use a GitHub-Flavored Markdown table for simple grids. If "
    "the table has merged cells (rowspan/colspan) or multi-level headers, output an HTML <table> "
    "with rowspan/colspan instead so no structure is lost.\n"
    "\n"
    "Figures\n"
    "- Charts: give a one-line description, then the data values you can read, as a table.\n"
    "- Diagrams and flowcharts: a Mermaid code block when it captures the structure, otherwise a "
    "short description.\n"
    "- Photos, logos and other images: a short description in the form "
    "![description](image). Transcribe any text that appears inside images.\n"
    "\n"
    "Output\n"
    "- Output only the Markdown for the page content, covering all pages in order. No code fence "
    "around the whole answer, no notes, no commentary, nothing that is not on the page."
)

GENERIC_MARKDOWN_USER_PROMPT = "Convert these page images to Markdown."
