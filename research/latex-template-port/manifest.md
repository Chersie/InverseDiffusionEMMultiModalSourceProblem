# Research Manifest — LaTeX Template Port of combined.md

**Task slug**: `latex-template-port`
**Date**: 2026-06-16
**Goal**: Port `paper/combined.md` (1100-line Russian thesis) into the LaTeX skeleton at `paper/Bachelor-Thesis-Template/` without altering text content.

## Scope of research

This is a mechanical Markdown → LaTeX conversion task that uses the existing template skeleton. No new APIs, libraries, or services are introduced. All LaTeX packages used are already declared in `include/preambule.tex`. Phase-3 research consists of confirming the template's preamble already covers every LaTeX feature required by the conversion.

## Source files consulted

| # | Source | Location | Verified facts taken from it |
|---|---|---|---|
| R1 | LaTeX template entry point | `paper/Bachelor-Thesis-Template/main.tex` | Uses `\documentclass[a4paper, openany, 12pt]{article}`; loads preamble and title page; expects `parts/Chapter*.tex` files via `\input`; runs `\nocite{*}` + `\bibliography{references}` for BibTeX. |
| R2 | Preamble of the template | `paper/Bachelor-Thesis-Template/include/preambule.tex` | Packages available: `amsmath, amsfonts, amssymb, amsthm, mathtools, icomma, graphicx, caption, array, tabularx, tabulary, booktabs, longtable, multirow, makecell, hhline, enumitem, hyperref, fancyhdr, float, gensymb, listings, indentfirst`. Russian locale active via `babel`. Margins set via `geometry`. `\setlength{\parindent}{1.25cm}` and `\linespread{1}` fixed. |
| R3 | Title page template | `paper/Bachelor-Thesis-Template/include/title-page.tex` | Two-column-like layout with `\begin{center}` blocks and `\begin{flushright}` author/advisor block; logo from `images/MIPT_logo.jpg`; uses `\thispagestyle{empty}`. |
| R4 | Annotation file | `paper/Bachelor-Thesis-Template/parts/Annotation.tex` | Uses `abstract` environment from `article` class; centred title, Russian text body, optional English `\textbf{Abstract}` block. |
| R5 | Skeleton chapter files | `paper/Bachelor-Thesis-Template/parts/Chapter0..5.tex` | Each uses `\section{...}` with a `\label{...}` and `\index{...}` line. So the canonical heading depth is `\section` for chapters, implying `\subsection` and `\subsubsection` for nested levels. |
| R6 | README of the template | `paper/Bachelor-Thesis-Template/README.md` | Confirms: documentclass is `article` (only 10/11/12 pt sizes; `\fontsize{14}{16}\selectfont` for 14pt); bibliography style `gost71u.bst` via BibTeX `references.bib`; structure conventions. |
| R7 | combined.md (full content) | `paper/combined.md` | 1100 lines: front matter, abstract, ToC table, notation/abbrev tables, 8-subsection introduction, 7 numbered chapters with up to 7 subsections each, conclusion with 3 sections, 13-entry bibliography. Heavy use of inline `$math$`, display `\[...\]`, markdown tables, bulleted lists, code-like identifiers in backticks, file paths in markdown links, GOST-formatted numbered references. |
| R8 | Master plan | `paper/THESIS_PLAN.md` | Confirms required structural elements (front matter, intro, chapters 1–7, conclusion, references, appendix). Confirms styling principles (Russian only, no anglicisms, references to `proposal.md` for terminology authority). |
| R9 | Proposal | `paper/proposal.md` | Confirms content-authoritative naming of methods (Матрёшка-trick, RankBinPLoss, PhysicsPowerLoss, etc.) — these are the verbatim terms used in combined.md and must be preserved as-is. |

## Conversion rules (verified against R2 preamble)

| Markdown element in combined.md | LaTeX target | Package providing support | Verified |
|---|---|---|---|
| `# H1` (chapter / front-matter section) | `\section*{...}` + `\addcontentsline{toc}{section}{...}` | base `article` class | R5 |
| `## H2` (subsections like 1.1, 2.3) | `\subsection*{...}` + `\addcontentsline{toc}{subsection}{...}` | base `article` class | R5 |
| `### H3` (sub-subsections like 2.5.1) | `\subsubsection*{...}` + `\addcontentsline{toc}{subsubsection}{...}` | base `article` class | R5 |
| `**bold**` | `\textbf{...}` | base | — |
| `*italic*` | `\textit{...}` | base | — |
| Inline `$math$` | unchanged inline `$math$` | `amsmath` | R2 |
| Display `\[...\]` | direct `\[...\]` (unnumbered display math) | `amsmath` | R2 |
| Markdown table `\| a \| b \|` | `\begin{tabular}{|l|l|}` … `\end{tabular}` inside `table[H]` | `array`, `float` | R2 |
| Long table (e.g., §7.2 module mapping) | `longtable` | `longtable` | R2 |
| Bulleted `- item` | `\begin{itemize} \item ... \end{itemize}` | `enumitem` (base also works) | R2 |
| Numbered `1. item` | `\begin{enumerate} \item ... \end{enumerate}` | `enumitem` | R2 |
| Code-fenced or backticked `\`x\`` | `\texttt{x}` | base; `_` inside `\texttt` is rendered literally if escaped as `\_` | R2 |
| Markdown link `[text](path)` where path is repo-relative | render path as `\texttt{path}` (drop URL — paths are not clickable) | — | — |
| `\newpage` literal in source | `\newpage` LaTeX command | base | R1 |
| `&nbsp;` HTML entity | `~` (non-breaking space) | base | — |
| Numeric reference markers `[1, 2]` | literal text `[1, 2]` (manual numbering matches `thebibliography`) | base | — |

## Special-character escaping (verified standard LaTeX)

- `_` outside math → `\_` or wrap entire token in `\texttt{...}` with literal underscore (acceptable inside `\texttt{}`).
- `%` → `\%`.
- `&` outside HTML entity → `\&`.
- `#` → `\#`.
- `\$` for literal dollar (rare in this text).
- Cyrillic letters: handled by `babel` + `T2A` per R2.

## Bibliography strategy

Decision: use a manual `\begin{thebibliography}{99} ... \end{thebibliography}` environment placed in `parts/References.tex`, with each `\bibitem{refN}` containing the exact GOST-formatted entry from combined.md lines 1089–1101. This avoids re-encoding into BibTeX (`gost71u.bst`) and keeps the visible bibliography text verbatim.

In `main.tex`, the auto-bibliography commands `\nocite{*}` and `\bibliography{references}` will be commented out and replaced with `\input{parts/References.tex}`. Numeric in-text references `[1, 2]` are left as plain visible text — they match the visible numbers in the manual bibliography.

## What is intentionally NOT researched

- `gost71u.bst` syntax: not needed because we don't author BibTeX entries.
- Overleaf-specific behaviors: out of scope; user can compile locally with the provided Makefile.
- Hyperref Cyrillic anchors: covered by `\usepackage[unicode, pdftex]{hyperref}` in R2, which is the standard setup.
- Font sizing escalation to 14pt: README (R6) notes the recipe; we keep default 12pt per template defaults.

## Verdict

All required LaTeX features for the conversion are already available in the template's preamble (R2). No further external sources need to be consulted before Phase 4 (implementation). The conversion is a mechanical syntax-mapping task; correctness is verifiable by inspecting compiled PDF (out-of-scope for this session) or by reading the .tex files against combined.md.
