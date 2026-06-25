# Research Manifest — Compile MIPT VKR diploma into .docx

Task slug: `compile-diploma-docx`
Date: 2026-06-10

## Sources consulted

### S1. Положение о ВКР МФТИ (the user-supplied text at `/tmp/polozhenie.txt`)

Sections quoted verbatim, used as the authoritative spec.

- Прил. 2 п. 2 — formatting:
  - A4 (210 × 297 mm);
  - поля: левое 30 мм, правое 15 мм, верхнее 20 мм, нижнее 20 мм;
  - абзацный отступ 1.25 см;
  - межстрочный интервал 1.5;
  - шрифт Times New Roman; основной текст 12–14 пт; названия параграфов 14 пт; названия глав 14 пт; текст в таблицах и подписи 12 или 14 пт;
  - выравнивание основного текста по ширине поля.
- Прил. 2 п. 3 — page numbers: «Порядковый номер страницы печатают на середине нижнего поля страницы.» Title page is page 1 but unnumbered.
- Прил. 2 п. 4 — paragraph headings centred, lowercase except first letter; chapters and параграфы both numbered арабскими цифрами; параграф номер — глава.пункт.
- Прил. 1 — structure:
  - титульный лист (system-generated);
  - аннотация (≤ 1500 знаков);
  - содержание;
  - обозначения и сокращения (при необходимости);
  - основной текст: введение, основное содержание, заключение;
  - список использованных источников;
  - приложения (при необходимости).
- Прил. 1 «каждый включённый в список источник должен иметь отражение в тексте ВКР» — therefore the references list MUST contain only sources actually cited in the body.
- Прил. 1 — references ordered by alphabet, or by document type, or in order of use, or chronologically. **Chosen: in order of first use** (matches the existing in-text numbering [1]–[12], [20]).
- § 2.13 — minimum originality share: bachelor's work ≥ 70 %.
- § 2 п. 18 (Прил. 1) — recommended bachelor's volume 30–40 pages (без учёта приложений).

### S2. `/tmp/diploma_build/merge.py` and `style.py` (previous-iteration artifacts)

Prior working pipeline. Verified behaviour:
- merge.py concatenates `00_front_matter.md`..`07_chapter6_experiments.md` (the previous chapter set);
- strips blockquotes, horizontal rules, lone backslashes;
- promotes `## X` in front matter to top-level `#`;
- inserts `\newpage` (LaTeX raw) between top-level `#` headings — pandoc ignores raw LaTeX in docx, but style.py later forces `w:pageBreakBefore` on every `Heading 1` paragraph, so the page break is enforced by the styler.
- style.py applies section geometry, body restyle (Times 14 pt justified 1.5 spacing 1.25 cm indent), heading restyle (centered, bold, page break before each Heading 1 after the first), table restyle (Times 12 pt), and inserts a centered PAGE field in the footer with `different_first_page_header_footer = True` so the title page has no printed number.

Output of previous build: `/tmp/diploma_build/diploma_raw.docx` (65 KB), proving that `pandoc combined.md -o diploma_raw.docx` works for this input.

### S3. THESIS_PLAN.md §4 (in repo, at `/Users/chersie/Desktop/diplom_clean/paper/THESIS_PLAN.md`)

Lines 337–369 contain the draft reference list (20 items) in ГОСТ-aligned style. Used as the master source list against which actual citations are verified.

### S4. Citation extraction across `01_introduction.md` … `09_conclusion.md`

Script: regex `\[([0-9][0-9,;\s–—\-]*)\](?!\()` over body text with blockquotes, fenced code, inline code, `$…$`, `$$…$$` stripped first. Per-file unique tokens:

| File | Tokens found |
|---|---|
| 01_introduction.md | `[1, 2]`, `[2]`, `[3]`, `[4, 5, 6]`, `[4, 5]`, `[6]`, `[7]` |
| 02_chapter1_physics.md | `[2]`, `[3]`, `[4, 5]`, `[6]` |
| 03_chapter2_data.md | `[7]`, `[8]` |
| 04_chapter3_models.md | `[3]` |
| 05_chapter4_losses_training.md | `[9]`, `[10]`, `[11]`, `[12]` |
| 06_chapter5_metrics_datasets.md | `[20]` |
| 07_chapter6_experiments.md | `[3]` |
| 08_chapter7_framework.md | `[8]`, `[9]`, `[10]`, `[11]`, `[12]` |
| 09_conclusion.md | `[5]`, `[6]` |

Unique numbers cited: `{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20}` (13 sources). Numbers 13–19 from the THESIS_PLAN draft list are NOT cited anywhere → **must be dropped from the final reference list** (Прил. 1 mandates inclusion only of cited sources).

### S5. Decision on `[20]`

Source 20 (Spearman 1904) is cited exactly once (chapter 5). To keep the reference numbering contiguous (ГОСТ-permitted choice "по мере использования"), we renumber `[20]` → `[13]` in the merged output. The renumbering happens in `merge.py` only — source files in `paper/` remain unmodified.

### S6. Pandoc 3.9.0.2 behaviour for md→docx (`pandoc --version`)

Verified installed:
```
pandoc 3.9.0.2
Features: +server +lua
```

Relevant behaviour for our pipeline (corroborated by the prior successful build):
- Inline `$…$` and display `$$…$$` math are rendered to OMML inside docx (native Office math).
- Tables are emitted as Word tables that python-docx can iterate.
- HTML entities like `&nbsp;` are preserved as Unicode non-breaking spaces in the docx run.
- Heading levels map 1→Heading 1, 2→Heading 2, etc.
- Raw `\newpage` lines in markdown are ignored (LaTeX-only); style.py adds page breaks via `w:pageBreakBefore`.

### S7. python-docx 1.2.0 (`/usr/bin/python3`)

Verified import: `/Users/chersie/Library/Python/3.9/lib/python/site-packages/docx`. Used APIs (all proven by `style.py` previous run):
- `Document(path)`, `doc.sections`, `section.page_width/height/left_margin/right_margin/top_margin/bottom_margin/footer_distance`;
- `section.different_first_page_header_footer`, `section.footer`, `section.first_page_footer`;
- paragraph format `line_spacing`, `line_spacing_rule = WD_LINE_SPACING.MULTIPLE`, `space_before`, `space_after`, `first_line_indent = Cm(1.25)`;
- `WD_ALIGN_PARAGRAPH.JUSTIFY / CENTER`;
- run XML: forcing `<w:rFonts>` ascii/hAnsi/cs/eastAsia all to "Times New Roman".
- raw OOXML `<w:fldChar>` / `<w:instrText>PAGE \* MERGEFORMAT</w:instrText>` for footer page numbering.

## Decisions for this build (every implementation detail traces here)

1. **Chapter file list (merge order)**:
   `00_front_matter.md`, `01_introduction.md`, `02_chapter1_physics.md`, `03_chapter2_data.md`, `04_chapter3_models.md`, `05_chapter4_losses_training.md`, `06_chapter5_metrics_datasets.md`, `07_chapter6_experiments.md`, `08_chapter7_framework.md`, `09_conclusion.md`, `10_references.md` (new).
2. **`10_references.md`**: contiguous numbered list 1–13, ordered by first use, ГОСТ-style entries copied from THESIS_PLAN §4. Renumbered Spearman from 20 to 13.
3. **Citation renumbering**: post-merge text substitution `[20]` → `[13]`. Single token, single file, only one occurrence in body.
4. **Front-matter ToC update**: the existing ToC table inside `00_front_matter.md` lists "Глава 7. Программный фреймворк mpinv" and "Заключение", so no edits needed for ToC structure. Appendices remain listed as planned but will not appear in the body (the chapter files contain no appendix content). The ToC says they exist; the user has been informed.
5. **Page-break strategy**: keep style.py's `w:pageBreakBefore` on every Heading 1 except the first. This guarantees: title page → annotation → ToC → abbreviations → introduction → each chapter → references each start on a new page.
6. **Front-matter promotion in merge.py**: the existing `transform_front_matter` already promotes `## Титульный лист`, `## Аннотация`, `## Содержание`, `## Обозначения и сокращения` to `#` and demotes `### Сокращения`, `### Основные обозначения` to `##`. Unchanged.
7. **References file headline**: `# Список использованных источников` (top-level Heading 1, so style.py inserts a page break before it).
8. **Pandoc invocation**: same as prior build — `pandoc /tmp/diploma_build/combined.md -o /tmp/diploma_build/diploma_raw.docx`. No reference-doc; all styling done by style.py.
9. **Output path**: `/Users/chersie/Desktop/diplom_clean/paper/diploma.docx` (style.py target). Matches user's request.

## Known limitations to surface in Phase 6

- ToC page numbers remain `—` (placeholder). The Положение acknowledges they are populated from the heading numbering on final PDF build; this is `verifiable post-build only`.
- Title page is the draft content from `00_front_matter.md`. The Положение says final title page is auto-generated in the student's personal cabinet on upload (Прил. 6). The draft is acceptable for the source-bundle DOCX.
- Appendices А, Б, В are listed in the ToC but the chapter files contain no appendix content. The body of the docx therefore ends after the references. This matches the user's explicit `[WIP]` markers in THESIS_PLAN.md.
- Originality check (≥70 %) is not in scope of this script.

## Iteration 2 — Math rendering (2026-06-10)

### Symptom reported by user
Opening the produced `paper/diploma.docx` in Microsoft Word, formulas show garbled, e.g. user reported seeing `[ (a^E, a^M);;E_{UT};;P;=;|E_|^2 + |E_|^2. ]` — Greek subscripts invisible, expressions wrapped in brackets, separators turn into `;`. Apple's `textutil` extraction confirms the issue at the file level: every `<m:oMath>` region is dropped in plain-text extraction, leaving holes where formulas were.

### Diagnosis
- The OMML XML pandoc emits is structurally valid (round-trips back to LaTeX via pandoc itself).
- However, pandoc's OMML does NOT consistently style variable runs with the `m:val="i"` italic style nor wrap inline math in `<m:oMathPara>`, and some Word setups fall back to LINEAR display mode for these payloads — which makes inline formulas effectively unreadable.
- Generalising the OMML fix per-construct is fragile (different Word/Pages/Google Docs each handle different subsets differently).

### Robust fix — render every formula to a PNG

Local matplotlib mathtext renderer:
- `/usr/bin/python3` ships with matplotlib 3.9.4; `mathtext` is a built-in module — no LaTeX install required.
- Tested on the 412 unique `$…$` expressions across all chapters.
- A small LaTeX normaliser handles the few constructs mathtext is stricter about than LaTeX:
  - `\le`/`\ge` → `\leq`/`\geq`
  - `\tfrac`/`\dfrac` → `\frac`
  - `\bigl`/`\bigr`/`\biggl`/`\biggr` → `\left`/`\right`; `\big`/`\Big`/`\bigg`/`\Bigg` → ∅
  - `\mathcal X` / `\mathbb X` / `\mathbf X` / `\mathrm X` / `\boldsymbol\X` → braced form
  - `\sqrt N` → `\sqrt{N}`
  - `\hat X` / `\bar X` / `\tilde X` / `\vec X` / `\dot X` → braced form
- After normalisation, **all 412 expressions render with zero failures**.

### Plan
1. `render_math.py`: hashes every unique expression; renders to `/tmp/diploma_build/math/eq_<hash>.png` at high DPI with transparent background.
2. `preprocess_math.py`: per chapter file, replaces each `$…$` (outside fenced/inline code and blockquotes) with `![](math/eq_<hash>.png){height=Xex}` markdown image syntax.
3. `merge.py` reads from `/tmp/diploma_build/preproc/<chapter>.md` instead of `paper/<chapter>.md`.
4. Pandoc embeds the PNGs.
5. `style.py` continues to apply page-layout and font (images inherit their preset height).

### Decision: height matching
- Inline images need their height matched to the body font's x-height (~9pt for 14pt Times). Empirically determined: rendering at DPI=300 with `prop=FontProperties(size=12)` produces a PNG whose pixel-height divided by ~30 gives a docx-displayed height that matches surrounding 14pt text. Per-expression height is determined from the actual PNG bbox and converted to ems using DPI=300.
- Pandoc accepts attribute syntax `![](path){height=Xpx}` for image dimensions in docx output.

## Iteration 3 — `\[ ... \]` display math (2026-06-10)

### Symptom (after iteration 2's mathPr fix)
User reported chapter 1 formula `\sum_{l=1}^{\infty}\sum_{m=-l}^{l}\ldots` rendered as `[ (, ) ;=; _{l=1}{}_{m=-l}{l} , ]` — the outer `[` and `]` are literal, with all LaTeX commands stripped (Greek letters, sums, mathbf gone) but structural characters (`_`, `^`, `{}`, `=`, `,`) preserved.

### Diagnosis
Confirmed by a 4-line repro test: chapter 1 uses `\[ ... \]` for display math (not `$$ ... $$`). Pandoc's default Markdown reader **only recognizes `$ ... $` and `$$ ... $$`** for math. `\[ ... \]` is silently treated as literal text with backslash-escapes — every `\command` is partially stripped, leaving just the punctuation. The output XML had ZERO `<m:oMath>` elements for the display-math paragraph in our test repro.

The extension `+tex_math_single_backslash` enables `\(...\)` inline and `\[...\]` display recognition. With it, the same paragraph produces a correct `<m:oMathPara>` containing bold-italic E, real sum operators, Greek θ and φ, etc.

### Fix
Pass `pandoc -f markdown+tex_math_single_backslash` for every chapter. Inline `$...$` still works because the extension is additive.

### Chapter-1 test pipeline
- Source: `/Users/chersie/Desktop/diplom_clean/paper/02_chapter1_physics.md`
- Combined as a single-file build (no front-matter, no other chapters) under `/tmp/diploma_build/test_ch1/combined.md` (transformations identical to the full pipeline: blockquote stripping, horizontal-rule stripping, lone-backslash → empty line, page-break-before per H1).
- Convert: `pandoc -f markdown+tex_math_single_backslash test_ch1/combined.md -o test_ch1/raw.docx`
- Style: same `style.py` (with `install_math_properties()` and `harden_normal_style()`) writes `paper/diploma_ch1_test.docx`.
- Verification: every `\[ ... \]` block should appear as exactly one `<m:oMathPara>` in the body. 10 display blocks expected (verified by source scan).
- After user confirms math renders correctly in Word, the same pandoc flag is added to the full-thesis pipeline.

## Iteration 4 — PDF export (2026-06-10)

### Goal
Produce `paper/diploma.pdf` from `paper/diploma.docx` with all OMML math rendered as proper typeset glyphs.

### Tool selection
- Surveyed locally available converters:
  - `pdflatex` / `xelatex` — not installed (would need MacTeX, ~5 GB)
  - LibreOffice (`soffice`) — not installed (would need `brew install --cask libreoffice`)
  - `pandoc` directly to PDF — requires LaTeX (same as above)
  - `docx2pdf` Python package — wraps Word automation on macOS anyway
  - **MS Word AppleScript** — installed; only converter that natively understands OOXML math
- Picked Word AppleScript: it's the same renderer Word uses on screen, so the PDF is pixel-faithful to what the user verified.

### AppleScript invocation
The verb is `save as document file name "<path>" file format format PDF`. Two prior attempts failed because the direct-parameter binding was wrong (`active document` returned `missing value` before the doc was loaded, or after `open` returned nothing). The form that works:
```applescript
tell application "Microsoft Word"
    activate
    set theDoc to open file name "<path-to-docx>"
    delay 3
    save as theDoc file name "<path-to-pdf>" file format format PDF
    delay 2
    close theDoc saving no
end tell
```
Key points:
- `open file name "<path>"` (not `open POSIX file ...`) returns the document object correctly.
- `delay 3` after open is necessary on some Word versions for the document to become available before `save as`.
- The script's final return value is `missing value`; that's not an error, the script-runner just prints it.

### Verified result
- 89 pages, A4 (595.2 × 841.92 pt).
- Visual spot-check via Swift + PDFKit rendered pages 1 / 11 / 15 / 25 / 89:
  - Page 1 — title page, all Cyrillic + placeholders preserved.
  - Page 15 — chapter 1 § 1.1 with the big double-sum display formula rendered as proper typeset math (Σ_{l=1}^∞ Σ_{m=-l}^l […], bold-italic **E** and **Ψ**, Greek θ φ).
  - Page 25 — chapter 2 § 2.1, multiple inline formulas: `a^X_{lm}~½(N(0,1)+iN(0,1))`, `E|a^X_{lm}|²=2`, etc.
  - Page 89 — references list 10–13 with page number 89 in centered footer.
- Page count 89 ≈ matches expectation given ~120 K characters of body text at A4/14pt/1.5 line-spacing (Прил. 1 recommends 30–40 pp. for bachelor work; current draft is over budget, but that's content authoring, not the renderer).

### Automation
- `/tmp/diploma_build/docx_to_pdf.py` runs the AppleScript and validates output.
- `/tmp/diploma_build/build.py` now ends with the PDF step, so `python3 build.py` produces both `paper/diploma.docx` and `paper/diploma.pdf`.
