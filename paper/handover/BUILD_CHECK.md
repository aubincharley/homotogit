# Build check — 14 September 2026

- Figures and tables regenerated with `py paper/tools/make_paper_assets.py` from
  `benchmark-organized@8d353f2` (and `continuation-gaussian-tv_exploration_1@591e125`
  for the ablation). The generator's consistency assertions (index, raw
  `metrics.json`, `continuation_core/methods.py`) passed.
- Compiled with pdfLaTeX and BibTeX through latexmk (MiKTeX), clean rebuild
  (`latexmk -C`, then `latexmk -pdf main.tex`).
- 24 pages, US Letter: two-column main text on pages 1–5 (references end on page 5),
  single-column appendices A–F on pages 6–24, including the complete longtable.
- Log: no undefined references or citations, no multiply defined labels, no overfull
  boxes, no BibTeX warnings. Remaining underfull-box notices are vertical-spacing
  notices on float pages and one ragged line in §6.
- Every page was inspected visually. Figures are vector PDFs within the text width;
  no table or figure overflows.
- `aistats2026.sty` and `fancyhdr.sty` are byte-identical to the supplied pack; the
  `preprint` option is used; author–year citations via `natbib`/`apalike`.
- `checklist_template.tex` is not included (`\includechecklistfalse`).
- The Overleaf ZIP was extracted into an empty directory and compiled there with the
  same result.

This is a compilation, layout and provenance check. It is not a validation of
future transfer, theory or visualization results, nor a submission-readiness check.
