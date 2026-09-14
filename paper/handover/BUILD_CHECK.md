# Build check — 14 September 2026 (restructured appendix)

- Figures, tables and macros regenerated with `py paper/tools/make_paper_assets.py`
  from `benchmark-organized@8d353f2` (ablation at `591e125`); the generator's
  consistency assertions passed.
- `main.tex`: pdfLaTeX + BibTeX via latexmk, clean rebuild. Two-column main text
  with references on pages 1–4, single-column appendices A–C after it. No undefined
  references or citations, no overfull boxes. All pages inspected visually.
- `archive/exploration_archive.tex`: compiled separately (standard article class);
  contains the exhaustive catalogue and campaign-wide figures.
- Style files unchanged; `preprint` option; checklist not included.
- Red notes mark unfinished work: abstract, §4.2 transfer, §5 theory, §6
  visualization, §7 discussion.

This is a compilation and layout check, not a submission-readiness check.
