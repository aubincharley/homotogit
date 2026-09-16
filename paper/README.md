# Training Convolutional Networks with Spatial Continuation

This revision preserves the records audited on 16 September 2026 and improves the paper's presentation. Author order, English voice, method definitions, and official AISTATS style files are preserved. This is a named preprint, not an anonymous submission.

## Documents

- `main.pdf`: the revised paper, 22 pages in total. Main text and all main figures finish on page 7; the appendices occupy pages 9–22.
- `catalogue/exploration_catalogue.pdf`: the complete historical catalogue. Its CSV and source remain alongside it.
- `handover/LAYOUT_REVISION.md`: the presentation changes and validation summary.
- `handover/FINAL_SOURCE_CHECKS.md`: the experiment owner's numerical audit, preserved unchanged.

## Compile

Upload the project contents to Overleaf, select `main.tex`, and use pdfLaTeX. All figures and tables are provided, so Python and the experiment repository are not required to compile the paper.

Locally, from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

All scientific results previously in the three-page companion are now integrated into Appendices B, D and E. There is no separate numerical companion to compile. The main bibliography is also supplied as a BBL. Margins, document font size, and official style files are unchanged. Ragged page bottoms avoid stretching paragraph gaps to justify a column vertically.

## Rebuild figures and tables from the audited records

```sh
python3 -m pip install -r requirements-assets.txt
python3 tools/build_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`build_assets.py` reads the frozen `data/records/` CSVs. Its presentation functions live in `tools/paper_layout.py`, which is called automatically. Rebuilding preserves the revised table layouts, accuracy highlighting, compact sensitivity panels, and four-panel loss surface. No model is loaded or evaluated.

The record extraction stage, `tools/build_records.py`, requires the experiment owner's repository and archived worktrees; it is not necessary for rebuilding from this bundle. `data/records/SOURCES.json` records those sources. The historical exploration figures can be rebuilt in that repository with `tools/make_legacy_assets_with_repo.py`; their audited PDFs are included here. `tools/build_catalogue.py` rebuilds the separate catalogue from the records.

## Reading conventions

- Numerical column headings are centred; numerical entries remain right-aligned. Tiny cross-entropies use decimal notation in nats.
- Bold accuracy means exceed the corresponding Plain mean. Bold does not denote statistical significance.
- Error bars are sample standard deviations across training seeds.
- Paired differences are formed within a setting and seed before aggregation.
- CE on a training probe is distinguished from CE on the complete training set.
- Timing values are observed wall times, including evaluation and checkpoint writes. Hardware energy and peak memory were not measured.
- The compact 3D surfaces join all measured 41-by-41 grid vertices for seed 0, with the same camera, axis limits and colour scale; they are illustrations, not additional replications.

One red French author note marks the outstanding reference to supplementary material and an anonymized code repository. No public URL has been invented.

## Contents

`sections/` and `appendices/` hold the article; `tables/` and `figures/` hold generated assets; `data/` contains the unchanged numerical records and supplied exports; `evidence/`, `provenance/` and `handover/` retain the audit trail. The full training repository, checkpoints and datasets are not included.
