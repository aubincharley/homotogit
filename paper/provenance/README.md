# Provenance

Two levels of evidence are kept apart in this project.

**Records.** `../data/records/` holds one tidy table per measurement family, built by
`tools/build_records.py` from the original experiment records: the `summary.json`/`metrics.json` of
every run in the audited experiment index, the per-run records of the transfer branches, the
optimizer job summaries, the input-probe shards of the diagnostic grid, and the raw evaluation files
of the landscape study. `SOURCES.json` records the source commit of each family, the row count and a
sha256 of every table. The tables in `tables/`, the figures in `figures/` and the prose macros in
`tables/numbers_generated.tex` are derived from these tables alone.

**Supplied exports.** `../data/*.csv` are the exports delivered with the earlier handoff, kept as
received. They are no longer the input of any generated asset; they remain as the material that the
record-level recomputation was checked against.

`generated_assets.json` is the provenance manifest of the historical exploration assets: the
commits, record paths and configuration identifiers used by their generator. It does not contain raw
training records for all 165 configurations; those are recomputed from the per-run summaries into
`data/records/catalogue_cells.csv` and `catalogue_configurations.csv`.

`CLAIM_AUDIT.md` maps the manuscript to its sources and separates recomputation from reported
implementation facts. `REFERENCES_CHECKED.md` identifies the primary literature.
`bundle_sha256.json` records hashes of the delivered project files. The per-item status, including
what remains unresolved, is in `../handover/FINAL_SOURCE_CHECKS.md`.

Snapshots: the manuscript text starts from `manuscript@1bd1fe7`; method definitions are audited
against `continuation-core@ef5564c`; the historical index is `benchmark-organized@8d353f20`; the
landscape material is `visualization@9e9275c` plus the raw outputs produced after that push; the
input-derivative grid is `continuation-core-experiments@4179584`. No git merge, remote write or
training job was performed for this revision.
