# Archived entry points

Superseded or one-off scripts, kept runnable from the repository root
(`py archive/scripts/<name>.py`). Their repository-root path was changed from
`parents[1]` to `parents[2]`; nothing else was edited. They are not maintained.

| file | why archived | what it produced / replaced by |
|---|---|---|
| `scripts/job_db2_study.py` | prepared 3-seed db2 study, **never launched** | superseded by `scripts/job_db2_pilot.py`; db2 thread closed |
| `scripts/wavelet_coarse_control.py` | visual control for the wavelet previews | figures in `results/wavelet_previews/` |
| `scripts/wavelet_net_microbench.py` | one-off in-network cost benchmark | wavelet cost notes in `docs/wavelet_shrinkage.md` |
| `scripts/wavelet_network_feasibility.py` | one-off feasibility timing | same |
| `scripts/plot_campaign.py` | first campaign figures | `scripts/make_presentation.py` (figures 01–13) |
| `scripts/preview_reductions.py` | input bilinear vs max previews | `results/campaign_previews/` |
| `docs/RESUME_UNIFIED.md` | resume note for the unified download, now done | `experiments/index.json`, `docs/BENCHMARK_TABLE.md`; its pairing and offset claims are corrected in `docs/AUDIT.md` (A1, A2) |
