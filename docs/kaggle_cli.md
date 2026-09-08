# Kaggle compute — agent quickstart

To run something on Kaggle instead of locally:

```
py scripts/kaggle_run.py <your_script.py> [--gpu]
```

- Script must write outputs to `/kaggle/working/` — that's what comes back.
- Results land in `results/kaggle_outputs/<slug>/` (script filename = slug, unless `--title` given).
- Blocks until done (polls every 20s, 1h default timeout — use `--timeout-seconds` for longer jobs).
- Add `--gpu` for GPU. Quota is limited (~30h/week), don't use it for CPU-only work.
- No shell access remotely — only the one script you push runs, in a fresh container each time.

Auth is already set up (`~/.kaggle/kaggle.json`). Nothing else to configure.
