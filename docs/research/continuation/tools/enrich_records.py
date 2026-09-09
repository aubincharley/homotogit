#!/usr/bin/env python3
"""Insert a verified 'traces dans le dépôt' section into each experiment record.

Idempotent: the section is delimited by HTML markers and rewritten in place on
every run, so re-running after new evidence appears updates it rather than
appending a second copy.  Historical protocol text above the marker is never
touched.

Read-only with respect to the raw sources; only the maintained records under
``experiments/`` are edited.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KB = Path(__file__).resolve().parents[1]
REPO = KB.parents[2]
BEGIN = "<!-- REPO-EVIDENCE:BEGIN -->"
END = "<!-- REPO-EVIDENCE:END -->"
UP = "../../../.."           # experiments/ -> repository root

ROLE_FR = {"code": "Implémentation", "config": "Configuration",
           "results": "Sorties d'exécution", "docs": "Documentation du dépôt"}


def section(exp: str, entry: dict) -> str:
    lines = [BEGIN,
             "",
             "## Traces dans le dépôt (généré — ne pas éditer à la main)",
             "",
             "Généré par `tools/enrich_records.py` depuis "
             "[`data/repo_evidence.json`](../data/repo_evidence.json). "
             "Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).",
             "",
             "**État du code par rapport à ce run** : " + entry["code_state"],
             "",
             "| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |",
             "|---|---|---|"]
    for role in ("code", "config", "results", "docs"):
        for d in entry[role]:
            if not d["exists"]:
                continue
            target = "%s/%s" % (UP, d["path"])
            stamp = d.get("last_commit") or (
                "%d fichiers" % d["n_files"] if d.get("n_files") else "—")
            lines.append("| %s | [`%s`](%s) | %s |"
                         % (ROLE_FR[role], d["path"], target, stamp))
    lines += ["",
              "Un chemin listé existe dans le dépôt au moment de la génération. "
              "Cela ne prouve pas seul quelle version du code a exécuté ce run : "
              "pour les runs Kaggle, le code réellement expédié est archivé dans "
              "le sous-dossier `_repo/` de la sortie correspondante.",
              "",
              END]
    return "\n".join(lines)


def main():
    evidence = json.loads((KB / "data" / "repo_evidence.json").read_text(
        encoding="utf-8"))["experiments"]
    registry = json.loads((KB / "data" / "experiments.json").read_text(
        encoding="utf-8"))
    files = {e["experiment_id"]: KB / e["file"] for e in registry["experiments"]}

    touched = []
    for exp, entry in evidence.items():
        path = files.get(exp)
        if path is None or not path.is_file():
            print("no record for %s" % exp)
            continue
        text = path.read_text(encoding="utf-8")
        block = section(exp, entry)
        if BEGIN in text and END in text:
            head, _, rest = text.partition(BEGIN)
            _, _, tail = rest.partition(END)
            new = head + block + tail
        else:
            new = text.rstrip("\n") + "\n\n" + block + "\n"
        if new != text:
            path.write_text(new, encoding="utf-8")
            touched.append(exp)
    print(json.dumps({"records_updated": touched,
                      "records_total": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
