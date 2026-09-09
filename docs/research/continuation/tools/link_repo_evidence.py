#!/usr/bin/env python3
"""Map every experiment record to the repository artifacts that evidence it.

Read-only.  No training, no network, no GPU.  Walks the repository for the
configurations, implementation files, execution outputs and figures that belong
to each ``EXP-xxx`` record, checks that each path exists, records the git commit
that last touched it, and cross-checks the numbers the records quote against the
repository's own raw metrics.

Output: ``data/repo_evidence.json`` plus a readable ``REPO_EVIDENCE.md``.

A path listed here means "this file exists in the repository today".  It does
**not** by itself prove that an earlier run executed that version of the code --
the ``code_state`` field of each entry says how far the evidence goes, and the
history of a file is what ties it to a dated run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

KB = Path(__file__).resolve().parents[1]
REPO = KB.parents[2]

# experiment -> repository artifacts, grouped by role.
# "code_state" records how the current file relates to the historical run.
EVIDENCE = {
    "EXP-000": {
        "code": ["continuation/experiments/exp0.py", "continuation/engine.py",
                 "continuation/transforms/gaussian.py"],
        "config": ["configs/exp0_gaussian.yaml"],
        "results": ["results/exp0_gaussian"],
        "docs": ["docs/experiment0.md", "docs/results.md"],
        "code_state": "gaussian.py a été étendu depuis (repli de réflexion "
                      "explicite, phase 8) ; le chemin input-space de exp0 n'est "
                      "pas affecté par ce repli, qui ne sert qu'aux cartes 4x4.",
    },
    "EXP-001": {
        "code": ["continuation/experiments/exp1.py", "continuation/engine.py"],
        "config": ["configs/exp1_warmstart.yaml"],
        "results": ["results/exp1_gaussian_warmstart"],
        "docs": ["docs/exp1_warmstart.md"],
        "code_state": "branchement full_state_v1 ; code inchangé depuis.",
    },
    "EXP-002": {
        "code": ["continuation/transforms/tv.py", "scripts/tv_previews.py"],
        "config": [],
        "results": ["results/tv_previews"],
        "docs": ["docs/tv_budget.md"],
        "code_state": "tv.py corrigé pendant EXP-002 (signe de l'adjoint D^T) ; "
                      "les apercus livres ont ete produits apres correction.",
    },
    "EXP-003": {
        "code": ["continuation/transforms/wavelet.py", "scripts/wavelet_previews.py",
                 "scripts/wavelet_benchmark.py", "scripts/wavelet_profile.py",
                 "scripts/wavelet_optimized_benchmark.py"],
        "config": [],
        "results": ["results/wavelet_previews"],
        "docs": ["docs/wavelet_shrinkage.md"],
        "code_state": "chemin 'fast' fusionne et bypass s=1 ajoutes apres les "
                      "premiers apercus ; les deux chemins calculent la meme "
                      "operation (verifie par les tests).",
    },
    "EXP-004": {
        "code": ["scripts/kaggle_pilot_continuation.py", "scripts/kaggle_run.py"],
        "config": [],
        "results": ["results/kaggle_outputs/pilot-continuation-20260908-103434",
                    "results/kaggle_outputs/pilot-continuation-20260908-122558"],
        "docs": ["docs/kaggle_cli.md"],
        "code_state": "pilote ancien ; sorties partiellement telechargees, "
                      "summary.json absent pour plusieurs cellules.",
    },
    "EXP-005": {
        "code": ["scripts/continuation_driver.py", "scripts/job_plain_study.py",
                 "scripts/job_gaussian_study.py", "scripts/job_lr_diagnostic.py",
                 "scripts/job_lr_control_002.py",
                 "scripts/audit_gaussian_placement.py",
                 "scripts/verify_grad_accumulation.py",
                 "continuation/models/resnet_gn.py"],
        "config": ["scripts/_study_common.py"],
        "results": ["results/kaggle_outputs/plain-study-20260908-132611",
                    "results/kaggle_outputs/gaussian-study-20260908-132957",
                    "results/kaggle_outputs/lr-diagnostic-20260908-132240",
                    "results/kaggle_outputs/lr-control-002-20260908-140604",
                    "results/gaussian_placement_audit.json",
                    "results/grad_accumulation_check.json",
                    "results/plain_vs_gaussian_paired.json"],
        "docs": [],
        "code_state": "continuation_driver.py a beaucoup evolue depuis (tables "
                      "par paliers, chemins courant/cible, resolution). Le code "
                      "expedie a ce run est conserve dans le _repo/ de chaque "
                      "sortie Kaggle, pas dans l'arbre courant.",
    },
    "EXP-006": {
        "code": ["continuation/models/resnet18_bn.py", "scripts/job_resnet18_gaussian.py",
                 "scripts/continuation_driver.py"],
        "config": [],
        "results": ["results/kaggle_outputs/resnet18-gaussian-20260908-144640",
                    "results/resnet18_plain_vs_gaussian.png"],
        "docs": [],
        "code_state": "le repli de reflexion explicite pour cartes 4x4 a ete "
                      "ajoute pour ce run ; voir gaussian.py.",
    },
    "EXP-007": {
        "code": ["continuation/models/resnet20_bn.py",
                 "scripts/job_resnet20bn_gaussian.py",
                 "scripts/continuation_driver.py"],
        "config": [],
        "results": ["results/kaggle_outputs/resnet20bn-gaussian-20260908-154226",
                    "results/resnet20bn_plain_vs_gaussian.png",
                    "results/resnet20bn_pilot_corrected.png"],
        "docs": [],
        "code_state": "modele inchange depuis ; driver modifie apres coup.",
    },
    "EXP-008": {
        "code": ["scripts/job_fulldata_campaign.py", "scripts/continuation_driver.py"],
        "config": [],
        "results": ["results/kaggle_outputs/fulldata-r20bn-20260908-161221",
                    "results/fulldata_campaign.png"],
        "docs": ["docs/HANDOVER.md"],
        "code_state": "assets partages de ce run (init_seed*.pt, "
                      "shared_indices.npz) reutilises comme etat epingle "
                      "d'EXP-010 et EXP-011.",
    },
    "EXP-009": {
        "code": ["scripts/job_db2_pilot.py", "scripts/verify_db2_operator.py",
                 "continuation/transforms/wavelet.py",
                 "scripts/continuation_driver.py"],
        "config": [],
        "results": ["results/kaggle_outputs/db2-pilot-r20bn-20260908-191741",
                    "results/db2_operator_verification.json",
                    "results/db2_pilot.png"],
        "docs": [],
        "code_state": "table par paliers de s ajoutee au driver pour ce run ; "
                      "is_active traite s=1 comme inactif.",
    },
    "EXP-010": {
        "code": ["scripts/job_progressive_resolution.py",
                 "scripts/verify_progressive_resolution.py",
                 "scripts/plot_progressive_resolution.py",
                 "continuation/pipeline.py", "scripts/continuation_driver.py"],
        "config": [],
        "results": ["results/kaggle_outputs/progres-r20bn-20260909-075846",
                    "results/progressive_resolution.png",
                    "results/progressive_resolution_verification.json"],
        "docs": [],
        "code_state": "le redimensionnement vit dans InputPipeline "
                      "(uint8/255 -> float -> resize -> T_eta -> normalisation) ; "
                      "la verification d'appariement par empreintes est dans le "
                      "job.",
    },
    "EXP-011": {
        "code": ["continuation/campaign_ops.py", "scripts/campaign_driver.py",
                 "scripts/campaign_manifest.py", "scripts/job_campaign.py",
                 "scripts/job_campaign_0.py", "scripts/job_campaign_1.py",
                 "scripts/job_campaign_2.py",
                 "scripts/verify_campaign_ops.py",
                 "scripts/stage_campaign_assets.py",
                 "scripts/analyze_campaign.py", "scripts/make_presentation.py",
                 "scripts/presentation_labels.py"],
        "config": ["results/campaign_manifest_frozen.json"],
        "results": ["results/kaggle_outputs/campaign-j0-20260909-095039",
                    "results/kaggle_outputs/campaign-j1-20260909-095101",
                    "results/kaggle_outputs/campaign-j2-20260909-095123",
                    "results/campaign_results.json",
                    "results/campaign_verification.json",
                    "results/campaign_previews",
                    "results/presentation"],
        "docs": ["docs/HANDOVER.md"],
        "code_state": "code de la campagne ecrit pour ce run et inchange depuis ; "
                      "commit f191fa6 contient les resultats.",
    },
}


def git(*args):
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], text=True,
                             capture_output=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def describe(rel: str) -> dict:
    p = REPO / rel
    entry = {"path": rel, "exists": p.exists(),
             "kind": "dir" if p.is_dir() else ("file" if p.is_file() else None)}
    if p.is_file():
        entry["bytes"] = p.stat().st_size
        entry["last_commit"] = git("log", "-1", "--format=%h %ad", "--date=short",
                                   "--", rel)
    elif p.is_dir():
        entry["n_files"] = sum(1 for _ in p.rglob("*") if _.is_file())
    return entry


def cross_check() -> dict:
    """Compare the snapshot's campaign JSON with the repository's own copy."""
    a = json.loads((KB / "sources" / "campaign_results.json").read_text(
        encoding="utf-8"))
    b_path = REPO / "results" / "campaign_results.json"
    if not b_path.is_file():
        return {"status": "repository copy absent"}
    b = json.loads(b_path.read_text(encoding="utf-8"))
    diffs = []
    for cid, ca in a.get("configurations", {}).items():
        cb = b.get("configurations", {}).get(cid)
        if cb is None:
            diffs.append({"config": cid, "issue": "absent du dépôt"})
            continue
        if abs(ca["acc_mean"] - cb["acc_mean"]) > 1e-12:
            diffs.append({"config": cid, "snapshot": ca["acc_mean"],
                          "repository": cb["acc_mean"]})
    return {"status": "compared", "n_configurations": len(a.get("configurations", {})),
            "n_cells_snapshot": a.get("n_cells_completed"),
            "n_cells_repository": b.get("n_cells_completed"),
            "differences": diffs}


def main():
    report = {"repository_root": str(REPO), "head": git("rev-parse", "--short", "HEAD"),
              "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
              "experiments": {}, "missing": [],
              "campaign_cross_check": cross_check()}
    for exp, groups in EVIDENCE.items():
        entry = {"code_state": groups["code_state"]}
        for role in ("code", "config", "results", "docs"):
            entry[role] = [describe(r) for r in groups[role]]
            for d in entry[role]:
                if not d["exists"]:
                    report["missing"].append({"experiment": exp, "path": d["path"]})
        report["experiments"][exp] = entry

    # the commit EXP-011 reports
    report["reported_commit_f191fa6"] = {
        "resolves": git("cat-file", "-t", "f191fa6") == "commit",
        "subject": git("log", "-1", "--format=%h %ad %s", "--date=short", "f191fa6"),
    }

    (KB / "data" / "repo_evidence.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["---", "id: DOC-REPO-EVIDENCE", "schema_version: 1",
             "updated_at: 2026-09-09", "status: generated", "---", "",
             "# Traces de preuve dans le dépôt (généré)", "",
             "[Accueil](README.md) · [Index des expériences](experiments/INDEX.md)"
             " · [JSON](data/repo_evidence.json)", "",
             "Généré par `tools/link_repo_evidence.py`. **Ne pas éditer à la main.**",
             "Un chemin présent signifie que le fichier existe aujourd'hui ; il ne",
             "prouve pas à lui seul quelle version de ce code a exécuté un run",
             "ancien. La colonne « état du code » précise la portée.", "",
             "Dépôt : `%s`, branche `%s`, HEAD `%s`."
             % (REPO.name, report["branch"], report["head"]), ""]
    for exp, entry in report["experiments"].items():
        lines += ["## %s" % exp, "",
                  "**État du code** : %s" % entry["code_state"], "",
                  "| Rôle | Chemin | Présent | Dernier commit |", "|---|---|---|---|"]
        for role in ("code", "config", "results", "docs"):
            for d in entry[role]:
                lines.append("| %s | `%s` | %s | %s |"
                             % (role, d["path"], "oui" if d["exists"] else "**non**",
                                d.get("last_commit") or ("%d fichiers" % d["n_files"]
                                                         if d.get("n_files") else "—")))
        lines.append("")
    cc = report["campaign_cross_check"]
    lines += ["## Recoupement numérique EXP-011", "",
              "Instantané contre copie du dépôt de `results/campaign_results.json` : "
              "%s configuration(s) comparée(s), %d écart(s). Cellules complétées : "
              "instantané %s, dépôt %s."
              % (cc.get("n_configurations"), len(cc.get("differences", [])),
                 cc.get("n_cells_snapshot"), cc.get("n_cells_repository")), "",
              "Commit `f191fa6` cité par la fiche EXP-011 : %s — %s."
              % ("résolu dans ce dépôt" if report["reported_commit_f191fa6"]["resolves"]
                 else "**non résolu**", report["reported_commit_f191fa6"]["subject"]), ""]
    if report["missing"]:
        lines += ["## Chemins annoncés mais absents", ""]
        for m in report["missing"]:
            lines.append("- %s : `%s`" % (m["experiment"], m["path"]))
        lines.append("")
    (KB / "REPO_EVIDENCE.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"experiments": len(report["experiments"]),
                      "missing_paths": len(report["missing"]),
                      "commit_f191fa6_resolves":
                          report["reported_commit_f191fa6"]["resolves"],
                      "campaign_differences": len(cc.get("differences", [])),
                      "n_cells_snapshot": cc.get("n_cells_snapshot"),
                      "n_cells_repository": cc.get("n_cells_repository")},
                     ensure_ascii=False, indent=2))
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
