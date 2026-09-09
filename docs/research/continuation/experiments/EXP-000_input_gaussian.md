---
id: EXP-000
schema_version: 1
updated_at: 2026-09-09
status: completed_reported
evidence: local_report_and_figures
protocol: P-INPUT-GN
---

# EXP-000 — Gaussian fixe sur les images d'entrée

[Index](INDEX.md) · Suite : [EXP-001](EXP-001_input_warmstart.md)

## Question et intervention

Le premier test cherche à savoir comment le niveau de flou des images affecte l'optimisation du problème transformé et le transfert sur les images originales. Il ne s'agit **pas encore d'une continuation** : cinq niveaux fixes sigma=0 / 0,5 / 1 / 2 / 3 pixels, chacun entraîné depuis zéro sur les graines 0/1/2, soit 15 runs.

## Protocole

45k train / 5k validation stratifiée (seed 12345), test officiel intact ; ResNet-20 GroupNorm avec huit canaux par groupe, option A ; batch physique 128, dernier groupe incomplet écarté ; 40 époques = 14 040 updates ; SGD LR max 0,1, momentum 0,9, WD 0,0005, warmup 400 puis cosine global ; sans augmentation ni AMP. Images float filtrées avant normalisation. Support du Gaussian d'entrée adapté à sigma max 3 : récupérer la config pour sa taille exacte, sans le confondre avec le noyau interne ultérieur de rayon 4.

Même budget, mêmes réglages et graines appariées. Une phase visuelle a précédé la grille ; aucun niveau n'a été ajouté ou supprimé en fonction des résultats de la comparaison principale.

## Résultats finaux

Moyenne sur trois graines ± SD échantillonnale ; accuracies en %.

| Sigma | Val originale | Val au flou entraîné | CE train originale | MSE de reconstruction | TV conservée |
|---:|---:|---:|---:|---:|---:|
| 0 | 84,11 ± 0,60 | 84,11 ± 0,60 | 0,0034 | 0 | 1 |
| 0,5 | 82,55 ± 0,06 | 83,55 ± 0,38 | 0,0219 | 0,00040 | 0,8200 |
| 1 | 71,33 ± 2,38 | 80,63 ± 0,36 | 0,7529 | 0,00345 | 0,5568 |
| 2 | 48,32 ± 2,96 | 74,30 ± 0,09 | 2,1384 | 0,00951 | 0,3505 |
| 3 | 38,85 ± 1,86 | 67,41 ± 0,33 | 2,5208 | 0,01445 | 0,2548 |

Les accuracies d'entraînement sur la distribution propre à chaque bras approchent 100 %, avec CE finales transformées entre environ 0,003 et 0,064. L'optimisation transformée n'est pas visiblement accélérée par les niveaux forts.

La validation sur images originales peut culminer bien avant la fin : le rapport donne pour sigma=2 un pic moyen 57,65 % vers l'update moyenne 6 333, puis 48,32 % final ; pour sigma=3, 49,28 % vers 5 000, puis 38,85 %. Dans le même temps, l'évaluation sur images floutées continue de s'améliorer. Les valeurs moyennes d'updates de pic ne sont pas de nouveaux checkpoints interpolés exacts.

## Conclusion et limite

Les modèles se spécialisent dans leur distribution floutée ; le transfert immédiat vers les images originales est fortement dégradé aux niveaux élevés. Cela motive un test de **réadaptation**. On n'a pas encore mesuré si des poids issus d'un préfixe flouté sont utiles après entraînement sur la cible.

Le compte rendu original contient des conclusions trop catégoriques (« ne peut agir que comme régularisation », « sigma>1 ne peut pas servir »). Ces formulations sont corrigées par EXP-001 et [CORRECTIONS](../CORRECTIONS.md). Aucun optimum global ni impossibilité générale n'a été démontré.

## Coût, exécution et reprise

RTX 3050 Laptop 4 Go, PyTorch 2.5.1+cu121, cuDNN déterministe rapporté. Temps typique environ 500 s par run, dont 10–12 s de transformation. Une exécution sigma=0,5 seed1 compte 7 446 s à cause d'une suspension machine : conserver ses métriques, exclure cette durée du débit normal.

Les checkpoints finaux ne conservaient pas tous les états RNG/sampler ; ils ne permettaient pas de reconstituer un branchement exact à 1 500 updates. Dossier d'exécution : `results/exp0_gaussian/`.

## Preuves disponibles et manques

- [Rapport original](../sources/results.md), [prompt original](../sources/prompt_homotopie_images.md), [figure](../sources/exp0_curves.png).
- Métriques brutes par run, configs et code non copiés dans cette base.
- La conclusion est documentée par le rapport ; les chiffres du tableau n'ont pas pu être recalculés depuis les 15 logs originaux.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : gaussian.py a été étendu depuis (repli de réflexion explicite, phase 8) ; le chemin input-space de exp0 n'est pas affecté par ce repli, qui ne sert qu'aux cartes 4x4.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/experiments/exp0.py`](../../../../continuation/experiments/exp0.py) | 86c00bc 2026-09-08 |
| Implémentation | [`continuation/engine.py`](../../../../continuation/engine.py) | 86c00bc 2026-09-08 |
| Implémentation | [`continuation/transforms/gaussian.py`](../../../../continuation/transforms/gaussian.py) | d643f75 2026-09-08 |
| Configuration | [`configs/exp0_gaussian.yaml`](../../../../configs/exp0_gaussian.yaml) | 86c00bc 2026-09-08 |
| Sorties d'exécution | [`results/exp0_gaussian`](../../../../results/exp0_gaussian) | 88 fichiers |
| Documentation du dépôt | [`docs/experiment0.md`](../../../../docs/experiment0.md) | 86c00bc 2026-09-08 |
| Documentation du dépôt | [`docs/results.md`](../../../../docs/results.md) | 86c00bc 2026-09-08 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
