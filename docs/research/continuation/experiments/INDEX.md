---
id: DOC-EXPERIMENTS
schema_version: 1
updated_at: 2026-09-09
status: index
---

# Registre et chronologie des expériences

[Accueil](../README.md) · [État actuel](../CURRENT_STATE.md) · [Registre machine](../data/experiments.json)

Les IDs sont propres à cette base. Les campagnes antérieures aux runs datés du 8 septembre n'ont pas toutes un timestamp exact disponible ; leur ordre décrit la progression du projet, pas une preuve d'heure de lancement.

| ID / fiche | Question | Réalisation | Enseignement principal |
|---|---|---|---|
| [EXP-000](EXP-000_input_gaussian.md) | Gaussian fixe sur RGB | 5 niveaux × 3 graines, 40 époques | Le flou fort se transfère mal aux images originales ; pas d'avantage d'optimisation établi |
| [EXP-001](EXP-001_input_warmstart.md) | Warm start et étape intermédiaire sur RGB | 3 branches × 3 graines, témoins réutilisés, préfixes partagés | Adaptation rapide, mais pas de gain final à budget global égal |
| [EXP-002](EXP-002_tv_previews.md) | Projections TV L² / homogène H⁻¹ | 10 images × 6 niveaux × 2 fidélités, CPU | Simplification visible, contraste retenu, coût élevé et non-convergence partielle |
| [EXP-003](EXP-003_wavelet_previews.md) | Seuillage ondelette non décimé | 4 familles, aperçus et microbenchmarks | Plus rapide que la projection TV, beaucoup plus cher que Gaussian |
| [EXP-004](EXP-004_early_internal_pilot.md) | Premier pilote interne plain/Gaussian/db2 | 5k, 600 updates, une graine ; traces partielles | Témoin faible ; ne suffit pas à valider les filtres |
| [EXP-005](EXP-005_gn_audit.md) | Gaussian interne GN et audit | 10k, 1 200 updates, 3 graines + contrôle LR mentionné | Gaussian environ −3,45 points ; placement correct, autres différences majeures |
| [EXP-006](EXP-006_resnet18_bn.md) | Architecture et initialisation proches de CBS | 10k, 1 200 updates, une paire seed0 | ResNet-18 BN : +7,90 points |
| [EXP-007](EXP-007_resnet20_bn.md) | Retour au petit réseau corrigé | 10k, 2 400 updates, une paire seed0 | +4,26 points ; correction majeure de la lecture active/bypassed |
| [EXP-008](EXP-008_full_gaussian.md) | Confirmation Gaussian sur train complet | 3 bras × 3 graines, 30 époques | Environ +3 points, deux calendriers positifs |
| [EXP-009](EXP-009_db2_bn.md) | Pilote db2 avec BN et bon contrôle | 10k, 2 400 updates, db2 + plain frais | +0,94 point sur une graine, coût ~64 min ; abandon utilisateur |
| [EXP-010](EXP-010_resolution_pilot.md) | Résolution seule et avec Gaussian | Deux nouveaux bras seed0, 50k, 30 époques | 79,57 / 79,93 %, signal favorable à confirmer |
| [EXP-011](EXP-011_full_grid.md) | Calendriers × résolution + ablations | **21 configs × 3 graines, 63 fraîches** | 80,71 % pour combo principal ; alternative max après stem sans Gaussian |
| [EXP-012](EXP-012_per_layer_sigma.md) | Profils de sigma par couche + contrôleur prédicteur-correcteur | 5 bras × 3 graines, puis 3 bras × 3 graines | Le sigma **uniforme** reste le meilleur (+3,2 à +3,7 pt) ; aucun profil en profondeur, fixe ou découvert, ne l'améliore ; plancher de bruit ≈ 0,5 pt mesuré |

## Différents niveaux de preuve

- **JSON numérique local** : EXP-002, EXP-003, EXP-011 et EXP-012 ; détails de calcul et agrégats accessibles.
- **Rapport local d'exécution** : EXP-000, EXP-001, EXP-010, EXP-011.
- **Compte rendu copié dans la conversation + figures** : EXP-006 à EXP-009.
- **Sources partielles, lectures graphiques** : EXP-004 et une partie d'EXP-005.

Ces niveaux ne disent pas si un résultat est vrai ou faux ; ils disent ce qui a pu être vérifié dans cette passation. Le niveau « rapport » ne doit pas être rebaptisé « code audité ».

## Ce qui n'a pas de fiche de résultat

STL-10, Gaussian RGB explicite avant réduction, diffusion TV à M étapes, compression à budget de bits, ablation causale BN/initialisation, recalibration BN, mélange de logits aux transitions et nouvelle optimisation de db2 sont des idées ou propositions. Leurs statuts figurent dans [OPEN_QUESTIONS](../OPEN_QUESTIONS.md).
