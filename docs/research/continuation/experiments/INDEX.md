---
id: DOC-EXPERIMENTS
schema_version: 1
updated_at: 2026-09-18
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
| [EXP-012](EXP-012_adaptive_phase0.md) | Signaux pour une résolution adaptative ; grille de frontières | 16 runs (6 appariés à EXP-011 + grille graine 0), 30 époques | Norme du gradient stationnaire : pas de plateau à détecter ; frontières quasi indifférentes, pas fins +1,1 point (une graine) ; BN explique la moitié de l'effondrement cible à r=16 |
| [EXP-013](EXP-013_adaptive_phase1.md) | Pas fins × trois graines, avec/sans Gaussian ; τ en mode train | 15 runs appariés, 30 époques | Rlin12/Rsteps4 : **+1,0 / +0,9 point** sur Rprog sans filtre, positifs sur chaque graine ; gain absorbé par le Gaussian ; 80,1 % pour 68 % du temps du meilleur bras filtré |
| [EXP-014](EXP-014_adaptive_phase2.md) | Contrôleur vivant sur τ ; rampe paire | 6 + 3 runs, 3 graines | Le déclencheur ne tire jamais ; calendrier = plancher, niveau Rprog ; le gain des rampes fines vient du **séjour égal entre échelles**, pas d'un signal ; parité sans effet |
| [EXP-015](EXP-015_stl10_resolution.md) | Résolution progressive sur STL-10 (96×96) | 7 bras × 3 graines, 60–74 époques | Époques grossières 4× moins chères ; à compute égal **+2,6…+2,8 points** (dont ~2/3 d'effet budget), à temps égal ≈ +2,3 ; calendriers indiscernables sur 3 graines |
| [EXP-016](EXP-016_bounded_specialisation.md) | Signal de spécialisation d'échelle g et contrôleur borné | 3 seuils × 3 graines, 30 époques | **35 bascules sur 36 décidées par le signal** ; 80,02 % = rampe fine fixe, +0,9 sur Rprog ; séjour non uniforme ; seuil robuste ×2 ; décalage BN inutilisable |
| [EXP-017](EXP-017_two_sided_specialisation.md) | Réchauffes adaptatives à 24 pendant la phase 32 (signal g vers le bas) | 36 runs valides, jusqu'à 6 graines | **Bat les rampes fines fixes : +0,61 ± 0,35 point, 6/6 graines** (80,52 contre 79,91) ; au moins égal à des réchauffes fixes réglées à la main ; au-dessus du gagnant filtré pour 68 % du temps |

## Différents niveaux de preuve

- **JSON numérique local** : EXP-002, EXP-003, EXP-011, EXP-012, EXP-013, EXP-014, EXP-015, EXP-016 et EXP-017 ; détails de calcul et agrégats accessibles.
- **Rapport local d'exécution** : EXP-000, EXP-001, EXP-010, EXP-011.
- **Compte rendu copié dans la conversation + figures** : EXP-006 à EXP-009.
- **Sources partielles, lectures graphiques** : EXP-004 et une partie d'EXP-005.

Ces niveaux ne disent pas si un résultat est vrai ou faux ; ils disent ce qui a pu être vérifié dans cette passation. Le niveau « rapport » ne doit pas être rebaptisé « code audité ».

## Ce qui n'a pas de fiche de résultat

Gaussian RGB explicite avant réduction, diffusion TV à M étapes, compression à budget de bits, ablation causale BN/initialisation, recalibration BN, mélange de logits aux transitions et nouvelle optimisation de db2 sont des idées ou propositions. Leurs statuts figurent dans [OPEN_QUESTIONS](../OPEN_QUESTIONS.md).
