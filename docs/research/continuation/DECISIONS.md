---
id: DOC-DECISIONS
schema_version: 1
updated_at: 2026-09-11
status: decision_log
---

# Journal des décisions et changements de direction

[Accueil](README.md) · [État actuel](CURRENT_STATE.md)

L'ordre décrit l'évolution scientifique. Les timestamps exacts de certains premiers messages ne sont pas disponibles ; les identifiants ne prétendent pas restituer une horloge complète. « Choix utilisateur » et « recommandation d'analyse » sont séparés.

| ID | Étape | Nature | Décision / raison | Statut actuel |
|---|---|---|---|---|
| D-01 | Départ | Protocole adopté | Construire une famille de simplifications d'entrée, commencer par Gaussian fixe, CIFAR-10 et petit CNN reproductible | Expérience achevée, conservée comme historique |
| D-02 | Après EXP-000 | Protocole adopté | Tester warm start et étape intermédiaire à budget total égal, sans rejouer toute la grille fixe | EXP-001 achevée |
| D-03 | Projections TV | Spécification utilisateur éditée | TV globale entre canaux, moyennes séparées, sans variance conservée ; fidélités L² et homogène Ḣ⁻¹ ; CPU et aperçus seulement | Spécification de référence toujours valable pour ces opérateurs |
| D-04 | Coût des projections | Orientation exploratoire | Essayer un seuillage ondelette explicite, comparer des familles et mesurer le coût | Aperçus puis pilotes réalisés |
| D-05 | Échec Gaussian interne GN | Audit + protocole adopté | Vérifier placement, LR et différences à CBS avant d'interpréter le résultat comme réfutation | Audit partiel documenté ; contrôles bruts à récupérer |
| D-06 | Changement de réseau | Choix utilisateur | Garder 10k images et l'échelle pilote ; modifier architecture/initialisation au lieu de passer immédiatement à 50k | Pilote ResNet-18 BN achevé |
| D-07 | Padding 4×4 | Correction technique adoptée | Utiliser réflexion répétée explicite, conserver support et tous les sites | Fallback conservé dans les essais de résolution |
| D-08 | Après gain ResNet-18 | Proposition remplacée | Campagne ResNet-18 200 époques, quatre bras | Rejetée pour coût ; ne pas lancer |
| D-09 | Réduction du plan | Proposition remplacée | Campagne ResNet-18 30 époques, batch 32×4 | Remplacée par retour au petit ResNet-20 |
| D-10 | Retour ResNet-20 | Choix utilisateur | BN + bonne initialisation, 10k, une graine, 2 400 updates | EXP-007 achevée |
| D-11 | Durée du filtre pilote | Correction utilisateur explicite | Refus d'extinction à 600 ; garder le filtre jusqu'à k=1700, paliers raisonnables puis 700 updates sans filtre | Calendrier pilote exécuté ; ne pas restaurer l'ancien brouillon |
| D-12 | Lecture des courbes | Correction utilisateur adoptée | Chemin actif/courant en principal ; bypassed comme diagnostic | Règle de présentation actuelle |
| D-13 | Confirmation Gaussian | Choix utilisateur | Passer au train complet, trois graines, plusieurs trajectoires gaussiennes, ResNet-20 | EXP-008 achevée |
| D-14 | Reprise db2 | Protocole adopté | Un pilote apparié avant une grande campagne ; RMS différentiée, même opérateur | EXP-009 achevée |
| D-15 | Après coût db2 | **Choix utilisateur** | Abandonner la piste ondelette, explorer d'autres idées depuis le succès Gaussian | **Actif : ne pas relancer db2 spontanément** |
| D-16 | Résolution | Choix utilisateur | Tester réduction réelle progressive seule et avec Gaussian interne | EXP-010 achevée |
| D-17 | Après pilote résolution | Choix utilisateur | Consolider CIFAR-10 avant STL-10 ; explorer grilles et pooling | EXP-011 couvre cette exploration |
| D-18 | Extension de budget | Choix utilisateur | Grande campagne, graines 0/1/2, croisements, six T4, cible 2h30–3h écoulées | Autorisation exécutée, pas budget permanent pour toute suite |
| D-19 | Résultats de la grille | Demande utilisateur | Refaire des figures lisibles sans nouveaux runs | Nouvelles figures indiquées comme vues, artifacts absents ici |
| D-20 | Lecture de la grille | Recommandation d'analyse | Déprioriser Gmix et early7 dans leurs réglages ; conserver combo bilinéaire+Gaussian et stem_max sans Gaussian | Candidates à discuter, pas nouveaux entraînements autorisés |
| D-21 | Dernière discussion opérateurs | Idée discutée | Gaussian explicite avant réduction RGB ; clarifier AA et linéarité | **Non exécuté**, protocole à arrêter |
| D-22 | Passation actuelle | Demande utilisateur | Base Markdown détaillée, sourcée, maintenable, à intégrer/enrichir par l'agent du dépôt | Livraison présente |
| D-24 | Question utilisateur sur l'anti-aliasing | Protocole adopté | Tester placement, masques, sigma constant et BlurPool avant d'interpréter les +3 points comme une continuation | [EXP-012](experiments/EXP-012_aa_ablation.md) achevée |
| D-25 | Assets d'EXP-011 irrécupérables | Contrainte subie, choix documenté | Générer un jeu épinglé neuf et relancer les témoins dans le même lot plutôt que citer les anciens chiffres | Appliqué ; voir [INTEGRATION](INTEGRATION.md) C-43 |
| D-26 | Six meilleurs bras dans 0,47 point à une graine | Protocole adopté | Rejouer les quatre premiers aux graines 0/1/2 avant de conclure | Fait ; a renversé la lecture, voir [CORRECTIONS](CORRECTIONS.md) C-36 |
| D-27 | A priori de profondeur sur sigma | Piste explorée puis écartée | Quatre profils × deux architectures × trois graines ; le profil plat gagne | Écartée sur données, pas sur opinion |

## Règle de prolongation

Ajouter une nouvelle entrée pour une décision future. Si elle en remplace une ancienne, citer son ID et mettre à jour CURRENT_STATE. Ne pas effacer le motif qui avait conduit à la décision ancienne ; ne pas transformer une recommandation de l'assistant en décision prise par l'utilisateur.

Les autorisations d'entraînement étaient rattachées à des campagnes précises. Elles ne sont pas transférées automatiquement à un nouveau dataset, à un nouveau sweep ou à la mise à jour de cette mémoire.
