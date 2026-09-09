# Repères de preuves dans la conversation disponible

**Établi le 9 septembre 2026.** Ce fichier est une extraction documentaire produite pour la passation, pas une transcription brute garantie de toute la conversation. Il conserve les éléments dont le rapport complet n'existe que dans les messages copiés par l'utilisateur. Les autres rapports sont copiés séparément dans ce dossier.

## CONV-01 — Audit et ResNet-18

Le contexte importé contient l'audit Gaussian/ResNet et le run `resnet18-gaussian-20260908-144640`. L'agent externe rapporte : 11 173 962 paramètres, 17 sites, 20 BN, trois projections non filtrées, std classifieur 0,0102. Plain : CE sonde 0,0530, CE val 1,5071, accuracy 0,5426, 151 s. Gaussian : 0,0987, 1,1019, 0,6216, 197 s. Sonde T4 110,8 / 175,3 ms par update. Les poids/buffers/batches partagés sont rapportés vérifiés.

L'utilisateur a demandé de garder l'échelle 10k plutôt que de passer immédiatement à une reproduction sur tout le dataset. La restriction du padding réfléchi sur 4×4 a conduit à l'extension d'indices explicitée dans OPERATORS.

## CONV-02 — ResNet-20 BN et durée de continuation

Instruction utilisateur visible : « disons qu'on arrête le filtre à partir de la 1700e update et je te laisse déterminer un rythme raisonnable pour ce qui précède ». Elle remplace le prompt initial d'extinction à 600 dans un run total de 2 400.

Rapport externe `resnet20bn-gaussian-20260908-154226` : plain CE sonde 0,7510, CE val 1,2335, accuracy 0,5536, 119 s, 394 MiB ; Gaussian 0,8887, 1,1075, 0,5962, 174 s, 467 MiB. Modèle 269 722 paramètres, 19 sites, std classifieur 0,0099. Aucun tableau complet des bornes de paliers n'est inclus dans le message.

Le rapport initial mettait en avant les accuracies bypassed à 600 et 1 200, 0,1118 et 0,2582. Le rapport corrigé retrouve `val_acc_filtered` déjà sauvegardée : à 200, 0,3686 contre 0,3106 plain ; à 600, 0,4694 contre 0,4366 ; à 1000, 0,5374 contre 0,5010 ; à 1400, 0,5746 contre 0,5414 ; à 1700, 0,5820 contre 0,5480 ; à 2400, 0,5962 contre 0,5536. L'utilisateur demande de faire du chemin actif la représentation principale.

## CONV-03 — Campagne Gaussian complète

Rapport `fulldata-r20bn-20260908-161221` : neuf runs, 50k/10k, 30 époques, 391 updates/époque, warmup 60, trois graines. Les paliers de trois époques sont 1/0,85/0,70/0,60/0,50/0,40/0,30 ; géométrique 0,9^e ; tous deux bypassés à e=21.

Plain accuracies [0,7510, 0,7526, 0,7585], paliers [0,7827, 0,7878, 0,7819], géométrique [0,7763, 0,7884, 0,7892]. CE moyennes 0,7615 / 0,6259 / 0,6636. Total 5 504 s GPU, environ 46 min écoulées à deux T4. Les dernières métriques sont au checkpoint de 30 époques, mais le test a été suivi pendant les runs.

## CONV-04 — Pilote db2 récent

Rapport `db2-pilot-r20bn-20260908-191741` : db2 et plain frais, avec vérification ultérieure des assets identiques au pilote Gaussian. Torch 2.10.0+cu128 ; RMS `sqrt(mean(d²)+1e-12)` non détachée ; s=0 / 0,25 / 0,50 / 0,70 / 0,85 / 0,95 / 0,99 puis 1 à 1700.

Final plain frais : accuracy 0,5540, CE val 1,2367, CE sonde 0,7561, 89 s. db2 : 0,5634, 1,2007, 1,0106, 3 856 s, pic 4 553 MiB. Sonde update filtrée 2,057–2,101 s ; bypass 0,0336 s. Les erreurs d'identité/adjoint décrites dans EXP-009 proviennent de ce rapport. L'ancien « environ 6 % de coarse-only » est explicitement déclaré non vérifié.

La réponse utilisateur suivante est explicite : « on va plutôt drop cette option en ondelettes ». Le travail se déplace vers résolution et Gaussian ; aucune reprise db2 ultérieure n'est documentée.

## CONV-05 — Résolution, max-pooling et benchmark complet

L'utilisateur demande d'explorer la réduction progressive, puis de consolider CIFAR-10 avec plusieurs graines, des grilles Gaussian/résolution et l'idée de conserver le signal le plus fort par pooling. Le budget initial sous 2h30 est élargi à environ 2h30–3h avec six T4 disponibles. Les comptes sont configurés par l'agent externe ; aucun accès ni secret n'est nécessaire à cette passation.

Les rapports complets sont joints dans `progressive_resolution_report.md` et `full_grid_report.md`. Le résultat numérique final de la grande campagne est `campaign_results.json`.

Après la figure A–E difficile à lire, l'utilisateur demande les baselines, le sens des noms et des pointillés, puis de nouvelles figures lisibles. Plus tard, il dit avoir vu ces nouvelles courbes. Les nouveaux fichiers ne sont pas joints ici.

## CONV-06 — Dernière piste : préfloutage et antialiasing

L'utilisateur réalise que la réduction est seulement au début, sur RGB ou après stem, puis demande si un Gaussian sur les images **avant** réduction a été testé. La question porte sur un nouvel opérateur d'entrée ; elle ne change pas rétroactivement le sens de « résolution + Gaussian » dans le benchmark, où Gaussian est interne.

Il interroge ensuite la linéarité du resize lorsqu'il inclut de l'antialiasing. La base explicite que des poids de réduction différents restent linéaires en intensités, tandis que le max-pooling est non linéaire. Aucune nouvelle campagne de préfloutage ou STL n'est fournie après cette discussion.

## CONV-07 — Demande de passation

L'utilisateur demande un ensemble Markdown solide, hiérarchisé, détaillé, destiné à des agents qui l'intégreront au dépôt et l'enrichiront. Il veut distinguer les statuts pour ne pas induire les agents en erreur dès les prochaines expériences. Il valide ensuite : « OK go vas y et relis bien tout ».

Le présent dossier est la réponse documentaire à cette demande. Les citations courtes ci-dessus sont des repères utilisateur ; les tableaux de rapports sont des transcriptions de leurs nombres, pas des mesures nouvelles.
