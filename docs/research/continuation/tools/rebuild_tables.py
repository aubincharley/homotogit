#!/usr/bin/env python3
"""Rebuild EXP-011 tables from the preserved final JSON. No training or network.

Every file is read and written as UTF-8 explicitly: the default encoding is
cp1252 on the Windows machine this repository is maintained from, which
cannot represent the arrows and accents these tables contain.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def paired(configs, a, b):
    av, bv = configs[a], configs[b]
    aa = dict(zip(av['seeds'], av['acc_per_seed']))
    bb = dict(zip(bv['seeds'], bv['acc_per_seed']))
    ac = dict(zip(av['seeds'], av['ce_per_seed']))
    bc = dict(zip(bv['seeds'], bv['ce_per_seed']))
    assert set(aa) == set(bb), (a, b, 'seed mismatch')
    seeds = sorted(aa)
    diffs = [100 * (aa[s] - bb[s]) for s in seeds]
    ces = [ac[s] - bc[s] for s in seeds]
    return dict(a=a, b=b, seeds=seeds, accuracy_unit='percentage_points',
                differences_pp=diffs, mean_pp=st.mean(diffs), sd_pp=st.stdev(diffs),
                ce_mean=st.mean(ces), ce_sd=st.stdev(ces),
                all_positive=all(v > 0 for v in diffs),
                all_negative=all(v < 0 for v in diffs))


def label(v):
    r = {'R32': '32 constant', 'Rprog': '16→24→32',
         'Rgentle': '24→32', 'Rreverse': '24→16→32'}[v['resolution']]
    g = {'Gnone': 'sans filtre', 'Gplateau': 'Gaussian paliers',
         'Ggeo': 'Gaussian géométrique', 'Gmix': 'mélange identité/Gaussian'}[v['gaussian']]
    red = {'input_bilinear': 'bilinéaire AA entrée', 'input_max': 'max entrée',
           'stem_bilinear': 'bilinéaire AA après stem', 'stem_max': 'max après stem'}[v['reduction']]
    sites = '7 sites' if v['mask'] == 'early7' else '19 sites'
    if v['resolution'] == 'R32' and v['gaussian'] == 'Gnone':
        return 'Témoin : 32 constant, aucun filtre'
    return f'{r} ; {g} ; {red}' + (f' ; {sites}' if v['gaussian'] != 'Gnone' else '')


def check_numeric(d):
    configs = d['configurations']
    assert len(configs) == 21
    assert d['n_cells_expected'] == d['n_cells_completed'] == 63
    assert not d['missing_cells'] and d['duplicate_cells'] == 0
    for key, v in configs.items():
        assert key == v['id']
        assert v['n_seeds'] == 3 and v['seeds'] == [0, 1, 2]
        for raw, mean, sd in [('acc_per_seed', 'acc_mean', 'acc_sd'),
                              ('ce_per_seed', 'ce_mean', 'ce_sd')]:
            assert len(v[raw]) == 3 and all(math.isfinite(x) for x in v[raw])
            assert math.isclose(st.mean(v[raw]), v[mean], abs_tol=1e-12)
            assert math.isclose(st.stdev(v[raw]), v[sd], abs_tol=1e-12)
    for name, c in d['comparisons'].items():
        p = paired(configs, c['a'], c['b'])
        assert math.isclose(p['mean_pp']/100, c['d_acc_mean'], abs_tol=1e-12), name
        assert math.isclose(p['sd_pp']/100, c['d_acc_sd'], abs_tol=1e-12), name
        assert math.isclose(p['ce_mean'], c['d_ce_mean'], abs_tol=1e-12), name
    assert math.isclose(sum(v['wall_mean']*v['n_seeds'] for v in configs.values()),
                        d['cumulative_gpu_seconds'], abs_tol=1e-7)
    return dict(configurations=21, cells=63, source_comparisons=len(d['comparisons']),
                all_means_and_sample_sds_verified=True,
                cumulative_gpu_seconds=d['cumulative_gpu_seconds'])


def main():
    source = ROOT/'sources/campaign_results.json'
    d = json.loads(source.read_text(encoding="utf-8"))
    audit = check_numeric(d)
    configs = d['configurations']
    ids = {key: f'C{i:02d}' for i, key in enumerate(configs, 1)}
    ordered = sorted(configs, key=lambda key: -configs[key]['acc_mean'])
    out = ROOT/'data'
    out.mkdir(exist_ok=True)
    lines = ['---', 'id: DATA-CAMPAIGN-TABLES', 'schema_version: 1',
             'updated_at: 2026-09-09', 'status: generated_from_source', '---', '',
             '# EXP-011 — Tous les résultats et contrastes', '',
             '[Fiche de campagne](../experiments/EXP-011_full_grid.md) · '
             '[Source numérique](../sources/campaign_results.json)', '',
             'Généré par `tools/rebuild_tables.py`. Les codes C01…C21 sont des repères '
             'documentaires stables dans cet export, pas de nouveaux IDs de runs. '
             'Moyenne et SD échantillonnale sur les graines 0/1/2 ; accuracy en %, '
             'différences en points. Tous les résultats sont finaux à epoch 30.', '',
             '## Correspondance des identifiants', '',
             '| Repère | Identifiant original | Description |', '|---|---|---|']
    for key, v in configs.items():
        lines.append(f"| {ids[key]} | `{key}` | {label(v)} |")
    lines += ['', '## Classement descriptif complet', '',
              '| Repère | Accuracy moyenne ± SD (%) | CE test moyenne ± SD | CE sonde train | Temps total moyen (min) | Pic (MiB) |',
              '|---|---:|---:|---:|---:|---:|']
    for key in ordered:
        v=configs[key]
        lines.append(f"| {ids[key]} | {100*v['acc_mean']:.3f} ± {100*v['acc_sd']:.3f} | "
                     f"{v['ce_mean']:.5f} ± {v['ce_sd']:.5f} | {v['probe_ce_mean']:.5f} | "
                     f"{v['wall_mean']/60:.3f} | {v['peak_mem_mib']:.1f} |")
    lines += ['', '## Valeurs par graine', '',
              '| Repère | Acc seed0 (%) | Acc seed1 (%) | Acc seed2 (%) | CE seed0 | CE seed1 | CE seed2 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for key in ordered:
        v=configs[key]
        cols=[ids[key]]+[f'{a*100:.2f}' for a in v['acc_per_seed']]+[f'{c:.6f}' for c in v['ce_per_seed']]
        lines.append('| '+' | '.join(cols)+' |')
    comparisons=[]
    for name,c in d['comparisons'].items():
        comparisons.append(dict(name=name, origin='source_comparison', **paired(configs,c['a'],c['b'])))
    def key(r,g,red='input_bilinear',mask='all19'):
        return '__'.join((r,g,red,mask))
    extras=[
        ('Combo moins plain', key('Rprog','Gplateau'), key('R32','Gnone')),
        ('Combo moins max stem sans filtre', key('Rprog','Gplateau'), key('Rprog','Gnone','stem_max')),
        ('Gmix moins absence de filtre à Rprog', key('Rprog','Gmix'), key('Rprog','Gnone')),
        ('Paliers moins géométrique à Rprog', key('Rprog','Gplateau'), key('Rprog','Ggeo')),
    ]
    for g in ['Gnone','Gplateau']:
        extras.extend([
            (f'Max stem moins bilinéaire stem, {g}',key('Rprog',g,'stem_max'),key('Rprog',g,'stem_bilinear')),
            (f'Max stem moins max entrée, {g}',key('Rprog',g,'stem_max'),key('Rprog',g,'input_max')),
        ])
    for red in ['input_max','stem_max','stem_bilinear']:
        extras.append((f'Ajout du Gaussian, {red}',key('Rprog','Gplateau',red),key('Rprog','Gnone',red)))
    for name,a,b in extras:
        comparisons.append(dict(name=name,origin='handoff_additional_descriptive',**paired(configs,a,b)))
    lines += ['', '## Contrastes appariés', '',
              'Les 24 premiers sont déjà présents dans le JSON source. Les suivants '
              'sont des recalculs descriptifs ajoutés pour clarifier les questions de la discussion ; '
              'ils ne sont pas rebaptisés analyses préspécifiées.', '',
              '| Contraste | A − B | Seed0 / seed1 / seed2 (points) | Moyenne ± SD (points) | Δ CE moyen |',
              '|---|---|---:|---:|---:|']
    for c in comparisons:
        vals=' / '.join(f'{v:+.3f}' for v in c['differences_pp'])
        origin=' [ajout]' if c['origin'].startswith('handoff') else ''
        lines.append(f"| {c['name']}{origin} | {ids[c['a']]} − {ids[c['b']]} | {vals} | "
                     f"{c['mean_pp']:+.3f} ± {c['sd_pp']:.3f} | {c['ce_mean']:+.5f} |")
    lines += ['', '## Limites de ce fichier', '',
              'Le tri ne constitue pas une validation indépendante du meilleur réglage. '
              'Le temps est celui mesuré par run, évaluations incluses, et non une extrapolation de FLOPs. '
              'Le JSON ne contient pas les trajectoires complètes ni le temps individuel de chaque seed : '
              'on ne les reconstruit pas à partir des moyennes.', '']
    (out/'campaign_tables.md').write_text('\n'.join(lines), encoding='utf-8')
    with (out/'campaign_configs.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['local_id','config_id','label','seed0_accuracy_fraction','seed1_accuracy_fraction',
                'seed2_accuracy_fraction','accuracy_mean_fraction','accuracy_sample_sd_fraction',
                'test_ce_mean','test_ce_sample_sd','probe_ce_mean','wall_mean_seconds','train_mean_seconds','peak_mib']
        w=csv.writer(f);w.writerow(fields)
        for k in ordered:
            v=configs[k]
            w.writerow([ids[k],k,label(v),*v['acc_per_seed'],v['acc_mean'],v['acc_sd'],v['ce_mean'],
                        v['ce_sd'],v['probe_ce_mean'],v['wall_mean'],v['train_mean'],v['peak_mem_mib']])
    (out/'campaign_contrasts.json').write_text(json.dumps(dict(schema_version=1, experiment_id='EXP-011',
                 source='../sources/campaign_results.json',comparisons=comparisons),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (out/'numeric_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':
    main()
