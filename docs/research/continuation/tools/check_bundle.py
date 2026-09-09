#!/usr/bin/env python3
"""Read-only checks for this documentation snapshot. No training or network.

Reads every text file as UTF-8 explicitly (the maintaining machine defaults
to cp1252) and tolerates generated documents that live beside the editorial
ones.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote
from rebuild_tables import check_numeric

ROOT = Path(__file__).resolve().parents[1]


def anchors(text):
    # Common GitHub heading anchors, plus explicit HTML anchors if used.
    result = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', text))
    seen = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        heading = re.sub(r'[`*_]', '', heading).lower()
        slug = ''.join(c for c in heading if c.isalnum() or c in ' _-')
        slug = slug.replace(' ', '-')
        n=seen.get(slug,0);seen[slug]=n+1
        result.add(slug if n==0 else f'{slug}-{n}')
    return result


def main():
    errors=[]
    manifest=json.loads((ROOT/'sources/manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        p=ROOT/'sources'/entry['path']
        if not p.is_file():
            errors.append(f'Missing source {p.name}');continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            errors.append(f'Changed source bytes: {p.name}')
    docs=[p for p in ROOT.rglob('*.md') if 'sources' not in p.relative_to(ROOT).parts]
    ids={}
    link_count=0
    for p in docs:
        txt=p.read_text(encoding='utf-8')
        fm=re.match(r'---\n(.*?)\n---\n',txt,re.S)
        if not fm:
            errors.append(f'Missing frontmatter: {p.relative_to(ROOT)}');continue
        m=re.search(r'^id:\s*(\S+)',fm.group(1),re.M)
        if not m:errors.append(f'Missing id: {p.name}')
        elif m[1] in ids:errors.append(f'Duplicate id: {m[1]}')
        else:ids[m[1]]=p
        # Fenced examples are not rendered hyperlinks.
        clean=re.sub(r'^```.*?^```\s*$', '', txt, flags=re.M|re.S)
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)',clean):
            if re.match(r'^[a-zA-Z]+:',target):continue
            link_count+=1
            path,sep,frag=target.partition('#')
            q=(p.parent/unquote(path)).resolve() if path else p
            # A link may target a file or a directory of run outputs; both are
            # navigable in a repository browser.  Only a path that exists as
            # neither is broken.
            if not q.exists():
                errors.append(f'{p.name}: broken path {target}');continue
            if q.is_dir():
                if sep:
                    errors.append(f'{p.name}: anchor on a directory {target}')
                continue
            if sep and q.suffix=='.md' and unquote(frag) not in anchors(q.read_text(encoding='utf-8')):
                errors.append(f'{p.name}: unknown anchor {target}')
    registry=json.loads((ROOT/'data/experiments.json').read_text(encoding='utf-8'))
    assert len(registry['experiments'])==12
    for e in registry['experiments']:
        if e['experiment_id'] not in ids:errors.append(f'Unresolved experiment {e["experiment_id"]}')
        elif ids[e['experiment_id']]!=(ROOT/e['file']):errors.append(f'Wrong experiment file {e["experiment_id"]}')
    numeric=check_numeric(json.loads((ROOT/'sources/campaign_results.json').read_text(encoding='utf-8')))
    print(json.dumps(dict(editorial_markdown_files=len(docs),source_files=len(manifest['files']),
                          local_links_checked=link_count,experiment_records=len(registry['experiments']),
                          numeric=numeric,errors=errors),ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':
    main()
