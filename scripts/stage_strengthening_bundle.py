#!/usr/bin/env python3
"""Verify and stage a CRBN review data bundle without a sibling checkout.

Checks every manifest entry and rejects traversal, symlinks, extra files and
conflicting existing files. Public code snapshots are verified but never used to
overwrite installed code. Raw cryo-EM maps remain separate acquisitions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or not p.parts:
        raise ValueError(f'Unsafe bundle path: {name}')
    return p

def destination(name, repo, output):
    safe_name(name)
    if name.startswith(('data/','render/')):return repo/name
    if name.startswith('strengthening/'):
        return output/name.removeprefix('strengthening/')
    return None

def write_verified(path,raw):
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError(f'Symlink in staging destination: {parent}')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes()!=raw:
            raise ValueError(f'Existing file conflicts with verified bundle: {path}')
    else:path.write_bytes(raw)

def stage(bundle,repo,output):
    with zipfile.ZipFile(bundle) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP entry')
        for info in z.infolist():
            safe_name(info.filename)
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Symlink in bundle')
        meta=json.loads(z.read('BUNDLE_MANIFEST.json'))
        expected={r['path'] for r in meta['files']}|{'BUNDLE_MANIFEST.json','README.md'}
        if set(names)!=expected:raise ValueError('Manifest/ZIP member mismatch')
        for row in meta['files']:
            raw=z.read(row['path'])
            if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise ValueError(f"Bundle hash mismatch: {row['path']}")
        aliases = meta.get('aliases', [])
        file_names = {row['path'] for row in meta['files']}
        alias_names = set()
        for alias in aliases:
            safe_name(alias['source']); safe_name(alias['path'])
            if (alias['source'] not in file_names or alias['path'] in file_names
                    or alias['path'] in alias_names
                    or destination(alias['source'], repo, output) is None
                    or destination(alias['path'], repo, output) is None):
                raise ValueError('Invalid alias role or duplicate alias')
            if hashlib.sha256(z.read(alias['source'])).hexdigest() != alias['sha256']:
                raise ValueError('Alias hash mismatch')
            alias_names.add(alias['path'])
        for row in meta['files']:
            dest=destination(row['path'],repo,output)
            if dest is not None:write_verified(dest,z.read(row['path']))
        for alias in aliases:
            src=destination(alias['source'],repo,output);dst=destination(alias['path'],repo,output)
            if src is None or dst is None:raise ValueError('Invalid alias role')
            raw=src.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=alias['sha256']:raise ValueError('Alias hash mismatch')
            write_verified(dst,raw)
    controls=output/'data/controls'
    for source in controls.glob('*.cif.gz'):
        write_verified(repo/'data/_controls_cif_cache'/source.name,source.read_bytes())
    return {'manifest_files':len(meta['files']),'aliases':len(meta.get('aliases',[])),
            'public_commit':meta['public_commit'],'output':str(output)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path)
    p.add_argument('--repo-root',type=Path,default=ROOT)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/strengthening')
    a=p.parse_args();print(json.dumps(stage(a.bundle,a.repo_root.resolve(),a.output_dir.resolve()),indent=2))
if __name__=='__main__':main()
