"""The review bundle must be complete and safe before any data are staged."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec=importlib.util.spec_from_file_location('stage_bundle',Path(__file__).parents[1]/'scripts/stage_strengthening_bundle.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)

def bundle(path, bad_hash=False, extra=None):
    raw=b'coordinates\n';name='data/example.csv'
    meta={'public_commit':'123','files':[{'path':name,'bytes':len(raw),'sha256':'bad' if bad_hash else hashlib.sha256(raw).hexdigest()}],
          'aliases':[{'path':'strengthening/data/structures/example.csv','source':name,'sha256':hashlib.sha256(raw).hexdigest()}]}
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(name,raw);z.writestr('README.md','guide');z.writestr('BUNDLE_MANIFEST.json',json.dumps(meta))
        if extra:z.writestr(extra,b'bad')

def test_verified_alias_reuses_identical_data(tmp_path):
    z=tmp_path/'bundle.zip';bundle(z)
    result=stage.stage(z,tmp_path/'repo',tmp_path/'out')
    assert result['aliases']==1
    assert (tmp_path/'repo/data/example.csv').read_bytes()==(tmp_path/'out/data/structures/example.csv').read_bytes()

@pytest.mark.parametrize('extra',['../escape','unexpected.txt'])
def test_extra_or_traversal_rejected_before_write(tmp_path,extra):
    z=tmp_path/'bundle.zip';bundle(z,extra=extra)
    with pytest.raises(ValueError):stage.stage(z,tmp_path/'repo',tmp_path/'out')
    assert not (tmp_path/'repo').exists()

def test_hash_rejected_before_write(tmp_path):
    z=tmp_path/'bundle.zip';bundle(z,bad_hash=True)
    with pytest.raises(ValueError,match='hash mismatch'):stage.stage(z,tmp_path/'repo',tmp_path/'out')
    assert not (tmp_path/'repo').exists()

def test_destination_parent_symlink_cannot_write_outside_repo(tmp_path):
    z=tmp_path/'bundle.zip';bundle(z)
    repo=tmp_path/'repo';repo.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (repo/'data').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink in staging destination'):
        stage.stage(z,repo,tmp_path/'out')
    assert list(outside.iterdir())==[]
