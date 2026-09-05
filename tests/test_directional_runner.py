from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_directional_mechanics as workflow


def test_verified_conditions_excludes_stale_or_tampered_outputs(tmp_path, monkeypatch):
    cfg = {'references': ['TEST'], 'cutoffs_A': [15], 'weightings': ['uniform']}
    inputs = {'pdb': 'TEST', 'config_sha256': 'current'}
    monkeypatch.setattr(workflow, 'input_record', lambda *args: inputs)
    monkeypatch.setattr(workflow, 'stage_signature', lambda record, stage: record['config_sha256'] + stage)
    case = tmp_path / 'analysis/mechanics/TEST_15A_uniform'
    case.mkdir(parents=True)
    workflow.write_json(case / 'inputs.json', inputs)
    workflow.write_rows(case / 'models.csv', [{'model': 'isolated', 'S_close': 2.0}])
    workflow.finish(case, 'oldmechanics', time.time())
    accepted, rejected = workflow.verified_conditions(tmp_path, cfg, tmp_path/'config.json', 'mechanics')
    assert accepted == [] and rejected == ['TEST_15A_uniform']
    workflow.finish(case, 'currentmechanics', time.time())
    accepted, rejected = workflow.verified_conditions(tmp_path, cfg, tmp_path/'config.json', 'mechanics')
    assert accepted == [case] and rejected == []
    (case / 'models.csv').write_text('model,S_close\nisolated,999\n')
    assert workflow.verified_conditions(tmp_path, cfg, tmp_path/'config.json', 'mechanics')[0] == []


def test_changed_input_hash_invalidates_complete_outputs(tmp_path, monkeypatch):
    cfg = {'references': ['TEST'], 'cutoffs_A': [15], 'weightings': ['uniform']}
    case = tmp_path / 'analysis/mechanics/TEST_15A_uniform'
    case.mkdir(parents=True)
    workflow.write_json(case/'inputs.json', {'data_sha256': 'old'})
    workflow.finish(case, 'unchanged', time.time())
    monkeypatch.setattr(workflow, 'input_record', lambda *args: {'data_sha256': 'new'})
    monkeypatch.setattr(workflow, 'stage_signature', lambda *args: 'unchanged')
    assert workflow.verified_conditions(tmp_path, cfg, None, 'mechanics')[0] == []


def test_empty_verified_set_cannot_retain_old_claims(tmp_path, monkeypatch):
    cfg = {'references': ['TEST'], 'cutoffs_A': [15], 'weightings': ['uniform']}
    monkeypatch.setattr(workflow, 'verified_conditions', lambda *args: ([], ['TEST_15A_uniform']))
    stale = tmp_path/'analysis/mechanics/claim_gates.json'
    workflow.write_json(stale, {'claim_category': 'internal_selective_preservation'})
    workflow.write_rows(stale.parent/'comparisons_all.csv', [{'effect': 999}])
    workflow.consolidate(tmp_path, cfg)
    assert json.loads(stale.read_text())['claim_category'] == 'incomplete_required_conditions'
    assert not (stale.parent/'comparisons_all.csv').exists()


def test_contact_cli_uses_coordinate_runner_without_pickle(monkeypatch):
    import directional_contacts
    captured = []
    monkeypatch.setattr(workflow, 'main', lambda args: captured.extend(args) or 0)
    assert directional_contacts.main(['--config', 'x.json', '--offline']) == 0
    assert captured == ['--config', 'x.json', '--offline', '--stages', 'contacts']
