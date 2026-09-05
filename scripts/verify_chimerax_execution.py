#!/usr/bin/env python3
"""Audit saved fit coordinates, orientations and state decisions independently."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from summarize_chimerax_plan_d import validate_execution


def rotation(a, b):
    u, _, vt = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
    return u @ np.diag([1, 1, np.linalg.det(u@vt)]) @ vt


def pdb_coords(path, ca_only=False):
    rows = [line for line in path.read_text().splitlines()
            if line.startswith('ATOM') and (not ca_only or line[12:16].strip() == 'CA')]
    return np.asarray([[float(line[i:i+8]) for i in (30, 38, 46)] for line in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    args = parser.parse_args()
    base = args.analysis_dir.resolve()
    fit = base/'chimerax_plan_d'
    summary_path = fit/'plan_d_fit_summary.json'
    summary = json.loads(summary_path.read_text())
    config = json.loads((base/'chimerax_plan_d_config.json').read_text())
    validate_execution(summary, config)
    with (base/config['window_csv']).open() as handle:
        residues = [int(row['author_resnum']) for row in csv.DictReader(handle)]
    # Use the same archived mean and coordinate scale, with independent linear algebra.
    from strengthen_maps import CHIMERAX_PLAN_D_RUNNER
    namespace = {'__name__': 'verification_helpers'}
    exec(compile(CHIMERAX_PLAN_D_RUNNER, '<runner>', 'exec'), namespace)
    reference = namespace['_load_score_reference'](Path(config['repository_root']), config, residues, base=base)
    mean = reference['mean']
    indices = {name: np.array([i for i, r in enumerate(residues) if limits[0] <= r <= limits[1]])
               for name, limits in config['domains'].items()}
    by_state = {}
    max_pdb_error = max_orientation_error = max_coordinate_error = 0.
    completed = 0
    for row in summary['results']:
        if row['fit_status'] != 'completed':
            continue
        completed += 1
        coords = np.asarray(row['coordinates'])
        assert coords.shape == (269, 3) and np.isfinite(coords).all()
        fname = f"{row['emdb_id']}_{row['train_half']}_fit_{row['heldout_half']}_score_{row['template']}_269ca.json"
        saved = json.loads((fit/'coordinates'/fname).read_text())
        np.testing.assert_array_equal(saved['coordinates'], coords)
        r = rotation(coords, mean)
        aligned = (coords-coords.mean(0)) @ r + mean.mean(0)
        rb = rotation(mean[indices['NTD+HB']], aligned[indices['NTD+HB']])
        rt = rotation(mean[indices['TBD']], aligned[indices['TBD']])
        relative = rb.T @ rt
        max_orientation_error = max(max_orientation_error, float(np.max(np.abs(relative-np.asarray(row['relative_orientation_matrix'])))))
        q = float(((aligned-mean).ravel() @ reference['pc1']) / np.sqrt(len(coords)))
        q = (q-reference['closed_mean'])/(reference['open_mean']-reference['closed_mean'])
        max_coordinate_error = max(max_coordinate_error, abs(q-row['normalized_structural_coordinate']))
        by_state.setdefault(row['state'], []).append((relative, q))
        for domain, metrics in row['domain_metrics'].items():
            stem = domain.replace('+', 'plus')
            pose = fit/'poses'/row['emdb_id']/row['train_half']/row['template']/f'{stem}.pdb'
            source = fit/'matched_templates'/f"{row['template']}_{stem}.pdb"
            original = pdb_coords(source)
            stored = pdb_coords(pose)
            matrix = np.asarray(metrics['transform_matrix'])
            predicted = original @ matrix[:, :3].T + matrix[:, 3]
            max_pdb_error = max(max_pdb_error, float(np.max(np.abs(predicted-stored))))
            np.testing.assert_allclose(pdb_coords(pose, True), coords[indices[domain]], atol=0.000501, rtol=0)
            assert np.isfinite(metrics['heldout_score']['heldout_correlation'])
            assert 'primary' in metrics['fixed_pose_map_scores']
    assert max_pdb_error < 0.000501
    assert max_orientation_error < 1e-10 and max_coordinate_error < 1e-10
    checks = {}
    for state, rows in by_state.items():
        angles = [float(Rotation.from_matrix(a[0].T@b[0]).magnitude()*180/np.pi) for a,b in itertools.combinations(rows, 2)]
        angle_range = max(angles)
        q_range = float(np.ptp([r[1] for r in rows]))
        recorded = summary['state_stability'][state]
        assert abs(angle_range-recorded['relative_orientation_range_deg']) < 1e-8
        assert abs(q_range-recorded['normalized_structural_coordinate_range']) < 1e-10
        expected_pass = len(rows)==4 and angle_range<=10 and q_range<=0.1
        assert recorded['state_gate_pass'] == expected_pass
        checks[state] = {'max_pairwise_orientation_deg': angle_range, 'normalized_coordinate_range': q_range, 'pass': expected_pass}
    out = {'status': 'PASS', 'completed_structures': completed,
           'max_PDB_coordinate_rounding_error_A': max_pdb_error,
           'max_orientation_matrix_error': max_orientation_error,
           'max_normalized_coordinate_error': max_coordinate_error,
           'state_recalculation': checks,
           'fit_summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
           'scope': 'Saved artifact consistency and independent post-fit linear algebra; not a new fitting execution or local-density validation'}
    target = base/'plan_d_artifact_verification.json'
    target.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
