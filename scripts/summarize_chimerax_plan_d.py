#!/usr/bin/env python3
"""Export all Plan D fits and stability decisions without selecting new poses."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def tables(summary, residues):
    domains, structures, states = [], [], []
    for row in summary['results']:
        identity = {k: row[k] for k in ('emdb_id', 'state', 'template', 'train_half', 'heldout_half')}
        structure = dict(identity, fit_status=row['fit_status'], coordinate_residue_count=row.get('coordinate_residue_count'),
                         relative_orientation_deg=row.get('relative_orientation_deg'),
                         raw_pc1_coordinate=row.get('raw_pc1_coordinate'),
                         normalized_structural_coordinate=row.get('normalized_structural_coordinate'),
                         boundary_CA_317_318_distance_A=None)
        if row.get('coordinates') and 317 in residues and 318 in residues:
            coords = np.asarray(row['coordinates'])
            structure['boundary_CA_317_318_distance_A'] = float(np.linalg.norm(coords[residues.index(317)] - coords[residues.index(318)]))
        structures.append(structure)
        for name, values in row['domain_metrics'].items():
            scores = values.get('fixed_pose_map_scores', {})
            domains.append(dict(identity, domain=name, fit_status=values.get('fit_status', 'completed'),
                residue_count=values.get('residue_count'), fit_atom_count=values.get('fit_atom_count'),
                search_placements=summary['search_placements'], retained_fit_clusters=len(values.get('search_candidates', [])),
                train_correlation=values.get('train_correlation'),
                heldout_correlation=values.get('heldout_score', {}).get('heldout_correlation'),
                primary_fixed_pose_correlation=scores.get('primary', {}).get('heldout_correlation'),
                focus_fixed_pose_correlation=scores.get('focus', {}).get('heldout_correlation'),
                mask_inside_fraction_at_0_5=values.get('mask_inside_fraction_at_0_5')))
    for state, values in summary['state_stability'].items():
        states.append({'state': state, **values})
    return domains, structures, states


def validate_execution(summary, config):
    """Reject partial, duplicate or mismatched runs before exporting a completion decision."""
    expected = {(e['emdb_id'], e['state'], t, p['train'], p['heldout'])
                for e in config['fit_entries'] for t in config['templates']
                for p in config['train_heldout_pairs']}
    keys = ('emdb_id', 'state', 'template', 'train_half', 'heldout_half')
    actual = [tuple(row[k] for k in keys) for row in summary['results']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Fit result identities are incomplete, duplicated or unexpected')
    if summary['seed'] != config['seed'] or summary['search_placements'] != config['search_placements']:
        raise ValueError('Fit execution does not match the frozen seed/search protocol')
    for row in summary['results']:
        if set(row['domain_metrics']) != set(config['domains']):
            raise ValueError('Missing domain fit attempt')
    if set(summary['state_stability']) != {e['state'] for e in config['fit_entries']}:
        raise ValueError('Missing state stability assessment')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No results for {path.name}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_map_diagnostics(base, summary, config):
    """Maximum-density projections are diagnostic views, not local resolution estimates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from strengthen_maps import read_mrc_array, read_mrc_header

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    fig, axes = plt.subplots(3, 2, figsize=(10, 14), constrained_layout=True)
    for i, entry in enumerate(config['fit_entries']):
        path = base / entry['primary_map']
        header = read_mrc_header(path)
        volume = read_mrc_array(path)
        origin = header['origin']
        step = header['voxel_spacing_angstrom']
        # These acquired maps use XYZ axes and zero grid starts; fail on another layout.
        from strengthen_maps import opener
        import struct
        with opener(path) as handle:
            raw_header = handle.read(1024)
        if struct.unpack('<3i', raw_header[64:76]) != (1, 2, 3) or any(struct.unpack('<3i', raw_header[16:28])):
            raise ValueError('Projection diagnostics require the recorded XYZ, zero-start grids')
        rows = [r for r in summary['results'] if r['emdb_id'] == entry['emdb_id'] and r['fit_status'] == 'completed']
        for j, (axis, dims, labels) in enumerate([(0, (0, 1), ('x', 'y')), (1, (0, 2), ('x', 'z'))]):
            ax = axes[i, j]
            projection = volume.max(axis=axis)
            x, y = dims
            extent = [origin[x], origin[x] + (header['dimensions'][x]-1)*step[x],
                      origin[y], origin[y] + (header['dimensions'][y]-1)*step[y]]
            ax.imshow(projection, origin='lower', extent=extent, cmap='Greys', vmin=0,
                      vmax=float(np.quantile(projection, 0.995)), interpolation='nearest')
            for k, row in enumerate(rows):
                coords = np.asarray(row['coordinates'])
                label = row['template'].split('_')[1].upper() + ', train ' + row['train_half']
                ax.scatter(coords[:, x], coords[:, y], s=3, color=colors[k], alpha=0.6, label=label)
            ax.set_title(entry['state'].capitalize() + ' / ' + entry['emdb_id'])
            ax.set_xlabel(labels[0] + ' (Å)'); ax.set_ylabel(labels[1] + ' (Å)')
            if j == 1:
                ax.legend(fontsize=7, markerscale=2, loc='upper right')
        del volume
    fig.suptitle('Independent domain fits: raw map maximum projections and fitted Cα positions', fontsize=12)
    fig.savefig(base/'plan_d_map_diagnostics.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Render diagnostic raw-map projections; requires map binaries')
    args = parser.parse_args()
    base = args.analysis_dir
    summary = json.loads((base/'chimerax_plan_d/plan_d_fit_summary.json').read_text())
    config = json.loads((base/'chimerax_plan_d_config.json').read_text())
    validate_execution(summary, config)
    with (base/config['window_csv']).open() as handle:
        residues = [int(row['author_resnum']) for row in csv.DictReader(handle)]
    result = tables(summary, residues)
    for name, rows in zip(('domain_scores', 'structure_scores', 'state_stability'), result):
        write_csv(base/f'plan_d_{name}.csv', rows)
    passed = [r['state'] for r in result[2] if r['state_gate_pass']]
    decision = {'runtime_status': 'completed', 'domain_fit_attempts': len(result[0]),
        'structure_fit_attempts': len(result[1]), 'states_passing_prespecified_stability': passed,
        'quantitative_directional_validation': False,
        'reason': 'No state passed the prespecified fitting-stability criteria.' if not passed else 'Stability alone does not establish correct CRBN-local density assignment; review local map support before quantitative interpretation.',
        'claim_boundary': 'Fitted models remain fitted models. Failure of this search protocol does not establish insufficient experimental map resolution or contradict an observed experimental state.'}
    (base/'plan_d_execution_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    acquisition_path = base/'strengthen_maps_summary.json'
    if acquisition_path.exists():
        from strengthen_maps import markdown_report
        acquisition = json.loads(acquisition_path.read_text())
        acquisition.setdefault('acquisition_quality_decision', acquisition['quality_decision'])
        quality = dict(acquisition['quality_decision'])
        quality.update({
            'density_support': 'not-established-by-this-fitting-protocol',
            'crbn_local_density_support': 'not-established-by-this-fitting-protocol',
            'overall_use': 'qualitative-only',
            'chimerax': {'available': True, 'install_status': 'executed',
                         'scope': 'recorded_fit_execution', 'version': summary.get('software_version', 'see execution provenance')},
            'fit_stability_gate': {'quantitative_fit_allowed': False, 'decision': 'qualitative-only',
                                  'states': summary['state_stability'], 'reasons': [decision['reason']]},
            'reasons': [decision['reason']], 'claim_boundary': decision['claim_boundary']})
        acquisition['quality_decision'] = quality
        acquisition['fitting_execution'] = decision
        acquisition_path.write_text(json.dumps(acquisition, indent=1, sort_keys=True)+'\n')
        (base/'strengthen_maps_report.md').write_text(markdown_report(acquisition))

    if args.plot:
        plot_map_diagnostics(base, summary, config)
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
