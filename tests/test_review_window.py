"""Coverage, contact-policy and output-contract tests for the window analysis."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import review_window as rw


@pytest.fixture
def require_window_data():
    def require(pdb):
        files = [rw.ROOT/"data/crbn_ensemble.ens.npz", rw.ROOT/"data/pca_diffvec.npz",
                 rw.ROOT/"data/crbn_residue_window.csv", Path(rw.sc.legacy.CIF_CACHE)/f"{pdb}.cif.gz"]
        missing = [str(path.relative_to(rw.ROOT)) for path in files if not path.is_file()]
        if missing:
            pytest.skip("Frozen integration input bundle is not staged: "+", ".join(missing))
    return require


def test_observed_window_excludes_tags_and_does_not_impute_missing_sensor_loop(require_window_data):
    require_window_data("8CVP")
    case = rw.load_window("8CVP", True)
    assert len(case["core"]) == 269
    assert len(case["added"]) == 80
    assert len(case["expanded_residues"]) == 349
    np.testing.assert_array_equal(case["expanded_residues"][:269], case["core"])
    np.testing.assert_array_equal(case["expanded_xyz"][:269], case["original_xyz"][:269])
    np.testing.assert_array_equal(case["expanded_xyz"][349:], case["original_xyz"][269:])
    by_residue = {r["residue"]:r for r in case["coverage"]}
    assert all(by_residue[r]["status"] == "observed_added" for r in range(198,221))
    assert all(by_residue[r]["status"] == "unobserved" for r in range(342,358))
    assert len(case["coverage"]) == 442


@pytest.mark.parametrize("pdb", ["8D7X","8D7Y","6H0F","7U8F"])
def test_all_reference_additions_are_mapped_observed_canonical_positions(pdb, require_window_data):
    require_window_data(pdb)
    case = rw.load_window(pdb, True)
    assert len(set(case["expanded_residues"])) == len(case["expanded_residues"])
    assert set(case["core"]).isdisjoint(case["added"])
    assert np.all(np.diff(case["added"]) > 0)
    assert 1 <= min(case["expanded_residues"]) <= max(case["expanded_residues"]) <= 442
    mapped = set()
    for row in case["mapping"]:
        mapped.update(range(int(row["db_align_beg"]),int(row["db_align_end"])+1))
    assert set(case["expanded_residues"]).issubset(mapped)


@pytest.mark.parametrize("suffix", [".csv",".csv.gz"])
def test_table_roundtrip_preserves_absence_and_scientific_notation(tmp_path,suffix):
    path = tmp_path/("table"+suffix)
    rw.write_table(path,[{"status":"present","effect":1e-12,"group_id":"221:CRBN_DDB1"},
                        {"status":"absent","effect":"","group_id":"339:CRBN_DDB1"}])
    rows = rw.read_table(path)
    assert float(rows[0]["effect"]) == 1e-12
    assert rows[1]["effect"] == ""
    assert rows[1]["status"] == "absent"


def test_bad_directions_fail_explicitly():
    xyz=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    for q in (np.zeros(12), np.full(12,np.nan), np.zeros(11)):
        with pytest.raises(ValueError):rw._core_basis(xyz,q)


def test_edge_identity_is_coordinate_order_independent():
    r=np.array([221,198,339]); p=[775,774]
    assert rw.edge_key((0,3),r,p) == "CRBN:221--DDB1:775"
    assert rw.edge_key((1,0),r,p) == rw.edge_key((0,1),r,p)
