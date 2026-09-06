"""Checks for acquisition boundaries, fixed observables, and source identities."""
import io
from pathlib import Path
import struct
import sys
import zipfile
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import review_external as ext

requires_frozen_bundle=pytest.mark.skipif(
    not (ext.ROOT/"data/crbn_ensemble.ens.npz").is_file(),
    reason="integration check requires the fixed CRBN data bundle; stage the documented data bundle first")


def test_zip_paths_and_range_validation():
    assert ext.safe_member("archive/path/frame.pdb")
    for name in ("../frame.pdb","/tmp/file","a\\..\\bad"):
        assert not ext.safe_member(name)
    raw=b"0123456789abcdefghijklmnopqrstuvwxyz"
    def fake(url,**kwargs):
        start,end=map(int,kwargs["headers"]["Range"][6:].split("-"))
        return 206,raw[start:end+1],{"Content-Range":f"bytes {start}-{end}/{len(raw)}"},""
    with patch.object(ext,"request",side_effect=fake):
        reader=ext.RemoteZipReader("https://example.invalid/archive",len(raw),block_size=8)
        reader.seek(-11,2)
        assert reader.read()==raw[-11:]
        reader.seek(2)
        assert reader.read(16)==raw[2:18]
    with patch.object(ext,"request",return_value=(200,raw,{},"")):
        with pytest.raises(RuntimeError,match="invalid HTTP range"):
            ext.RemoteZipReader("https://example.invalid",len(raw)).read(1)


def test_xtc_framing_rejects_truncation_and_does_not_decompress():
    header=struct.pack(">iiif",1995,3,7,2.0)+np.eye(3,dtype=">f4").tobytes()+struct.pack(">i",3)
    coordinates=np.arange(9,dtype=">f4").tobytes()
    frame=header+coordinates
    assert list(ext.xtc_records(io.BytesIO(frame*2)))==[frame,frame]
    with pytest.raises(EOFError):list(ext.xtc_records(io.BytesIO(frame[:-1])))
    bad=struct.pack(">i",88)+frame[4:]
    with pytest.raises(ValueError,match="malformed"):list(ext.xtc_records(io.BytesIO(bad)))


@requires_frozen_bundle
def test_frozen_coordinate_and_partner_metrics_are_rigid_motion_invariant():
    window,mean,*_=ext.census.load_reference()
    rng=np.random.default_rng(9)
    ddb1=rng.normal(size=(20,3))*5+mean.mean(0)+np.array([5,8,-9])
    ca=np.vstack([mean,ddb1])
    mapping={"missing_core":[],"core_ca_indices":np.arange(269),
             "crbn_ca_indices":np.arange(269),"ddb1_ca_indices":np.arange(269,289)}
    scorer=ext.FrameScorer(mapping)
    original=scorer.score(ca)
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    rotation[:,0]*=np.linalg.det(rotation)
    moved=ca@rotation.T+np.array([10,-23,14])
    result=scorer.score(moved)
    assert np.isclose(result["closure_coordinate"],ext.census.closure_score(mean)[1],atol=1e-12)
    for key in ("closure_coordinate","NTD_TBD_centroid_distance_A","DDB1_body_translation_A","DDB1_internal_RMSD_A"):
        assert np.isclose(result[key],original[key],atol=1e-10),key
    assert result["DDB1_body_rotation_deg"]<1e-5


@requires_frozen_bundle
def test_periodic_partner_shift_is_removed_before_comparison():
    window,mean,*_=ext.census.load_reference()
    # Consecutive CA positions avoid an artificial disconnected-chain fixture.
    ddb1=np.column_stack([np.arange(20)*3.8,np.zeros(20),np.zeros(20)])+mean.mean(0)
    ca=np.vstack([mean,ddb1])
    mapping={"missing_core":[],"core_ca_indices":np.arange(269),
             "crbn_ca_indices":np.arange(269),"ddb1_ca_indices":np.arange(269,289)}
    box=np.diag([300.,300.,300.])
    scorer=ext.FrameScorer(mapping)
    before=scorer.score(ca,box)
    ca[269:]+=np.array([300,-300,600])
    after=scorer.score(ca,box)
    assert np.isclose(before["closure_coordinate"],after["closure_coordinate"],atol=1e-12)
    assert after["DDB1_body_translation_A"]<1e-10
    assert after["DDB1_internal_RMSD_A"]<1e-10


@requires_frozen_bundle
def test_frame_metrics_are_invariant_to_consistent_ca_reordering():
    _,mean,*_=ext.census.load_reference()
    rng=np.random.default_rng(87)
    ca=np.vstack([mean,rng.normal(size=(12,3))+mean.mean(0)])
    permutation=rng.permutation(len(ca));inverse=np.argsort(permutation)
    original={"missing_core":[],"core_ca_indices":np.arange(269),
              "crbn_ca_indices":np.arange(269),"ddb1_ca_indices":np.arange(269,281)}
    reordered={k:inverse[v] if k.endswith("indices") else v for k,v in original.items()}
    a=ext.FrameScorer(original);b=ext.FrameScorer(reordered)
    a.score(ca);b.score(ca[permutation])
    ca[269:]+=np.array([2.,-1.,3.])
    first=a.score(ca);second=b.score(ca[permutation])
    for key in first:
        if isinstance(first[key],(float,int)):assert np.isclose(first[key],second[key],atol=1e-9),key


@requires_frozen_bundle
def test_core_membership_is_observation_based():
    core=set(map(int,ext.census.read_window()))
    assert 347 not in core and 352 not in core and 50 not in core
    assert {221,222,339,378}<=core


def test_duplicate_trajectory_roles_do_not_create_independent_replicates():
    assert ext.trajectory_role("archive/Simulations/Apo/aligned-10kframes.xtc")[0]=="duplicate_alignment"
    assert ext.trajectory_role("archive/path/simul-chainB-open.xtc")[0]=="duplicate_CRBN_subset"
    assert ext.trajectory_role("archive/path/simul.xtc")[0]=="biased_simulation_coordinate_comparator"


def test_member_stream_uses_standard_zip_crc_and_resumes_byte_ranges():
    content=b"source coordinates\n"*1000
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,"w",compression=zipfile.ZIP_DEFLATED) as z:z.writestr("archive/member.txt",content)
    raw=memory.getvalue()
    class Response(io.BytesIO):
        status=206
        def __init__(self,start,end):
            super().__init__(raw[start:end+1]);self.headers={"Content-Range":f"bytes {start}-{end}/{len(raw)}"}
    def fake_urlopen(req,timeout):
        start,end=map(int,req.headers["Range"][6:].split("-"))
        return Response(start,end)
    def fake_request(url,**kwargs):
        start,end=map(int,kwargs["headers"]["Range"][6:].split("-"))
        return 206,raw[start:end+1],{"Content-Range":f"bytes {start}-{end}/{len(raw)}"},""
    with patch.object(ext,"request",side_effect=fake_request),patch.object(ext.urllib.request,"urlopen",side_effect=fake_urlopen):
        reader=ext.RemoteZipReader("https://example.invalid/archive",len(raw),64)
        with zipfile.ZipFile(reader) as archive:
            with reader.open_member(archive,"archive/member.txt") as member:
                assert member.read()==content


def test_missing_offline_sequence_does_not_acquire_network(tmp_path):
    atoms=[{"atom":"CA","resname":"ALA","altloc":" "}]
    with patch.object(ext,"request",side_effect=AssertionError("offline network request")):
        with pytest.raises(FileNotFoundError,match="offline public source absent"):
            ext.topology_mapping(atoms,tmp_path,offline=True)


def test_standard_reader_batches_match_full_reader_when_available(tmp_path):
    pytest.importorskip("mdtraj")
    from mdtraj.formats import XTCTrajectoryFile
    xyz=np.random.default_rng(18).normal(size=(8,15,3)).astype(np.float32)
    path=tmp_path/"small.xtc"
    with XTCTrajectoryFile(str(path),"w") as f:f.write(xyz,time=np.arange(8,dtype=np.float32))
    with XTCTrajectoryFile(str(path)) as f:expected=f.read(atom_indices=np.array([0,4,9]))[0]*10
    batches=ext.xtc_batches(io.BytesIO(path.read_bytes()),{"ca_atom_indices":np.array([0,4,9])},tmp_path,batch_bytes=200)
    actual=np.concatenate([b[0] for b in batches])
    assert np.array_equal(expected,actual)


def test_trajectory_quantiles_do_not_impute_absent_partner_metrics():
    groups=[{"trajectory_id":"path","role":"model_predicted_refined_path","frames_analyzed":3}]
    rows=[{"trajectory_id":"path","closure_coordinate":x,"DDB1_body_translation_A":""} for x in (.1,.2,.4)]
    summary=ext.trajectory_metric_summary(groups,rows)
    assert len(summary)==1
    assert summary[0]["metric"]=="closure_coordinate"
    assert summary[0]["finite_frame_count"]==3
    assert summary[0]["median"]==.2
    assert np.isclose(summary[0]["p05"],.11)
    assert np.isclose(summary[0]["p95"],.38)


def test_manual_checkpoint_version_does_not_prove_scientific_cache_identity(tmp_path):
    table=tmp_path/"frames.csv.gz";table.write_bytes(b"derived observations")
    old={"analysis_version":2,"trajectory_id":"source.xtc","source_sha256":"raw-source",
         "output_sha256":ext.sha256(table),"source_bytes":123,"zip_crc32_verified":True}
    with patch.object(ext,"scientific_fingerprint",return_value={"sha256":"current-formula"}):
        assert not ext.checkpoint_is_verified(old,table,tmp_path,{},tmp_path,{"bytes":123})
        old.update(analysis_version=3,scientific_fingerprint="previous-formula")
        assert not ext.checkpoint_is_verified(old,table,tmp_path,{},tmp_path,{"bytes":123})


def test_structural_zinc_does_not_change_the_apo_ligand_classification():
    atoms=[{"resname":"ALA"},{"resname":"ZN"},{"resname":"WAT"},{"resname":"Cl-"}]
    with patch.object(ext,"zenodo_member",return_value=b"primary topology"), \
         patch.object(ext,"pdb_frames",return_value=iter([(1,atoms)])), \
         patch.object(ext,"topology_mapping",return_value={"maps":{},"missing_core":[]}):
        evidence=ext.topology_identity("topology.pdb",Path("unused"),True)
    assert evidence["source_ligand_components"]==[]
    assert evidence["structural_metal_components"]==["ZN"]
    condition=ext.trajectory_source_condition("archive/apo/relax.xtc",evidence["source_ligand_components"])
    assert condition.startswith("apo;")
    assert "bias flag" in condition
