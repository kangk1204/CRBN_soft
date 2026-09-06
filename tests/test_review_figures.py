"""Numerical identity and failure-gate checks for the review figure layer."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import build_review_figures as subject


def small_contacts() -> pd.DataFrame:
    return pd.DataFrame([
        {"group_id":"221:CRBN_DDB1","residue":221,"flexible_D_g":.0013456620404622,
         "flexible_derivative_log_C_close":-.0001040932363507,
         "flexible_derivative_log_mean_compliance":-.0014426671122309,
         "delta_R_body_derivative_log_S_close":.0002451280361372,
         "delta_R_internal_derivative_log_S_close":-.00007985015047026127},
        {"group_id":"222:CRBN_DDB1","residue":222,"flexible_D_g":.001807298337357,
         "flexible_derivative_log_C_close":-.0001252396033778,
         "flexible_derivative_log_mean_compliance":-.001921877096236,
         "delta_R_body_derivative_log_S_close":.0003389180602306,
         "delta_R_internal_derivative_log_S_close":-.0001212529885334},
    ])


def test_contact_plot_identity_never_changes_close_neighbour_coordinates():
    data=small_contacts()
    rows=subject.contact_trace(data,{221:"Y221",222:"K222"},"source.csv")
    assert len(rows)==4
    for row in rows:
        source=data.set_index("group_id").loc[row["identity"]]
        assert row["x"]==source[row["x_source_column"]]
        assert row["y"]==source[row["y_source_column"]]
    assert {r["label"] for r in rows}=={"Y221","K222"}
    assert rows[0]["x"] != rows[2]["x"]


@pytest.mark.parametrize("fault",["duplicate","unknown_identity","nonfinite"])
def test_contact_source_corruption_is_rejected(fault):
    data=small_contacts(); labels={221:"Y221",222:"K222"}
    if fault=="duplicate":data=pd.concat([data,data.iloc[[0]]],ignore_index=True)
    elif fault=="unknown_identity":labels.pop(222)
    else:data.loc[0,"flexible_derivative_log_C_close"]=np.nan
    with pytest.raises(subject.FigureSourceError):
        subject.contact_trace(data,labels,"source.csv")


def test_cif_labels_use_the_observed_chain_and_first_ca_identity(tmp_path):
    path=tmp_path/"identity.cif.gz"
    with gzip.open(path,"wt") as f:
        f.write("data_example\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n"
                "_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n_atom_site.label_comp_id\n"
                "ATOM CA A 221 ALA\nATOM CA B 221 TYR\nATOM CA B 221 TYR\n"
                "ATOM CA B 222 LYS\nATOM CB B 339 SER\nATOM CA B 339 SER\n#\n")
    assert subject.read_residue_labels(path)=={221:"Y221",222:"K222",339:"S339"}


def test_snapshots_preserve_exact_bytes(tmp_path):
    source=tmp_path/"source.csv"
    source.write_bytes(b"id,value\n1,0.0013456620404622\n")
    target=subject.snapshot(source,tmp_path/"sources")
    assert subject.digest(source)==subject.digest(target)


def test_legend_replacement_preserves_other_figures_and_requires_exact_heading(tmp_path):
    baseline=tmp_path/"old.md"
    baseline.write_text("# Figures\n\n## Fig1\n\nUnchanged one.\n\n## Fig3\n\nOld three.\n\n"
                        "## Fig4\n\nOld four.\n\n## FigS3\n\nUnchanged S3.\n\n")
    output=tmp_path/"output"; output.mkdir()
    text=subject.update_legends(baseline,output,{"FigS8":"New window figure."}).read_text()
    assert "## Fig1\n\nUnchanged one.\n\n" in text
    assert "## FigS3\n\nUnchanged S3.\n\n" in text
    assert text.count("## Fig3\n")==1 and text.count("## FigS8\n")==1
    assert "Old three" not in text and "Old four" not in text


def test_default_build_does_not_mark_missing_window_result_complete(tmp_path):
    config=tmp_path/"config.json"; config.write_text("{}")
    output=tmp_path/"package/analysis/figure_sources"
    with pytest.raises(subject.FigureSourceError):
        subject.build(config,output,figures=["FigS8"],offline=True)
    assert not (output/"review_figure_build_summary.json").exists()


def test_unknown_figure_does_not_fall_back_to_another_result(tmp_path):
    config=tmp_path/"config.json"; config.write_text("{}")
    with pytest.raises(subject.FigureSourceError,match="Unknown figures"):
        subject.build(config,tmp_path/"out",figures=["Fig6"])


def toy_window_tables(tmp_path):
    """Schema-only unit fixture; these toy values are never rendered or exported."""
    conditions=[(p,c,w) for p in subject.REF_ORDER for c in (13.,15.,18.)
                for w in ("uniform","inverse_square")]
    coverage=[{"pdb":p,"reference_type":"apo","residue":r,"domain":"HB",
               "status":"core" if r<=269 else "observed_added" if r<=279 else "unobserved"}
              for p in subject.REF_ORDER for r in range(1,443)]
    models=[{"pdb":p,"cutoff_A":c,"weighting":w,"window":window,"model":model,
             "n_core":269,"n_added":0 if window=="original" else 10,
             "C_close":2.,"mean_compliance":1.,"S_close":2.}
            for p,c,w in conditions for window in ("original","expanded")
            for model in subject.MODEL_ORDER]
    comparisons=[{"pdb":p,"cutoff_A":c,"weighting":w,"window":window,
                  "role":role,"target":target,"effect":0.}
                 for p,c,w in conditions for window in ("original","expanded")
                 for role in ("R_body","R_internal","R_total","M")
                 for target in ("finite","tangent")]
    robustness=[{"group_id":f"{r}:CRBN_DDB1","residue":r,"contact_class":"CRBN_DDB1",
                 "policy":policy,"legacy_stable_apo":r<195,"legacy_engineered":r<192,
                 "expanded_apo_stable":False,"expanded_engineered_consistent":False,
                 "expanded_all_weighted_stable":False,"condition_results":"toy fixture"}
                for r in range(187,329) for policy in subject.WINDOW_POLICIES]
    effects=[{"pdb":p,"cutoff_A":c,"weighting":w,"window":window,"policy":policy,
              "group_id":f"{r}:CRBN_DDB1","residue":r,"contact_class":"CRBN_DDB1",
              "status":"present","flexible_D_g":0.}
             for p,c,w in conditions
             for window,policy in (("original","original_edges"),("expanded","original_edges"),
                                   ("expanded","expanded_incident_edges"))
             for r in range(187,329)]
    for name,rows in {"coverage":coverage,"models":models,"comparisons":comparisons,
                      "robustness":robustness,"group_effects":effects}.items():
        pd.DataFrame(rows).to_csv(tmp_path/f"{name}.csv",index=False)
    (tmp_path/"summary.json").write_text(json.dumps({"status":"complete","condition_count":30,
                                                    "all_verification_pass":True}))
    return tmp_path


@pytest.mark.parametrize("fault",["pilot_gate","missing_condition","changed_observable",
                                  "mismatched_decomposition","missing_candidate","unclassified_flag"])
def test_completed_window_gate_rejects_partial_or_inconsistent_sources(tmp_path,fault):
    directory=toy_window_tables(tmp_path)
    subject.validated_window_sources(directory,{})
    if fault=="pilot_gate":
        (directory/"summary.json").write_text(json.dumps({"status":"pilot","condition_count":1,
                                                         "all_verification_pass":True}))
    elif fault in {"missing_condition","changed_observable"}:
        path=directory/"models.csv";data=pd.read_csv(path)
        if fault=="missing_condition":data=data.iloc[1:]
        else:data.loc[0,"S_close"]=3.
        data.to_csv(path,index=False)
    elif fault=="mismatched_decomposition":
        path=directory/"comparisons.csv";data=pd.read_csv(path)
        data.loc[0,"effect"]=0.1;data.to_csv(path,index=False)
    elif fault=="missing_candidate":
        path=directory/"group_effects.csv"
        pd.read_csv(path).iloc[1:].to_csv(path,index=False)
    else:
        path=directory/"robustness.csv";data=pd.read_csv(path).astype({"expanded_apo_stable":object})
        data.loc[0,"expanded_apo_stable"]="unknown";data.to_csv(path,index=False)
    with pytest.raises(subject.FigureSourceError):
        subject.validated_window_sources(directory,{})


def toy_path_tables(tmp_path):
    """Only table-shape/adoption tests; no scientific output is rendered."""
    rows=[];groups=[];mapping=[]
    for name in subject.PATH_LABELS:
        groups.append({"trajectory_id":name,"status":"eligible","frames_analyzed":20})
        mapping.append({"trajectory_id":name,"core_count":269,"DDB1_mapped_count":0})
        for frame in range(1,21):
            rows.append({"trajectory_id":name,"frame":frame,"source_member":"unit-fixture.pdb",
                         "source_model":frame,"closure_coordinate":frame/40,
                         "NTD_TBD_centroid_distance_A":30.,"TBD_body_rotation_deg":0.,
                         "TBD_internal_RMSD_A":0.,"DDB1_observed_mapped_CA_count":0,
                         "quantitative_adoption":subject.PATH_ADOPTION if name=="final_refined" else "unrefined_geometry_diagnostic_only",
                         "CRBN_adjacent_CA_below_2p5A":0 if name=="final_refined" else 1,
                         "CRBN_adjacent_CA_above_4p5A":0})
    pd.DataFrame(rows).to_csv(tmp_path/subject.PATH_FILES[0],index=False)
    (tmp_path/subject.PATH_FILES[1]).write_text(json.dumps({"raw_sources_retained":True,
         "offline_raw_PDB_recomputation":True,"frames":60,"groups":groups}))
    (tmp_path/subject.PATH_FILES[2]).write_text(json.dumps(mapping))
    return tmp_path


@pytest.mark.parametrize("fault",["compression","diagnostic_promoted","missing_frame","missing_core"])
def test_path_adoption_does_not_promote_diagnostics_or_incomplete_coordinates(tmp_path,fault):
    directory=toy_path_tables(tmp_path)
    retained,_=subject.validated_path_sources(directory)
    assert len(retained)==60
    assert (retained.quantitative_adoption==subject.PATH_ADOPTION).sum()==20
    if fault=="missing_core":
        path=directory/subject.PATH_FILES[2];mapping=json.loads(path.read_text())
        mapping[-1]["core_count"]=268;path.write_text(json.dumps(mapping))
    else:
        path=directory/subject.PATH_FILES[0];data=pd.read_csv(path)
        if fault=="compression":data.loc[data.trajectory_id=="final_refined","CRBN_adjacent_CA_below_2p5A"]=1
        elif fault=="diagnostic_promoted":data.loc[0,"quantitative_adoption"]=subject.PATH_ADOPTION
        else:data=data.iloc[1:]
        data.to_csv(path,index=False)
    with pytest.raises(subject.FigureSourceError):
        subject.validated_path_sources(directory)


def toy_trajectory_tables(tmp_path):
    """Small artificial distributions for contract tests, never rendered."""
    inventory=[];metrics=[];frames=[]
    for index in range(21):
        name=f"unit-fixture/{index}.xtc"
        if index>=2:
            inventory.append({"trajectory_id":name,"status":"excluded_duplicate_representation",
                              "frames_analyzed":0})
            continue
        inventory.append({"trajectory_id":name,"status":"all_frames_analyzed","frames_analyzed":3,
                          "DDB1_mapped_positions":830,"zip_crc32_verified":True,"core_positions":269})
        for frame in range(3):
            frames.append({"trajectory_id":name,"frame":frame+1,
                           **{metric:frame for metric in subject.TRAJECTORY_METRICS}})
        for metric in subject.TRAJECTORY_METRICS:
            metrics.append({"trajectory_id":name,"metric":metric,"frames_analyzed":3,"finite_frame_count":3,
                            "DDB1_mapped_positions":830,"coordinate_role":"biased_simulation_coordinate_comparator",
                            "source_condition":"three-CV meta-eABF apo","quantitative_adoption":subject.PATH_ADOPTION,
                            "minimum":0.,"p05":.1,"median":1.,"p95":1.9,"maximum":2.})
    pd.DataFrame(metrics).to_csv(tmp_path/subject.TRAJECTORY_FILES[0],index=False)
    (tmp_path/subject.TRAJECTORY_FILES[1]).write_text(json.dumps(inventory))
    pd.DataFrame(inventory).to_csv(tmp_path/subject.TRAJECTORY_FILES[2],index=False)
    with gzip.open(tmp_path/subject.TRAJECTORY_FILES[3],"wt") as handle:
        pd.DataFrame(frames).to_csv(handle,index=False)
    summary={"analysis_complete":True,"all_XTC_members_accounted_for":True,
             "xtc_archive_member_count":21,"xtc_analyzed_comparison_count":2}
    (tmp_path/subject.TRAJECTORY_FILES[-1]).write_text(json.dumps(summary))
    refresh_trajectory_hashes(tmp_path)
    return tmp_path


def refresh_trajectory_hashes(directory):
    path=directory/subject.TRAJECTORY_FILES[-1];summary=json.loads(path.read_text())
    summary["retained_output_hashes"]=[{"file":subject.TRAJECTORY_FILES[i],
                                         "sha256":subject.digest(directory/subject.TRAJECTORY_FILES[i])}
                                        for i in (0,1,3)]
    path.write_text(json.dumps(summary))


@pytest.mark.parametrize("fault",["partial","missing_metric","missing_frames","mixed_conditions",
                                  "unordered_quantile","raw_hash","auxiliary_hash","unresolved_member"])
def test_trajectory_figure_requires_all_frames_and_completed_adoption(tmp_path,fault):
    directory=toy_trajectory_tables(tmp_path)
    observed,_,_=subject.validated_trajectory_sources(directory)
    assert observed.trajectory_id.nunique()==2
    if fault=="partial":
        path=directory/subject.TRAJECTORY_FILES[-1];summary=json.loads(path.read_text())
        summary["analysis_complete"]=False;path.write_text(json.dumps(summary))
    elif fault=="raw_hash":
        with (directory/subject.TRAJECTORY_FILES[3]).open("ab") as handle:handle.write(b"altered")
    elif fault=="auxiliary_hash":
        extra=directory/"source_condition_audit.csv";extra.write_text("status\nverified\n")
        path=directory/subject.TRAJECTORY_FILES[-1];summary=json.loads(path.read_text())
        summary["retained_output_hashes"].append({"file":extra.name,"sha256":subject.digest(extra)})
        path.write_text(json.dumps(summary));extra.write_text("status\nchanged\n")
    elif fault=="unresolved_member":
        path=directory/subject.TRAJECTORY_FILES[1];inventory=json.loads(path.read_text())
        inventory[0]["status"]="streaming";path.write_text(json.dumps(inventory))
        refresh_trajectory_hashes(directory)
    else:
        path=directory/subject.TRAJECTORY_FILES[0];data=pd.read_csv(path)
        if fault=="missing_metric":data=data.iloc[1:]
        elif fault=="missing_frames":data.loc[0,"finite_frame_count"]=2
        elif fault=="mixed_conditions":data.loc[0,"source_condition"]="path-CV meta-eABF apo"
        else:data.loc[0,"median"]=5.
        data.to_csv(path,index=False);refresh_trajectory_hashes(directory)
    with pytest.raises(subject.FigureSourceError):
        subject.validated_trajectory_sources(directory)


@pytest.mark.parametrize("condition,stage,ligand", [
    ("three-CV meta-eABF; apo", "three_cv", "Apo"),
    ("path-CV meta-eABF; ligand-containing (K6X)", "path_cv", "Ligand-containing"),
    ("path-selected coordinates from path-CV meta-eABF; ligand-containing (K6X)",
     "path_selected", "Ligand-containing"),
    ("processed protein-only coordinates from path-CV meta-eABF directory; apo", "protein_only", "Apo"),
    ("ligand-containing (K6X); relaxation stage (bias flag for this execution not established)",
     "relaxation", "Ligand-containing"),
])
def test_trajectory_labels_preserve_source_processing_role(condition, stage, ligand):
    display = subject.trajectory_display(condition)
    assert display["source_stage"] == stage
    assert display["ligand_label"] == ligand


@pytest.mark.parametrize("condition", ["three-CV meta-eABF apo", "unidentified; apo",
                                        "path-CV meta-eABF; unknown ligand state"])
def test_trajectory_labels_reject_ambiguous_stage_or_ligand(condition):
    with pytest.raises(subject.FigureSourceError):
        subject.trajectory_display(condition)
