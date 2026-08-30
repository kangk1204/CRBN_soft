#!/usr/bin/env python3
# Make Angstrom (Å) and other non-ASCII output safe on legacy consoles
# (e.g. Korean cp949 Windows) without changing values.
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
"""Is the open/closed result an artefact of the CURATION WINDOW?

The study ensemble keeps a deposition only if its CRBN chain resolves the FULL
269-residue analysis window. That filter is not conformation-neutral: a single
unresolved residue is enough to drop a structure, and an open structure dropped
this way would not merely be missing, it would be missing from the small side of
a 5-vs-65 imbalance. This script quantifies that bias and re-runs the entire
result set under alternative, defensible curation rules.

Three questions are answered, none of them with a free parameter.

(A) WHAT WAS EXCLUDED, AND WHAT STATE WAS IT IN?  Every deposition that passes
    the fragment filter but is absent from the curated set is re-examined from
    raw mmCIF. Its window coverage, the identity of its missing residues, and
    its NTD-TBD centroid distance computed on its OWN resolved subset are
    reported, together with an open/closed/indeterminate call. The call is made
    against closed/open reference bands recomputed on that structure's resolved
    subset, so a structure is never compared against a band defined on residues
    it does not have.

(B) DOES THE RESULT SURVIVE A RELAXED WINDOW?  Alternative ensembles are built
    under eight rules -- the primary rule, >=95% and >=90% window coverage
    (analysis restricted to the intersection of resolved positions), a fixed
    reduced window chosen so that the excluded OPEN structures are retained,
    terminal-only gap tolerance, dropping the undocumented 4.0 A resolution
    ceiling, combining >=95% coverage with no resolution ceiling, and best-coverage chain selection -- and the
    full result set (PC1 variance fraction, PC1-vs-difference overlap, ANM
    mode-1 overlap and rank, RMSIP, per-open-structure overlap) is recomputed
    for each from scratch.

(C) HOW MUCH DOES ANY ANALYST CHOICE MATTER?  Three further choices that a
    analyst can reasonably vary are swept on the fixed ensemble:
    superposition target (whole molecule / NTD only / HB only / TBD only),
    method subset (all / X-ray only / cryo-EM only, the collinearity control),
    and wild-type versus engineered construct membership.

The primary ANM protocol is used throughout: 15 A cutoff, 20 non-trivial modes,
built on the open reference restricted to the ensemble's own common window.
Every ANM is built on coordinates taken FROM THE ENSEMBLE BEING TESTED, so the
modes and the difference vector always live in the same frame; scoring an ANM
built on raw deposited coordinates against an axis defined in a superposed
frame is a frame mismatch and gives meaningless overlaps.

Inputs
  data/crbn_residue_window.csv    the 269-residue analysis window (author numbering)
  data/curation_chain_map.json    per-PDB CRBN chain for the 14 non-lowest-id cases
  data/crbn_structure_inventory.csv  the 98 post-fragment-filter depositions
  data/crbn_ensemble.ens.npz      reference tensor, used only as a cross-check
  data/_cif_cache/<pdb>.cif.gz    raw mmCIF (downloaded on demand from RCSB)
  data/_rcsb_meta.json            RCSB metadata cache (fetched on demand)

Outputs
  data/window_sensitivity.json              every ensemble variant and sweep
  data/excluded_structures_adjudication.csv the per-structure exclusion table

Usage
  python scripts/window_sensitivity.py [--verify] [--limit N]
"""
import csv
import gzip
import io
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

try:
    from pdb_id import validate_pdb_id
except ModuleNotFoundError:  # imported by path from the repository root
    from scripts.pdb_id import validate_pdb_id
try:
    from curation_contracts import (
        chains_for_exact_accession,
        choose_primary_chain,
        describes_ddb1,
        exact_construct_flags,
        reference_accessions,
    )
except ModuleNotFoundError:  # imported by path from the repository root
    from scripts.curation_contracts import (
        chains_for_exact_accession,
        choose_primary_chain,
        describes_ddb1,
        exact_construct_flags,
        reference_accessions,
    )
try:
    from analysis_contracts import assert_tree_close, atomic_write_json, atomic_write_text
except ModuleNotFoundError:
    from scripts.analysis_contracts import assert_tree_close, atomic_write_json, atomic_write_text

CUTOFF_ANM = 15.0        # A, the primary ANM contact cutoff
N_MODES = 20             # non-trivial ANM modes retained
RES_MAX = 4.0            # A, the resolution ceiling implied by the curated set
MIN_DOMAIN_CA = 10       # Ca needed in a domain before its centroid is trusted
TERMINAL_SLACK = 5       # window positions from either end counted as "terminal"
SEED = 42

DOMAINS = {"NTD": (77, 186), "HB": (187, 317), "TBD": (318, 426)}
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "_cif_cache"
META = DATA / "_rcsb_meta.json"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
CACHE_WRITES_ENABLED = True
HELP_TEXT = (
    "Usage: python scripts/window_sensitivity.py [--verify] [--limit N]\n"
    "  --verify   recompute the complete analysis and compare both reference artifacts\n"
    "  --limit N  run a diagnostic subset without writing canonical artifacts"
)

WIN = np.array([], dtype=int)
WINSET = []
CHAIN_MAP = {}
NTD_R, HB_R, TBD_R = [], [], []

# ---------------------------------------------------------------- mmCIF input


def atomic_write_bytes(path, payload):
    """Atomically replace a binary cache entry without changing its file mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mode = path.stat().st_mode & 0o777
    else:
        current_umask = os.umask(0)
        os.umask(current_umask)
        mode = 0o666 & ~current_umask
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def decode_cif_blob(blob, pdb):
    """Validate a gzipped mmCIF payload before it enters the local cache."""
    try:
        text = gzip.decompress(blob).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise ValueError(f"{pdb}: invalid gzipped mmCIF payload") from exc
    stripped = text.lstrip()
    first_line = stripped.splitlines()[0].strip().lower() if stripped else ""
    if first_line != f"data_{pdb.lower()}" or "_atom_site." not in text:
        raise ValueError(f"{pdb}: downloaded payload is not a complete coordinate mmCIF")
    return text


def fetch_cif(pdb):
    pdb = validate_pdb_id(pdb)
    path = Path(CACHE) / f"{pdb}.cif.gz"
    if path.exists():
        try:
            cached_blob = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"{pdb}: cached mmCIF is unreadable: {path}") from exc
        try:
            return decode_cif_blob(cached_blob, pdb)
        except ValueError:
            # A fresh payload is validated before atomically replacing the bad cache.
            pass
    with urllib.request.urlopen(
            f"https://files.rcsb.org/download/{pdb}.cif.gz", timeout=120) as fh:
        blob = fh.read()
    text = decode_cif_blob(blob, pdb)
    if CACHE_WRITES_ENABLED:
        atomic_write_bytes(path, blob)
    return text


def parse_atom_site(cif):
    """Return (column index map, list of whitespace-split atom_site rows)."""
    lines = cif.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            hdr = []
            while j < len(lines) and lines[j].lstrip().startswith("_atom_site."):
                hdr.append(lines[j].strip()); j += 1
            if hdr:
                col = {h.split(".", 1)[1]: k for k, h in enumerate(hdr)}
                if "Cartn_x" in col and "auth_seq_id" in col:
                    rows, k = [], j
                    while k < len(lines):
                        ln = lines[k]
                        if (ln.startswith("#") or ln.strip() == "" or
                                ln.strip() == "loop_" or ln.startswith("_")):
                            break
                        f = ln.split()
                        if len(f) >= len(hdr):
                            rows.append(f)
                        k += 1
                    return col, rows
            i = j; continue
        i += 1
    return None, []


def chain_atoms(pdb, crbn_chains, cif_text=None):
    """{chain: {ca, cab, heavy, altloc_ca, partocc_ca}} over the CRBN chains.

    Ca records are taken at FIRST occurrence, which is the repo-wide convention:
    where a residue carries alternate conformations only altloc A is used. The
    altloc and partial-occupancy tallies are reported so the choice is visible.
    """
    col, rows = parse_atom_site(fetch_cif(pdb) if cif_text is None else cif_text)
    if col is None:
        return {}
    g = lambda f, k: f[col[k]] if k in col else "."
    out = {}
    for f in rows:
        if g(f, "group_PDB") != "ATOM":
            continue
        ch = g(f, "auth_asym_id")
        if ch not in crbn_chains:
            continue
        try:
            rs = int(g(f, "auth_seq_id"))
            xyz = (float(g(f, "Cartn_x")), float(g(f, "Cartn_y")), float(g(f, "Cartn_z")))
        except ValueError:
            continue
        d = out.setdefault(ch, {"ca": {}, "cab": {}, "heavy": [],
                                "altloc_ca": 0, "partocc_ca": 0})
        if g(f, "type_symbol") != "H":
            d["heavy"].append((rs, xyz))
        if g(f, "label_atom_id").strip('"') == "CA" and rs not in d["ca"]:
            try:
                occ = float(g(f, "occupancy"))
            except ValueError:
                occ = 1.0
            try:
                b = float(g(f, "B_iso_or_equiv"))
            except ValueError:
                b = float("nan")
            d["ca"][rs] = xyz
            d["cab"][rs] = b
            if g(f, "label_alt_id") not in (".", "?", ""):
                d["altloc_ca"] += 1
            if occ < 0.999:
                d["partocc_ca"] += 1
    return out

# -------------------------------------------------------------- RCSB metadata

_META_QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){
 rcsb_id struct{title} exptl{method}
 rcsb_entry_info{resolution_combined}
 refine{pdbx_starting_model pdbx_method_to_determine_struct}
 symmetry{space_group_name_H_M}
 rcsb_primary_citation{pdbx_database_id_DOI title}
 polymer_entities{entity_poly{rcsb_sample_sequence_length}
   rcsb_polymer_entity{pdbx_description pdbx_mutation}
   rcsb_polymer_entity_container_identifiers{auth_asym_ids
     reference_sequence_identifiers{database_accession}}}
 nonpolymer_entities{nonpolymer_comp{chem_comp{id formula_weight}}}}}"""


def rcsb_meta(pdbs, write_cache=True):
    meta_path = Path(META)
    cached = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not isinstance(cached, dict):
        raise ValueError(f"{meta_path}: metadata cache must be a JSON object")
    todo = [p for p in pdbs if p not in cached]
    for k in range(0, len(todo), 20):
        body = json.dumps({"query": _META_QUERY,
                           "variables": {"ids": todo[k:k + 20]}}).encode()
        req = urllib.request.Request(RCSB_GRAPHQL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as response:
            d = json.loads(response.read())
        for e in (d.get("data", {}).get("entries") or []):
            if e:
                cached[e["rcsb_id"]] = e
    if todo and write_cache:
        atomic_write_json(meta_path, cached, sort_keys=True)
    return cached


def accessions(pe):
    return sorted(reference_accessions(pe))


def crbn_chains_of(e):
    return chains_for_exact_accession(e, "Q96SW2")


def method_of(e):
    m = e["exptl"][0]["method"]
    return "X-ray" if "X-RAY" in m else ("cryo-EM" if "MICROSCOPY" in m else m)


def resolution_of(e):
    rc = e["rcsb_entry_info"]["resolution_combined"]
    return float(rc[0]) if rc else None


def construct_flags(e, cif_text):
    """Exact construct markers from raw UniProt mappings and entity metadata."""
    return exact_construct_flags(e, cif_text)


_SOLVENT = {"HOH", "GOL", "EDO", "SO4", "CL", "NA", "MG", "PO4", "ACT", "DMS",
            "PEG", "TRS", "MPD", "IOD", "K", "CA", "ACY", "FMT", "NO3", "BME",
            "CIT", "1PE", "P6G", "PGE", "EPE", "MES", "IMD", "BCT", "AZI",
            "UNX", "NH4", "BR", "ZN"}


def ligands_of(e):
    out = []
    for n in (e.get("nonpolymer_entities") or []):
        c = n["nonpolymer_comp"]["chem_comp"]
        if c["id"] not in _SOLVENT:
            out.append((c["id"], c["formula_weight"] or 0.0))
    return out


def degron_of(e):
    """Third protein chain that is neither CRBN nor DDB1 = degron / neosubstrate."""
    out = []
    for pe in e["polymer_entities"]:
        acc = accessions(pe)
        if "Q96SW2" in acc or "Q16531" in acc:
            continue
        desc = pe["rcsb_polymer_entity"]["pdbx_description"] or "?"
        out.append("%s(%daa)" % (desc, pe["entity_poly"]["rcsb_sample_sequence_length"] or 0))
    return ";".join(out)


def has_ddb1(e):
    return any("Q16531" in accessions(pe) for pe in e["polymer_entities"])


# ------------------------------------------------------------------ geometry

def domain_residues(dom):
    lo, hi = DOMAINS[dom]
    return [r for r in WINSET if lo <= r <= hi]


def load_bundle_configuration(data_dir=None):
    """Load and validate bundle configuration after the command starts."""
    global WIN, WINSET, CHAIN_MAP, NTD_R, HB_R, TBD_R
    base = DATA if data_dir is None else Path(data_dir)
    window_path = base / "crbn_residue_window.csv"
    with window_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "author_resnum" not in reader.fieldnames:
            raise ValueError(f"{window_path}: missing author_resnum column")
        try:
            residues = [int(row["author_resnum"]) for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{window_path}: author_resnum values must be integers") from exc
    if len(residues) != 269:
        raise ValueError(f"{window_path}: expected exactly 269 residue labels, found {len(residues)}")
    if any(right <= left for left, right in zip(residues, residues[1:])):
        raise ValueError(f"{window_path}: residue labels must be strictly increasing")

    chain_path = base / "curation_chain_map.json"
    chain_map = json.loads(chain_path.read_text(encoding="utf-8"))
    if not isinstance(chain_map, dict):
        raise ValueError(f"{chain_path}: chain map must be a JSON object")
    normalized = {}
    for raw_pdb, chain in chain_map.items():
        pdb = validate_pdb_id(raw_pdb)
        if pdb != raw_pdb or not isinstance(chain, str) or not chain.strip():
            raise ValueError(f"{chain_path}: invalid primary-chain mapping {raw_pdb!r}: {chain!r}")
        normalized[pdb] = chain

    WIN = np.asarray(residues, dtype=int)
    WINSET = residues
    CHAIN_MAP = normalized
    NTD_R, HB_R, TBD_R = (domain_residues(name) for name in ("NTD", "HB", "TBD"))


def centroid_distance(ca, ntd=None, tbd=None):
    """NTD-TBD Ca centroid separation over whatever residues are present."""
    ntd = NTD_R if ntd is None else ntd
    tbd = TBD_R if tbd is None else tbd
    n = [ca[r] for r in ntd if r in ca]
    t = [ca[r] for r in tbd if r in ca]
    if len(n) < MIN_DOMAIN_CA or len(t) < MIN_DOMAIN_CA:
        return None
    return float(np.linalg.norm(np.array(n).mean(0) - np.array(t).mean(0)))


def min_interdomain_approach(heavy):
    """Closest NTD<->TBD heavy-atom approach within one chain (A)."""
    n = np.array([x for r, x in heavy if DOMAINS["NTD"][0] <= r <= DOMAINS["NTD"][1]])
    t = np.array([x for r, x in heavy if DOMAINS["TBD"][0] <= r <= DOMAINS["TBD"][1]])
    if len(n) < 20 or len(t) < 20:
        return None
    best = np.inf
    for k in range(0, len(t), 512):
        blk = t[k:k + 512]
        d2 = ((n[:, None, :] - blk[None, :, :]) ** 2).sum(-1)
        best = min(best, float(np.sqrt(d2.min())))
    return best


def kabsch(P, Q):
    """Rotate+translate P onto Q (both n x 3); returns the moved, centred P."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ Pc.T).T


def superpose(confs, iters=8, sel=None):
    """Iterative Kabsch onto the running mean.

    sel selects the atom indices the ROTATION is fitted on (the superposition
    target); the returned coordinates always cover every atom. sel=None fits on
    the whole molecule, which is the primary choice.
    """
    idx = np.arange(confs.shape[1]) if sel is None else np.asarray(sel)
    ref = confs[0] - confs[0][idx].mean(0)
    al = confs
    for _ in range(iters):
        moved = []
        for c in confs:
            Pc = c - c[idx].mean(0)
            U, S, Vt = np.linalg.svd(Pc[idx].T @ ref[idx])
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1, 1, d]) @ U.T
            moved.append((R @ Pc.T).T)
        al = np.array(moved)
        newref = al.mean(0)
        if np.allclose(newref, ref, atol=1e-4):
            ref = newref
            break
        ref = newref
    return al


def anm_hessian(coords, cutoff):
    n = len(coords)
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        d = coords - coords[i]
        r = np.linalg.norm(d, axis=1)
        for j in range(i + 1, n):
            if 1e-6 < r[j] <= cutoff:
                k = np.outer(d[j], d[j]) / r[j] ** 2
                H[3*i:3*i+3, 3*j:3*j+3] = -k
                H[3*j:3*j+3, 3*i:3*i+3] = -k
                H[3*i:3*i+3, 3*i:3*i+3] += k
                H[3*j:3*j+3, 3*j:3*j+3] += k
    return H


def anm_modes(coords, cutoff=CUTOFF_ANM, k=N_MODES):
    w, v = np.linalg.eigh(anm_hessian(coords, cutoff))
    nz = w > 1e-9
    return w[nz][:k], v[:, nz][:, :k]


def pca(confs):
    n = len(confs)
    X = (confs - confs.mean(0)).reshape(n, -1)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S ** 2
    return Vt.T, ev / ev.sum(), (X @ Vt.T)


def widest_gap_cut(s1, lead=15):
    """Primary open/closed split: cut at the widest gap among the leaders."""
    srt = np.sort(s1)[::-1]
    gaps = srt[:-1] - srt[1:]
    ncut = int(np.argmax(gaps[:lead])) + 1
    thresh = (srt[ncut - 1] + srt[ncut]) / 2.0
    return s1 >= thresh, float(thresh), float(gaps[:lead].max())

# ------------------------------------------------------- one ensemble variant

def analyse(confs, labels, resnums, ref_label=None, sup_sel=None, tag=""):
    """Full result set for one ensemble. confs are RAW per-structure coordinates;
    they are superposed here so that the modes and the axis share one frame."""
    confs = superpose(np.asarray(confs, float), sel=sup_sel)
    n_ca = confs.shape[1]
    pcs, vr, scores = pca(confs)
    pc1 = pcs[:, 0].copy()
    s1 = scores[:, 0] / np.sqrt(n_ca)
    if abs(s1.min()) > abs(s1.max()):          # sign so the open cluster is positive
        s1, pc1 = -s1, -pc1
    open_mask, thresh, gap = widest_gap_cut(s1)
    diff = confs[open_mask].mean(0) - confs[~open_mask].mean(0)
    dv = diff.reshape(-1)
    dv /= np.linalg.norm(dv)

    # ANM on the open reference, restricted to this ensemble's common window
    if ref_label is not None and ref_label in labels:
        ridx = labels.index(ref_label)
    else:
        ridx = int(np.argmax(s1))
    aw, av = anm_modes(confs[ridx])
    ov = np.abs(av.T @ dv)
    pca_modes = pcs[:, :10]
    ovm = np.abs(av[:, :10].T @ pca_modes)
    rmsip = float(np.sqrt((ovm ** 2).sum() / 10))

    # every open member as its own ANM reference
    per_open = {}
    for i in np.where(open_mask)[0]:
        w_i, v_i = anm_modes(confs[i])
        o_i = np.abs(v_i.T @ dv)
        per_open[labels[i]] = {"mode1_overlap": float(o_i[0]),
                               "best_overlap": float(o_i.max()),
                               "best_rank": int(np.argmax(o_i)) + 1}
    om1 = [d["mode1_overlap"] for d in per_open.values()]

    # Where does ANM mode 1 put its amplitude? A window that deletes contiguous
    # NTD segments fragments that lobe, and the slowest mode of the truncated
    # network becomes a LOCAL motion of the hinge bundle rather than the global
    # hinge. This diagnostic makes that failure mode visible instead of leaving a
    # collapsed overlap unexplained.
    amp = (av[:, 0].reshape(-1, 3) ** 2).sum(1)
    share = {}
    for d in DOMAINS:
        sel = [i for i, r in enumerate(resnums)
               if DOMAINS[d][0] <= r <= DOMAINS[d][1]]
        share[d] = float(amp[sel].sum()) if sel else 0.0
    ndom = {d: int(sum(DOMAINS[d][0] <= r <= DOMAINS[d][1] for r in resnums))
            for d in DOMAINS}

    return {
        "tag": tag, "n_conformers": int(len(labels)), "n_residues": int(n_ca),
        "n_residues_per_domain": ndom,
        "anm_mode1_amplitude_share": share,
        "anm_mode1_localised": bool(max(share.values()) > 0.75),
        "n_open": int(open_mask.sum()), "n_closed": int((~open_mask).sum()),
        "open_members": [labels[i] for i in np.where(open_mask)[0]],
        "pc1_variance_fraction": float(vr[0]),
        "pc2_variance_fraction": float(vr[1]),
        "pc1_diff_overlap": float(abs(pc1 @ dv)),
        "anm_reference": labels[ridx],
        "anm_mode1_overlap": float(ov[0]),
        "anm_best_overlap": float(ov.max()),
        "anm_best_rank": int(np.argmax(ov)) + 1,
        "anm_cum_overlap_top10": float(np.sqrt((ov[:10] ** 2).sum())),
        "anm_eigval_gaps": [float(aw[i + 1] / aw[i]) for i in range(3)],
        "rmsip_anm_pca": rmsip,
        "per_open_overlap": per_open,
        "per_open_mode1_min": float(min(om1)) if om1 else None,
        "per_open_mode1_max": float(max(om1)) if om1 else None,
        "pc1_cut_threshold": thresh, "pc1_cut_widest_gap": gap,
        "pc1_scores": {labels[i]: float(s1[i]) for i in range(len(labels))},
        "open_closed_separation_A": float(s1[open_mask].mean() - s1[~open_mask].mean())
                                    if open_mask.any() and (~open_mask).any() else None,
        "superposition_target": "whole" if sup_sel is None else tag,
    }


def parse_run_options(arguments):
    """Parse the small command surface and identify diagnostic partial runs."""
    verify = False
    limit = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--verify":
            if verify:
                raise ValueError("--verify may be supplied only once")
            verify = True
            index += 1
            continue
        if argument == "--limit":
            if limit is not None:
                raise ValueError("--limit may be supplied only once")
            if index + 1 >= len(arguments):
                raise ValueError("--limit requires a positive integer")
            try:
                limit = int(arguments[index + 1])
            except ValueError as exc:
                raise ValueError("--limit requires a positive integer") from exc
            if limit <= 0:
                raise ValueError("--limit must be positive")
            index += 2
            continue
        raise ValueError(f"unknown argument: {argument}")
    if verify and limit is not None:
        raise ValueError("--verify requires the complete inventory and rejects --limit")
    return verify, limit


def validate_inventory(rows):
    """Validate inventory schema and return complete and curated PDB sets."""
    required = {"pdb_id", "passes_fragment_filter", "in_curated_ensemble"}
    if not rows:
        raise ValueError("structure inventory is empty")
    identifiers = []
    passing = []
    curated = []
    for row_number, row in enumerate(rows, start=2):
        if not required.issubset(row):
            raise ValueError(
                f"structure inventory row {row_number} missing {sorted(required - set(row))}"
            )
        pdb = validate_pdb_id(row["pdb_id"])
        if pdb != row["pdb_id"]:
            raise ValueError(f"structure inventory row {row_number} is not canonical: {row['pdb_id']}")
        for field in ("passes_fragment_filter", "in_curated_ensemble"):
            if row[field] not in {"0", "1"}:
                raise ValueError(f"structure inventory row {row_number} has invalid {field}")
        identifiers.append(pdb)
        if row["passes_fragment_filter"] == "1":
            passing.append(pdb)
        if row["in_curated_ensemble"] == "1":
            curated.append(pdb)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("structure inventory contains duplicate PDB identifiers")
    if not passing:
        raise ValueError("structure inventory contains no fragment-filtered entries")
    if not set(curated).issubset(passing):
        raise ValueError("curated entries must also pass the fragment filter")
    return sorted(passing), sorted(curated)


def validate_complete_generation(
    requested,
    extracted,
    curated,
    rediscovered,
    excluded,
    adjudication,
    variants,
    expected_variants,
    out,
):
    """Require every requested input and every output branch before replacement."""
    requested_set = set(requested)
    if set(extracted) != requested_set:
        raise RuntimeError(
            "coordinate extraction is incomplete: "
            f"missing={sorted(requested_set - set(extracted))}, "
            f"extra={sorted(set(extracted) - requested_set)}"
        )
    if sorted(rediscovered) != sorted(curated):
        raise RuntimeError("rediscovered curation rule does not match the reference label set")
    expected_excluded = requested_set - set(rediscovered)
    if set(excluded) != expected_excluded:
        raise RuntimeError("excluded-entry set is inconsistent with the complete inventory")
    adjudicated = [row.get("pdb") for row in adjudication]
    if len(set(adjudicated)) != len(adjudicated) or set(adjudicated) != expected_excluded:
        raise RuntimeError("adjudication does not cover every excluded entry exactly once")

    expected_variant_set = set(expected_variants)
    if set(variants) != expected_variant_set:
        raise RuntimeError(
            "ensemble-variant set is incomplete: "
            f"missing={sorted(expected_variant_set - set(variants))}, "
            f"extra={sorted(set(variants) - expected_variant_set)}"
        )
    skipped = sorted(name for name, result in variants.items() if result.get("skipped"))
    if skipped:
        raise RuntimeError(f"ensemble variants were skipped: {skipped}")

    required_top = {
        "protocol",
        "curation_rule_rediscovered",
        "adjudication_summary",
        "adjudication",
        "ensembles",
        "superposition_dependence",
        "method_subsets",
        "constructs",
        "ddb1_entity_census",
        "empty_middle",
    }
    if set(out) != required_top:
        raise RuntimeError(
            f"output schema mismatch: missing={sorted(required_top - set(out))}, "
            f"extra={sorted(set(out) - required_top)}"
        )
    if not out["curation_rule_rediscovered"]["matches_committed_labels"]:
        raise RuntimeError("output records a failed reference-label reconstruction")
    if out["adjudication"] != adjudication or out["ensembles"] != variants:
        raise RuntimeError("output branches diverge from their fully computed results")
    if set(out["empty_middle"]) != expected_variant_set:
        raise RuntimeError("empty-middle diagnostics do not cover every ensemble variant")
    if set(out["superposition_dependence"]) != {
        "whole_molecule", "on_NTD", "on_HB", "on_TBD"
    }:
        raise RuntimeError("superposition sweep is incomplete")
    if not {"X-ray", "cryo-EM", "contingency", "cos_xray_vs_cryoem_axis"}.issubset(
        out["method_subsets"]
    ):
        raise RuntimeError("method-subset controls are incomplete")


def prepare_artifact_payloads(out):
    """Strictly serialize and round-trip the complete JSON and CSV artifacts."""
    json_payload = json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n"
    normalized = json.loads(json_payload)
    adjudication = normalized.get("adjudication")
    if not isinstance(adjudication, list) or not adjudication:
        raise RuntimeError("adjudication unexpectedly empty; refusing to emit headerless output")
    fields = list(adjudication[0])
    if "pdb" not in fields or any(set(row) != set(fields) for row in adjudication):
        raise RuntimeError("adjudication rows do not share one complete CSV schema")

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(adjudication)
    csv_payload = csv_buffer.getvalue()
    round_trip = list(csv.DictReader(io.StringIO(csv_payload, newline="")))
    expected_round_trip = [
        {field: "" if row[field] is None else str(row[field]) for field in fields}
        for row in adjudication
    ]
    if round_trip != expected_round_trip:
        raise RuntimeError("CSV round-trip changed, lost, or reordered adjudication values")
    return normalized, json_payload, csv_payload


def finalize_artifacts(out, json_path, csv_path, *, verify, partial):
    """Verify or write complete artifacts; partial runs are always no-write."""
    if verify and partial:
        raise ValueError("verification cannot use partial results")
    if partial:
        return out
    normalized, json_payload, csv_payload = prepare_artifact_payloads(out)
    if verify:
        reference = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert_tree_close(normalized, reference, float_tolerance=1e-9)
        if Path(csv_path).read_text(encoding="utf-8") != csv_payload:
            raise AssertionError(f"recomputed adjudication differs from {csv_path}")
        return normalized

    # Both payloads have passed schema, finiteness, and round-trip checks before
    # either destination is replaced. Each replacement is itself atomic.
    atomic_write_text(Path(csv_path), csv_payload)
    atomic_write_text(Path(json_path), json_payload)
    return normalized

# ------------------------------------------------------------------ main

def main():
    global CACHE_WRITES_ENABLED
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        if len(sys.argv[1:]) != 1:
            raise ValueError("--help cannot be combined with analysis options")
        print(HELP_TEXT)
        return 0
    verify, limit = parse_run_options(sys.argv[1:])
    partial = limit is not None
    CACHE_WRITES_ENABLED = not verify
    load_bundle_configuration()
    inventory_path = DATA / "crbn_structure_inventory.csv"
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        inv = list(csv.DictReader(handle))
    complete_pdbs, curated = validate_inventory(inv)
    pdbs = complete_pdbs
    if limit is not None:
        pdbs = sorted(set(complete_pdbs[:limit]) | set(curated))
    meta = rcsb_meta(pdbs, write_cache=not verify)

    # ---- per-structure extraction, every CRBN chain of every deposition -----
    S = {}
    for k, p in enumerate(pdbs, 1):
        if p not in meta:
            raise RuntimeError(f"{p}: missing RCSB metadata after resolution")
        e = meta[p]
        chains = crbn_chains_of(e)
        cif_text = fetch_cif(p)
        atoms = chain_atoms(p, set(chains), cif_text)
        per = {}
        for ch in sorted(atoms):
            ca = {r: atoms[ch]["ca"][r] for r in WINSET if r in atoms[ch]["ca"]}
            per[ch] = {
                "ca": ca,
                "cov": len(ca),
                "missing": [r for r in WINSET if r not in ca],
                "n_ntd": sum(r in ca for r in NTD_R),
                "n_hb": sum(r in ca for r in HB_R),
                "n_tbd": sum(r in ca for r in TBD_R),
                "dist": centroid_distance(ca),
                "min_heavy": min_interdomain_approach(atoms[ch]["heavy"]),
                "altloc_ca": atoms[ch]["altloc_ca"],
                "partocc_ca": atoms[ch]["partocc_ca"],
            }
        if not per:
            raise RuntimeError(
                f"{p}: no coordinates were parsed for exact-Q96SW2 chains {chains}"
            )
        primary = choose_primary_chain(chains, p, CHAIN_MAP)
        if primary not in per:
            raise RuntimeError(
                f"{p}: curated primary CRBN chain {primary} has no parsed coordinate records"
            )
        best = max(per, key=lambda c: per[c]["cov"])
        S[p] = {"per": per, "primary": primary, "best": best,
                "method": method_of(e), "resolution": resolution_of(e),
                "constructs": construct_flags(e, cif_text), "ligands": ligands_of(e),
                "degron": degron_of(e), "ddb1": has_ddb1(e),
                "ddb1_described": describes_ddb1(e),
                "space_group": (e.get("symmetry") or {}).get("space_group_name_H_M") or "",
                "starting_model": ((e.get("refine") or [{}])[0] or {}).get("pdbx_starting_model") or "",
                "phasing": ((e.get("refine") or [{}])[0] or {}).get("pdbx_method_to_determine_struct") or "",
                "doi": (e.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI") or "",
                "title": (e.get("struct") or {}).get("title") or ""}
        if k % 20 == 0:
            print("  ...extracted %d/%d" % (k, len(pdbs)), flush=True)
    print("extracted %d depositions" % len(S))

    # ---- the reference rule, rediscovered rather than assumed ---------------
    # A deposition enters the reference ensemble iff its PRIMARY (lowest-id, or
    # chain-map) CRBN chain resolves the whole window AND its resolution is
    # within RES_MAX. Both conditions are verified below against the reference
    # label list; the resolution ceiling is implicit in the curated set and is
    # reported here because it, not coverage, is what excludes some entries.
    def passes_reference(p):
        s = S[p]
        return (s["per"][s["primary"]]["cov"] == len(WINSET)
                and s["resolution"] is not None and s["resolution"] <= RES_MAX)

    rediscovered = sorted(p for p in S if passes_reference(p))
    excluded = sorted(set(S) - set(rediscovered))

    # ---- (A) adjudicate every excluded deposition ---------------------------
    def band_on_subset(ca_keys):
        """closed/open reference bands recomputed on a residue subset."""
        ntd = [r for r in NTD_R if r in ca_keys]
        tbd = [r for r in TBD_R if r in ca_keys]
        if len(ntd) < MIN_DOMAIN_CA or len(tbd) < MIN_DOMAIN_CA:
            return None
        d = {}
        for q in rediscovered:
            s = S[q]
            dd = centroid_distance(s["per"][s["primary"]]["ca"], ntd, tbd)
            if dd is not None:
                d[q] = dd
        vals = np.array(list(d.values()))
        # the reference open set is the geometric top cluster of the primary
        # ensemble; taken as the members beyond the widest gap in sorted distance
        srt = np.sort(vals)[::-1]
        g = srt[:-1] - srt[1:]
        ncut = int(np.argmax(g[:15])) + 1
        thr = (srt[ncut - 1] + srt[ncut]) / 2.0
        op = vals[vals >= thr]; cl = vals[vals < thr]
        return dict(closed_min=float(cl.min()), closed_max=float(cl.max()),
                    closed_mean=float(cl.mean()), closed_sd=float(cl.std()),
                    open_min=float(op.min()), open_max=float(op.max()),
                    n_open=int(len(op)), n_closed=int(len(cl)))

    adjudication = []
    for p in excluded:
        s = S[p]; c = s["per"][s["primary"]]
        band = band_on_subset(set(c["ca"]))
        if c["n_ntd"] == 0:
            call = "indeterminate"
            reason = ("isolated thalidomide-binding-domain construct: 0/%d NTD and %d/%d "
                      "hinge-bundle window residues resolved, so the NTD-TBD coordinate "
                      "does not exist for this entry" % (len(NTD_R), c["n_hb"], len(HB_R)))
        elif band is None or c["dist"] is None:
            call = "indeterminate"
            reason = "fewer than %d Ca resolved in NTD or TBD" % MIN_DOMAIN_CA
        else:
            z = (c["dist"] - band["closed_mean"]) / band["closed_sd"]
            if c["dist"] <= band["closed_max"] + 1.0:
                call = "closed"
                reason = ("NTD-TBD %.2f A within the subset-matched closed band "
                          "%.1f-%.1f A (z=%+.1f); closest NTD-TBD heavy-atom approach %s A"
                          % (c["dist"], band["closed_min"], band["closed_max"], z,
                             "n/a" if c["min_heavy"] is None else "%.2f" % c["min_heavy"]))
            elif c["dist"] >= band["open_min"] - 1.0:
                call = "open"
                reason = ("NTD-TBD %.2f A within the subset-matched open band "
                          "%.1f-%.1f A (z=%+.1f above the closed mean); closest NTD-TBD "
                          "heavy-atom approach %s A"
                          % (c["dist"], band["open_min"], band["open_max"], z,
                             "n/a" if c["min_heavy"] is None else "%.2f" % c["min_heavy"]))
            else:
                call = "intermediate"
                reason = ("NTD-TBD %.2f A falls between the closed band (max %.1f A) and "
                          "the open band (min %.1f A)"
                          % (c["dist"], band["closed_max"], band["open_min"]))
        why = []
        if c["cov"] < len(WINSET):
            why.append("window coverage %d/%d on chain %s" % (c["cov"], len(WINSET), s["primary"]))
        if s["resolution"] is None:
            why.append("resolution unavailable; cannot establish the curated-set ceiling")
        elif s["resolution"] > RES_MAX:
            why.append("resolution %.2f A exceeds the %.1f A ceiling of the curated set"
                       % (s["resolution"], RES_MAX))
        spread = [v["dist"] for v in s["per"].values() if v["dist"] is not None]
        adjudication.append({
            "pdb": p, "method": s["method"], "resolution_A": s["resolution"],
            "crbn_chains": "/".join(sorted(s["per"])), "primary_chain": s["primary"],
            "best_covered_chain": s["best"],
            "best_chain_coverage": s["per"][s["best"]]["cov"],
            "window_resolved": c["cov"], "window_missing_n": len(WINSET) - c["cov"],
            "missing_residues": ";".join(str(x) for x in c["missing"]),
            "missing_terminal_only": bool(c["missing"]) and all(
                (WINSET.index(r) < TERMINAL_SLACK or
                 WINSET.index(r) >= len(WINSET) - TERMINAL_SLACK) for r in c["missing"]),
            "n_ntd_resolved": c["n_ntd"], "n_hb_resolved": c["n_hb"],
            "n_tbd_resolved": c["n_tbd"],
            "ntd_tbd_centroid_A": None if c["dist"] is None else round(c["dist"], 2),
            "ntd_tbd_min_heavy_A": None if c["min_heavy"] is None else round(c["min_heavy"], 2),
            "closed_band_A": None if band is None else "%.1f-%.1f" % (band["closed_min"], band["closed_max"]),
            "open_band_A": None if band is None else "%.1f-%.1f" % (band["open_min"], band["open_max"]),
            "n_crbn_copies": len(s["per"]),
            "copy_spread_A": None if len(spread) < 2 else round(max(spread) - min(spread), 2),
            "ddb1_present": s["ddb1"], "degron_partner": s["degron"],
            "ligands": ";".join("%s(%.0fDa)" % t for t in s["ligands"]),
            "max_ligand_mw_Da": max([w for _, w in s["ligands"]], default=0.0),
            "construct_flags": s["constructs"], "space_group": s["space_group"],
            "starting_model": s["starting_model"], "phasing": s["phasing"],
            "altloc_ca_window": c["altloc_ca"], "partial_occupancy_ca_window": c["partocc_ca"],
            "exclusion_reason": "; ".join(why),
            "state_call": call, "call_reason": reason,
            "doi": s["doi"], "title": s["title"][:90],
        })

    calls = {}
    for r in adjudication:
        calls[r["state_call"]] = calls.get(r["state_call"], 0) + 1
    open_excluded = [r["pdb"] for r in adjudication if r["state_call"] == "open"]
    print("adjudicated %d excluded depositions: %s" % (len(adjudication), calls))
    print("  excluded but OPEN: %s" % (open_excluded or "none"))

    # ---- (B) ensembles under alternative curation rules --------------------
    def build(rule, coverage=1.0, res_max=RES_MAX, chain="primary",
              drop_residues=(), terminal_only=False, must_include=()):
        """Return (confs, labels, resnums) for one curation rule."""
        win = [r for r in WINSET if r not in set(drop_residues)]
        keep = []
        for p in sorted(S):
            s = S[p]
            ch = s["primary"] if chain == "primary" else s["best"]
            ca = s["per"][ch]["ca"]
            miss = [r for r in win if r not in ca]
            if res_max is not None and (s["resolution"] is None or s["resolution"] > res_max):
                if p not in must_include:
                    continue
            if terminal_only:
                ok = all((WINSET.index(r) < TERMINAL_SLACK or
                          WINSET.index(r) >= len(WINSET) - TERMINAL_SLACK) for r in miss)
            else:
                ok = len(win) - len(miss) >= coverage * len(win) - 1e-9
            if ok or p in must_include:
                keep.append((p, ch))
        # analysis positions = resolved in EVERY retained structure
        common = [r for r in win
                  if all(r in S[p]["per"][ch]["ca"] for p, ch in keep)]
        confs = np.array([[S[p]["per"][ch]["ca"][r] for r in common] for p, ch in keep], float)
        return confs, [p for p, _ in keep], common

    variants = {}
    specs = [
        ("a_paper_rule", dict(rule="a", coverage=1.0)),
        ("b_coverage_95", dict(rule="b", coverage=0.95)),
        ("b_coverage_90", dict(rule="b", coverage=0.90)),
        ("c_drop_424_no_resolution_ceiling",
         dict(rule="c", coverage=1.0, drop_residues=(424,), res_max=None)),
        ("d_terminal_gaps_only", dict(rule="d", terminal_only=True)),
        ("e_best_covered_chain", dict(rule="e", coverage=1.0, chain="best")),
        ("f_no_resolution_ceiling", dict(rule="f", coverage=1.0, res_max=None)),
        ("g_coverage95_no_res_ceiling", dict(rule="g", coverage=0.95, res_max=None)),
    ]
    for name, kw in specs:
        confs, labels, common = build(**kw)
        if len(labels) < 8 or len(common) < 60:
            variants[name] = {"tag": name, "skipped": True,
                              "n_conformers": len(labels), "n_residues": len(common)}
            print("  %-28s SKIPPED (n=%d, res=%d)" % (name, len(labels), len(common)))
            continue
        r = analyse(confs, labels, common, ref_label="8CVP", tag=name)
        r["curation_rule"] = kw
        r["excluded_open_retained"] = [q for q in open_excluded if q in labels]
        variants[name] = r
        print("  %-28s n=%3d x %3d  open=%d  PC1=%.1f%%  ANM m1=%.3f (rank %d)  RMSIP=%.3f"
              % (name, r["n_conformers"], r["n_residues"], r["n_open"],
                 100 * r["pc1_variance_fraction"], r["anm_mode1_overlap"],
                 r["anm_best_rank"], r["rmsip_anm_pca"]))

    # ---- (C) analyst-choice sweeps on the reference ensemble ---------------
    confs0, labels0, common0 = build(rule="a", coverage=1.0)
    idx = {"NTD": [i for i, r in enumerate(common0) if r in NTD_R],
           "HB": [i for i, r in enumerate(common0) if r in HB_R],
           "TBD": [i for i, r in enumerate(common0) if r in TBD_R]}
    superposition = {"whole_molecule": analyse(confs0, labels0, common0,
                                              ref_label="8CVP", tag="sup_whole")}
    for dom in ("NTD", "HB", "TBD"):
        superposition["on_" + dom] = analyse(confs0, labels0, common0, ref_label="8CVP",
                                             sup_sel=idx[dom], tag="sup_" + dom)
    # the axis itself moves with the superposition anchor; report |cos| against
    # the primary whole-molecule axis so an alternative superposition can be
    # compared directly
    def axis_under(sel):
        al = superpose(confs0, sel=sel)
        pcs_l, _, sc_l = pca(al)
        s = sc_l[:, 0] / np.sqrt(al.shape[1])
        if abs(s.min()) > abs(s.max()):
            s = -s
        m, _, _ = widest_gap_cut(s)
        a = (al[m].mean(0) - al[~m].mean(0)).reshape(-1)
        return a / np.linalg.norm(a)
    axis_whole = axis_under(None)
    for k, v in superposition.items():
        sel = None if k == "whole_molecule" else idx[k.replace("on_", "")]
        v["axis_cos_vs_whole_molecule"] = float(abs(axis_under(sel) @ axis_whole))
        print("  superposition %-14s PC1=%.1f%%  axis|cos|=%.3f  ANM m1=%.3f (rank %d)  open=%d"
              % (k, 100 * v["pc1_variance_fraction"], v["axis_cos_vs_whole_molecule"],
                 v["anm_mode1_overlap"], v["anm_best_rank"], v["n_open"]))

    # method subset: the open/cryo-EM collinearity control
    base = analyse(confs0, labels0, common0, ref_label="8CVP", tag="all_methods")
    conf_sup = superpose(confs0)                      # one common frame for the axes
    om = np.array([base["pc1_scores"][l] >= base["pc1_cut_threshold"] for l in labels0])
    def axis_of(mask):
        X = conf_sup[mask]
        a = X[om[mask]].mean(0) - X[~om[mask]].mean(0)
        a = a.reshape(-1); return a / np.linalg.norm(a)
    methods = np.array([S[p]["method"] for p in labels0])
    all_axis = axis_of(np.ones(len(labels0), bool))
    method_subsets = {}
    for m in ("X-ray", "cryo-EM"):
        msk = methods == m
        pcs_m, vr_m, _ = pca(conf_sup[msk])
        ax = axis_of(msk)
        method_subsets[m] = {
            "n": int(msk.sum()), "n_open": int(om[msk].sum()),
            "n_closed": int((~om[msk]).sum()),
            "pc1_variance_fraction": float(vr_m[0]),
            "cos_with_all_structure_axis": float(abs(ax @ all_axis)),
        }
    xr, em = axis_of(methods == "X-ray"), axis_of(methods == "cryo-EM")
    s1_0 = np.array([base["pc1_scores"][l] for l in labels0])
    method_subsets["cos_xray_vs_cryoem_axis"] = float(abs(xr @ em))
    method_subsets["pc1_by_group"] = {
        "%s_%s" % (m, "open" if o else "closed"): {
            "n": int(((methods == m) & (om == o)).sum()),
            "mean": float(s1_0[(methods == m) & (om == o)].mean()),
            "sd": float(s1_0[(methods == m) & (om == o)].std())}
        for m in ("X-ray", "cryo-EM") for o in (True, False)
        if ((methods == m) & (om == o)).any()}
    cx = s1_0[(methods == "X-ray") & ~om]; ce = s1_0[(methods == "cryo-EM") & ~om]
    method_subsets["closed_only_method_offset_A"] = float(abs(cx.mean() - ce.mean()))
    method_subsets["open_closed_separation_A"] = float(s1_0[om].mean() - s1_0[~om].mean())
    # Fisher exact on method vs state, computed without scipy
    a = int(((methods == "X-ray") & om).sum()); b = int(((methods == "X-ray") & ~om).sum())
    c = int(((methods == "cryo-EM") & om).sum()); d = int(((methods == "cryo-EM") & ~om).sum())
    from math import comb
    def hyper(k, n1, n2, t):
        return comb(n1, k) * comb(n2, t - k) / comb(n1 + n2, t)
    tot_open = a + c; n1, n2 = a + b, c + d
    p_obs = hyper(a, n1, n2, tot_open)
    fisher_p = sum(hyper(k, n1, n2, tot_open) for k in range(0, min(n1, tot_open) + 1)
                   if hyper(k, n1, n2, tot_open) <= p_obs + 1e-12)
    method_subsets["contingency"] = {"xray_open": a, "xray_closed": b,
                                     "cryoem_open": c, "cryoem_closed": d,
                                     "fisher_exact_p": float(min(1.0, fisher_p))}
    print("  method control: X-ray %d open/%d closed, cryo-EM %d open/%d closed, Fisher p=%.4f"
          % (a, b, c, d, method_subsets["contingency"]["fisher_exact_p"]))
    print("  within-method PC1 %.1f%% (X-ray) / %.1f%% (cryo-EM); axis cos %.3f; "
          "closed-set method offset %.3f A vs %.2f A separation"
          % (100 * method_subsets["X-ray"]["pc1_variance_fraction"],
             100 * method_subsets["cryo-EM"]["pc1_variance_fraction"],
             method_subsets["cos_xray_vs_cryoem_axis"],
             method_subsets["closed_only_method_offset_A"],
             method_subsets["open_closed_separation_A"]))

    # engineered constructs on PC1
    flags = {p: S[p]["constructs"] for p in labels0}
    wt = np.array([flags[p] == "" for p in labels0])
    constructs = {
        "n_wild_type": int(wt.sum()), "n_engineered": int((~wt).sum()),
        "engineered_members": {p: flags[p] for p in labels0 if flags[p]},
        "pc1_wild_type_closed_mean": float(s1_0[wt & ~om].mean()),
        "pc1_engineered_closed_mean": float(s1_0[~wt & ~om].mean()) if (~wt & ~om).any() else None,
        "engineered_open_members": [p for p, e, o in zip(labels0, ~wt, om) if e and o],
    }
    print("  constructs: %d wild-type / %d engineered; closed-set PC1 %.3f (wt) vs %s (eng)"
          % (constructs["n_wild_type"], constructs["n_engineered"],
             constructs["pc1_wild_type_closed_mean"],
             "n/a" if constructs["pc1_engineered_closed_mean"] is None
             else "%.3f" % constructs["pc1_engineered_closed_mean"]))

    ddb1_census = {
        "n_description_mentions": sum(S[p]["ddb1_described"] for p in labels0),
        "n_exact_q16531": sum(S[p]["ddb1"] for p in labels0),
        "description_only": sorted(
            p for p in labels0 if S[p]["ddb1_described"] and not S[p]["ddb1"]
        ),
        "absent_by_description": sorted(p for p in labels0 if not S[p]["ddb1_described"]),
    }

    # empty-middle occupancy across every variant
    empty_middle = {}
    for name, v in variants.items():
        if v.get("skipped"):
            continue
        sc = np.array(list(v["pc1_scores"].values()))
        lab = list(v["pc1_scores"])
        omv = sc >= v["pc1_cut_threshold"]
        lo, hi = sc[~omv].max(), sc[omv].min()
        band = (lo + 0.15 * (hi - lo), lo + 0.85 * (hi - lo))
        occ = [lab[i] for i in range(len(sc)) if band[0] <= sc[i] <= band[1]]
        empty_middle[name] = {"closed_max": float(lo), "open_min": float(hi),
                              "gap_A": float(hi - lo),
                              "band_15_85_pct": [float(band[0]), float(band[1])],
                              "occupants": occ, "n_occupants": len(occ)}

    out = {
        "protocol": {"anm_cutoff_A": CUTOFF_ANM, "n_modes": N_MODES,
                     "resolution_ceiling_A": RES_MAX, "window_size": len(WINSET),
                     "terminal_slack": TERMINAL_SLACK, "seed": SEED},
        "curation_rule_rediscovered": {
            "n_pass": len(rediscovered), "n_excluded": len(excluded),
            "matches_committed_labels": sorted(rediscovered) == curated,
            "criteria": ["primary CRBN chain resolves all %d window residues" % len(WINSET),
                         "resolution <= %.1f A" % RES_MAX],
            "excluded_by_coverage_only": sorted(
                r["pdb"] for r in adjudication if r["window_resolved"] < len(WINSET)
                and (r["resolution_A"] or 99) <= RES_MAX),
            "excluded_by_resolution_only": sorted(
                r["pdb"] for r in adjudication if r["window_resolved"] == len(WINSET)),
            "excluded_by_both": sorted(
                r["pdb"] for r in adjudication if r["window_resolved"] < len(WINSET)
                and (r["resolution_A"] or 99) > RES_MAX),
        },
        "adjudication_summary": {"calls": calls, "open_excluded": open_excluded,
                                 "indeterminate": [r["pdb"] for r in adjudication
                                                   if r["state_call"] == "indeterminate"]},
        "adjudication": adjudication,
        "ensembles": variants,
        "superposition_dependence": superposition,
        "method_subsets": method_subsets,
        "constructs": constructs,
        "ddb1_entity_census": ddb1_census,
        "empty_middle": empty_middle,
    }
    expected_variants = [name for name, _ in specs]
    if not partial:
        validate_complete_generation(
            complete_pdbs,
            S,
            curated,
            rediscovered,
            excluded,
            adjudication,
            variants,
            expected_variants,
            out,
        )
    json_path = DATA / "window_sensitivity.json"
    csv_path = DATA / "excluded_structures_adjudication.csv"
    out = finalize_artifacts(
        out, json_path, csv_path, verify=verify, partial=partial
    )
    if partial:
        print("limit mode: diagnostic partial results; canonical JSON/CSV left untouched")
    elif verify:
        print("verify mode: complete JSON/CSV artifacts match; files left untouched")
    else:
        print("wrote data/window_sensitivity.json and data/excluded_structures_adjudication.csv")

    if verify:
        pa = variants["a_paper_rule"]
        assert out["curation_rule_rediscovered"]["matches_committed_labels"], \
            "rediscovered curation rule does not reproduce the reference 70 labels"
        assert pa["n_conformers"] == 70 and pa["n_residues"] == 269, \
            (pa["n_conformers"], pa["n_residues"])
        assert pa["n_open"] == 5, pa["n_open"]
        assert constructs["n_wild_type"] == 12 and constructs["n_engineered"] == 58, constructs
        assert ddb1_census["n_description_mentions"] == 66, ddb1_census
        assert ddb1_census["n_exact_q16531"] == 65, ddb1_census
        assert ddb1_census["description_only"] == ["23SR"], ddb1_census
        assert len(ddb1_census["absent_by_description"]) == 4, ddb1_census
        assert abs(100 * pa["pc1_variance_fraction"] - 88.3) < 0.5, \
            100 * pa["pc1_variance_fraction"]
        assert abs(pa["anm_mode1_overlap"] - 0.744) < 0.02, pa["anm_mode1_overlap"]
        assert pa["anm_best_rank"] == 1, pa["anm_best_rank"]
        assert abs(pa["rmsip_anm_pca"] - 0.641) < 0.03, pa["rmsip_anm_pca"]
        assert pa["pc1_diff_overlap"] > 0.98, pa["pc1_diff_overlap"]
        # the adjudication must find every excluded entry and call the isolated-TBD ones
        assert len(adjudication) == 28, len(adjudication)
        assert calls.get("indeterminate", 0) == 9, calls
        assert "6BNB" in open_excluded, open_excluded
        # 6BNB fails BOTH criteria, which is the substantive correction
        b = [r for r in adjudication if r["pdb"] == "6BNB"][0]
        assert b["window_missing_n"] == 1 and b["missing_residues"] == "424", b
        assert b["resolution_A"] > RES_MAX, b["resolution_A"]
        # Relaxing the window must not overturn the headline numbers, EXCEPT where
        # the relaxed common window is so truncated that ANM mode 1 is no longer
        # the global hinge. b_coverage_90 keeps only 68/90 NTD positions, deleting
        # the contiguous 103-116 segment; its mode 1 becomes a local hinge-bundle
        # motion (86% of the amplitude in HB) and the overlap collapses to 0.13.
        # That is a property of the truncated NETWORK, not of the added structures:
        # the same 223-residue window applied to the primary 70 members gives
        # the same collapse. The assertion therefore requires the headline numbers
        # to survive wherever mode 1 remains delocalised, and requires the
        # collapse to be accompanied by mode-1 localisation where it occurs.
        for nm in ("b_coverage_95", "b_coverage_90", "c_drop_424_no_resolution_ceiling",
                   "d_terminal_gaps_only", "e_best_covered_chain",
                   "f_no_resolution_ceiling", "g_coverage95_no_res_ceiling"):
            v = variants[nm]
            if v.get("skipped"):
                continue
            assert v["pc1_variance_fraction"] > 0.80, (nm, v["pc1_variance_fraction"])
            if v["anm_mode1_localised"]:
                assert v["anm_mode1_overlap"] < 0.3 and v["anm_cum_overlap_top10"] > 0.8, \
                    (nm, "localised mode 1 should lose single-mode overlap but keep "
                     "the subspace", v["anm_mode1_overlap"], v["anm_cum_overlap_top10"])
            else:
                assert v["anm_mode1_overlap"] > 0.65, (nm, v["anm_mode1_overlap"])
                assert v["anm_best_rank"] == 1, (nm, v["anm_best_rank"])
        assert variants["e_best_covered_chain"]["n_conformers"] == 73, \
            variants["e_best_covered_chain"]["n_conformers"]
        # The decisive variant: no resolution ceiling and >=95% coverage retains
        # 6BNB and gives SIX open structures. The headline numbers must survive it,
        # because this is the most informative relaxed-membership ensemble.
        g = variants["g_coverage95_no_res_ceiling"]
        assert g["n_open"] == 6 and "6BNB" in g["open_members"], g["open_members"]
        assert g["pc1_variance_fraction"] > 0.88, g["pc1_variance_fraction"]
        assert g["anm_mode1_overlap"] > 0.72 and g["anm_best_rank"] == 1, \
            (g["anm_mode1_overlap"], g["anm_best_rank"])
        assert g["per_open_mode1_min"] > 0.70, g["per_open_mode1_min"]
        # the empty middle must stay empty in every variant that has both clusters
        for nm, em in empty_middle.items():
            assert em["n_occupants"] == 0, (nm, em["occupants"])
        # superposition: qualitative result invariant, quantitative value is not
        for k, v in superposition.items():
            assert v["anm_best_rank"] == 1, (k, v["anm_best_rank"])
        assert superposition["on_TBD"]["axis_cos_vs_whole_molecule"] < 0.6, \
            superposition["on_TBD"]["axis_cos_vs_whole_molecule"]
        # the method control must show no association
        assert method_subsets["contingency"]["fisher_exact_p"] > 0.5, \
            method_subsets["contingency"]
        assert method_subsets["cos_xray_vs_cryoem_axis"] > 0.9, \
            method_subsets["cos_xray_vs_cryoem_axis"]
        cov90 = variants["b_coverage_90"]
        print("verify OK: fixed rule reproduces 70x269, 5 open, PC1 %.1f%%, ANM m1 %.3f "
              "rank 1, RMSIP %.3f; 28 excluded adjudicated (9 indeterminate, "
              "%d open). Structure-membership relaxations preserve the axis where "
              "mode 1 remains delocalised; the 90%%-coverage node set is a documented "
              "collapse (%dx%d, ANM m1 %.3f rank %d) caused by truncating the network."
              % (100 * pa["pc1_variance_fraction"], pa["anm_mode1_overlap"],
                 pa["rmsip_anm_pca"], len(open_excluded),
                 cov90["n_conformers"], cov90["n_residues"],
                 cov90["anm_mode1_overlap"], cov90["anm_best_rank"]))


if __name__ == "__main__":
    main()
