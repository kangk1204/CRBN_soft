#!/usr/bin/env python3
# Make Angstrom (Å) and other non-ASCII output safe on legacy consoles
# (e.g. Korean cp949 Windows) without changing values.
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
"""Rebuild the 70 x 269 x 3 Cα coordinate tensor END-TO-END from raw mmCIF.

This is the step that makes the "complete, reproducible ensemble" claim real:
it does not trust the committed tensor. For every curated PDB it downloads the
mmCIF from RCSB, selects the CRBN (Q96SW2) chain, extracts the 269-residue
analysis window (author residues from data/crbn_residue_window.csv), and iteratively
superposes all conformers onto their running mean. The rebuilt tensor is then
compared to the committed data/crbn_ensemble.ens.npz.

What this script does and does not verify. It rebuilds and superposes the COORDINATES for
a given entry list; the entry list itself is derived from the curation criteria by
scripts/window_sensitivity.py, not here. A median Ca RMSD of 0.000 A is therefore evidence
that the coordinate extraction is faithful, not that the 98 -> 70 selection is correct.

Curation rule (per PDB):
  - one chain: among the auth chains carrying the CRBN entity (data/_rcsb_meta.json), the
    one resolving the most of the 269-residue window, ties broken by lowest chain id.
    Restricting to the CRBN entity matters: ranking ALL chains lets DDB1 win in entries
    where it occupies chain A.
  - require the full 269-residue window to be resolved (Cα present); a PDB whose
    best chain is missing any window residue is dropped from the analysis set.
  - superpose: iterative Kabsch onto the running mean until convergence.

Usage:  python scripts/reproduce_tensor.py [--verify] [--limit N]
        python scripts/reproduce_tensor.py --unsafe-allow-ambiguous-chain [...]
"""
import os, sys, json, csv, urllib.request, gzip
import numpy as np

try:
    from pdb_id import validate_pdb_id
except ModuleNotFoundError:  # imported by path from the repository root
    from scripts.pdb_id import validate_pdb_id

# 269-residue analysis window: committed plain-text input written by
# reproduce_ensemble.py --write-window. Reading it here keeps this script independent
# of both ProDy and the mode artifact it is meant to validate.
WIN = np.array([int(r["author_resnum"]) for r in
                csv.DictReader(open("data/crbn_residue_window.csv"))]).astype(int)
WINSET = list(WIN)
CACHE = "data/_cif_cache"
CACHE_WRITES_ENABLED = True
UNSAFE_ALLOW_AMBIGUOUS_CHAIN = "--unsafe-allow-ambiguous-chain" in sys.argv

# Explicit CRBN chain overrides. These are the 14 depositions where the
# content-based rule below (chain resolving the most window residues, ties broken
# by lowest chain id) would pick a partner chain instead of CRBN -- e.g. 8CVP,
# where DDB1 (chain A) also resolves all 269 window positions. This is NOT the
# list of entries whose CRBN chain is not chain A: CRBN is at chain B or C in
# most of the 70 depositions, but the content-based rule already gets those
# right. Loaded from the sidecar if present.
try:
    CHAIN_MAP = json.load(open("data/curation_chain_map.json"))
except Exception:
    CHAIN_MAP = {}

# Committed RCSB entity metadata: which auth chains carry the CRBN entity. This is what
# makes chain selection safe without the sidecar; see crbn_chains().
try:
    RCSB_META = json.load(open("data/_rcsb_meta.json", encoding="utf-8"))
except Exception:
    RCSB_META = {}

def fetch_cif(pdb):
    pdb = validate_pdb_id(pdb)
    p = f"{CACHE}/{pdb}.cif.gz"
    if os.path.exists(p):
        try:
            with gzip.open(p, "rt") as fh:
                return fh.read()
        except (OSError, EOFError, UnicodeDecodeError):
            # A corrupt local entry is replaced during generation. Verification
            # fetches a clean copy in memory without deleting or rewriting it.
            pass
    with urllib.request.urlopen(
            f"https://files.rcsb.org/download/{pdb}.cif.gz", timeout=120) as fh:
        blob = fh.read()
    text = gzip.decompress(blob).decode("utf-8")
    if CACHE_WRITES_ENABLED:
        os.makedirs(CACHE, exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(blob)
    return text

def parse_ca(cif):
    """Return {chain_id: {author_resnum: (x,y,z)}} for Cα atoms, and a map
    chain->uniprot accession via struct_ref if available."""
    lines = cif.splitlines()
    # locate atom_site loop
    ca = {}
    # find the atom_site column order
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            # peek headers
            j = i + 1
            hdr = []
            while j < len(lines) and lines[j].lstrip().startswith("_atom_site."):
                hdr.append(lines[j].strip()); j += 1
            if hdr:
                col = {h.split(".")[1]: k for k, h in enumerate(hdr)}
                need = ["label_atom_id", "auth_asym_id", "auth_seq_id",
                        "Cartn_x", "Cartn_y", "Cartn_z", "group_PDB"]
                if all(c in col for c in need):
                    k = j
                    while k < len(lines):
                        ln = lines[k]
                        if ln.startswith("#") or ln.strip() == "" or ln.strip() == "loop_":
                            break
                        if ln.startswith("_"):
                            break
                        f = ln.split()
                        if len(f) < len(hdr):
                            k += 1; continue
                        if f[col["group_PDB"]] != "ATOM":
                            k += 1; continue
                        if f[col["label_atom_id"]].strip('"') != "CA":
                            k += 1; continue
                        ch = f[col["auth_asym_id"]]
                        try:
                            rs = int(f[col["auth_seq_id"]])
                            xyz = (float(f[col["Cartn_x"]]), float(f[col["Cartn_y"]]),
                                   float(f[col["Cartn_z"]]))
                        except ValueError:
                            k += 1; continue
                        ca.setdefault(ch, {})
                        if rs not in ca[ch]:
                            ca[ch][rs] = xyz
                        k += 1
                    i = k; continue
        i += 1
    return ca

def crbn_chains(pdb):
    """Auth chain ids of the CRBN entity, from the committed RCSB entity metadata.

    Ranking ALL chains by window coverage is unsafe: in an entry where DDB1 occupies chain A
    it can resolve more of a numerically overlapping range than CRBN does, and the fallback
    silently returns the wrong protein. With the chain map emptied, that rule picks chain A
    of 8CVP -- DDB1 -- and yields a 269-residue "conformer" 20.7 A from the committed one.
    Restricting the candidates to the CRBN entity removes the failure mode rather than
    patching it per entry.
    """
    e = RCSB_META.get(pdb.upper())
    if not e:
        return None
    out = []
    for pe in e.get("polymer_entities") or []:
        blob = json.dumps(pe).lower()
        if "cereblon" in blob or "q96sw2" in blob:
            ids = (pe.get("rcsb_polymer_entity_container_identifiers") or {}).get(
                "auth_asym_ids") or []
            out += list(ids)
    return sorted(out) or None


def best_chain(ca, pdb=None):
    """CRBN chain resolving the most window residues (tie: lowest id)."""
    candidates = crbn_chains(pdb) if pdb else None
    if candidates is None:
        if UNSAFE_ALLOW_AMBIGUOUS_CHAIN:
            pool = sorted(ca)
            print(f"WARNING: {pdb or '<unknown>'} has no committed CRBN chain metadata; "
                  "using all chains because --unsafe-allow-ambiguous-chain was passed",
                  file=sys.stderr)
        else:
            raise RuntimeError(f"{pdb or '<unknown>'}: missing CRBN chain metadata; refusing "
                               "to rank all chains. Pass --unsafe-allow-ambiguous-chain to "
                               "use the historical fail-open fallback.")
    else:
        pool = [c for c in sorted(ca) if c in candidates]
        if not pool:
            if UNSAFE_ALLOW_AMBIGUOUS_CHAIN:
                pool = sorted(ca)
                print(f"WARNING: {pdb or '<unknown>'} metadata lists CRBN chains "
                      f"{candidates}, but none were found in coordinates; using all "
                      "chains because --unsafe-allow-ambiguous-chain was passed",
                      file=sys.stderr)
            else:
                raise RuntimeError(f"{pdb or '<unknown>'}: CRBN chain metadata lists "
                                   f"{candidates}, but none are present in the parsed "
                                   "coordinates; refusing ambiguous fallback.")
    best, best_n = None, -1
    for ch in pool:
        n = sum(r in ca[ch] for r in WINSET)
        if n > best_n:
            best, best_n = ch, n
    return best, best_n


def select_chain(ca, pdb):
    """Apply a recorded primary-chain override without bypassing entity metadata."""
    if pdb not in CHAIN_MAP:
        return best_chain(ca, pdb)

    chosen = CHAIN_MAP[pdb]
    candidates = crbn_chains(pdb)
    problems = []
    if candidates is None:
        problems.append("missing CRBN chain metadata")
    elif chosen not in candidates:
        problems.append(f"recorded chain {chosen} is not among metadata CRBN chains {candidates}")
    if chosen not in ca:
        problems.append(f"recorded chain {chosen} is absent from parsed coordinates")

    if problems:
        detail = "; ".join(problems)
        if not UNSAFE_ALLOW_AMBIGUOUS_CHAIN:
            raise RuntimeError(f"{pdb}: {detail}; refusing to bypass chain provenance")
        print(f"WARNING: {pdb}: {detail}; using the historical all-chain fallback because "
              "--unsafe-allow-ambiguous-chain was passed", file=sys.stderr)
        return best_chain(ca, pdb)

    n = sum(r in ca[chosen] for r in WINSET)
    return chosen, n

def extract(pdb):
    ca = parse_ca(fetch_cif(pdb))
    if not ca:
        return None, 0
    ch, n = select_chain(ca, pdb)
    if n < len(WINSET):
        return None, n           # window not fully resolved -> drop
    coords = np.array([ca[ch][r] for r in WINSET], float)
    return coords, n

def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return (R @ Pc.T).T

def superpose(confs, iters=8):
    ref = confs[0] - confs[0].mean(0)
    for _ in range(iters):
        al = np.array([kabsch(c, ref) for c in confs])
        newref = al.mean(0)
        if np.allclose(newref, ref, atol=1e-4):
            ref = newref; confs = al; break
        ref, confs = newref, al
    return al, ref

def main():
    global CACHE_WRITES_ENABLED
    verify = "--verify" in sys.argv
    CACHE_WRITES_ENABLED = not verify
    ens = np.load("data/crbn_ensemble.ens.npz", allow_pickle=False)
    labels = [str(x) for x in ens["_labels"]]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    todo = labels[:limit] if limit else labels

    # Two distinct failure classes, kept apart: `dropped` = genuine curation
    # exclusion (window not fully resolved), `failed` = download/gzip/parse
    # error, which says nothing about the structure and must not silently
    # shrink the analysis set.
    coords, ok, dropped, failed = [], [], [], []
    for n, pdb in enumerate(todo, 1):
        try:
            c, nres = extract(pdb)
        except Exception as e:
            failed.append((pdb, f"{type(e).__name__}: {e}")); continue
        if c is None:
            dropped.append((pdb, f"window {nres}/{len(WINSET)}")); continue
        coords.append(c); ok.append(pdb)
        if n % 10 == 0:
            print(f"  ...{n}/{len(todo)}", flush=True)
    print(f"extracted {len(ok)} conformers; dropped {len(dropped)}: {dropped[:8]}")
    if failed:
        print(f"fetch/parse errors on {len(failed)}: {failed[:8]}", flush=True)
        if verify:
            sys.exit(f"verify aborted: {len(failed)} deposition(s) could not be "
                     f"fetched or parsed: {failed}")

    C = np.array(coords)
    al, ref = superpose(C)
    if verify:
        print(f"rebuilt tensor {al.shape} in memory; verify mode left output files untouched")
    else:
        np.savez("data/crbn_ensemble_rebuilt.npz",
                 confs=al.astype(np.float32), labels=np.array(ok), ref=ref.astype(np.float32))
        print(f"rebuilt tensor {al.shape} -> data/crbn_ensemble_rebuilt.npz")

    # Compare to committed tensor on shared labels (Kabsch-align rebuilt onto committed)
    comm = ens["_confs"]; clab = labels
    shared = [p for p in ok if p in clab]
    rms = []
    for p in shared:
        a = al[ok.index(p)]
        b = comm[clab.index(p)]
        aa = kabsch(a, b - b.mean(0))
        rms.append(np.sqrt(((aa - (b - b.mean(0)))**2).sum(-1).mean()))
    rms = np.array(rms)
    print(f"rebuilt-vs-committed per-conformer Cα RMSD: "
          f"median {np.median(rms):.3f} Å, max {rms.max():.3f} Å, n={len(shared)}")

    if verify:
        # every requested deposition must rebuild; no silent shrinkage
        assert len(ok) == len(todo), (f"expected {len(todo)} conformers, rebuilt "
                                      f"{len(ok)}; dropped {dropped}")
        assert np.median(rms) < 0.5, f"rebuilt tensor diverges: median {np.median(rms):.3f}"
        print(f"verify OK: {len(ok)} conformers rebuilt from raw mmCIF, "
              f"median Cα RMSD to committed {np.median(rms):.3f} Å (<0.5)")

if __name__ == "__main__":
    main()
