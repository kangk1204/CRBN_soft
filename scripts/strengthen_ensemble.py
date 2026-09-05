#!/usr/bin/env python3
"""Strengthen the CRBN ensemble analysis without moving the frozen frame.

This workflow adds three reproducible checks around the fixed 70-conformer
analysis:

* per-reference ANM rankings for each frozen conformer;
* 5 x 65 open-closed pair rankings under fixed-8CVP and own-open bases;
* a live RCSB Q96SW2 inventory, with newer eligible structures scored on the
  frozen PC1 coordinate instead of refitting PCA.

Generated files default to results/strengthening/. Release builders can pass
explicit package paths. The committed data/ inputs are read only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.linalg import eigh as scipy_eigh
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
except Exception:  # pragma: no cover - exercised only without scipy
    scipy_eigh = None
    csr_matrix = None
    eigsh = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curation_contracts import (  # noqa: E402
    CRBN_ACCESSION,
    accession_deletion_range_groups,
    accession_range_groups,
)
from pdb_id import validate_pdb_id, validate_polymer_entity_id  # noqa: E402
from reproduce_modes import pca  # noqa: E402
from reproduce_tensor import parse_ca, superpose  # noqa: E402
from score_structure import classify, closure_score, load_reference  # noqa: E402
from softmode_lib import kabsch_apply  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "strengthening" / "ensemble"
DEFAULT_STRUCTURE_DIR = ROOT / "results" / "strengthening" / "data" / "structures"
FROZEN_QUERY = ROOT / "data" / "rcsb_query_Q96SW2.json"
CHAIN_MAP = ROOT / "data" / "curation_chain_map.json"
ENSEMBLE = ROOT / "data" / "crbn_ensemble.ens.npz"
PCA_DIFF = ROOT / "data" / "pca_diffvec.npz"
PCA_INPUT = ROOT / "data" / "crbn_pca.npz"
WINDOW = ROOT / "data" / "crbn_residue_window.csv"
ANM_CUTOFF = 15.0
MAX_MODES = 60
FROZEN_OPEN_THRESHOLD = 0.95
FROZEN_CLOSED_THRESHOLD = 0.25
USER_AGENT = "CRBN-softmode-strengthen-ensemble/1.0"
SEARCH_ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"
DOWNLOAD_ENDPOINT = "https://files.rcsb.org/download/{pdb}.cif.gz"

OUTPUT_FILES = {
    "query": "live_rcsb_query_Q96SW2.json",
    "metadata": "live_rcsb_metadata_Q96SW2.json",
    "inventory": "live_q96sw2_chain_inventory.csv",
    "scores": "newer_eligible_frozen_scores.csv",
    "per_reference": "per_reference_anm_rankings.csv",
    "pairs": "open_closed_pair_basis_comparison.csv",
    "temporal_modes": "temporal_mode_rankings.csv",
    "summary": "summary.json",
}


@dataclass(frozen=True)
class FrozenFrame:
    labels: list[str]
    conformers: np.ndarray
    open_mask: np.ndarray
    axis: np.ndarray
    window: np.ndarray
    frozen_entities: set[str]
    frozen_pdbs: set[str]
    freeze_date: date


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_window(path: Path = WINDOW) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        values = [int(row["author_resnum"]) for row in csv.DictReader(handle)]
    if len(values) != 269:
        raise ValueError(f"{path}: expected 269 residue positions, found {len(values)}")
    return np.asarray(values, dtype=int)


def load_frozen_frame() -> FrozenFrame:
    with np.load(ENSEMBLE, allow_pickle=False) as ensemble:
        conformers = np.asarray(ensemble["_confs"], dtype=float)
        labels = [validate_pdb_id(str(value)) for value in ensemble["_labels"]]
    with np.load(PCA_DIFF, allow_pickle=False) as diff:
        open_mask = np.asarray(diff["open_mask"], dtype=bool)
    if conformers.shape != (70, 269, 3):
        raise ValueError(f"{ENSEMBLE}: expected 70 x 269 x 3, found {conformers.shape}")
    if open_mask.shape != (70,) or int(open_mask.sum()) != 5:
        raise ValueError(f"{PCA_DIFF}: expected five frozen open labels")
    axis = (conformers[open_mask].mean(0) - conformers[~open_mask].mean(0)).reshape(-1)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 0:
        raise ValueError("frozen open-closed axis is degenerate")
    frozen_query = json.loads(FROZEN_QUERY.read_text(encoding="utf-8"))
    freeze = date.fromisoformat(str(frozen_query["freeze_date"]))
    entities = {str(entity).upper() for entity in frozen_query["entities"]}
    pdbs = {validate_polymer_entity_id(entity)[0] for entity in entities}
    return FrozenFrame(
        labels=labels,
        conformers=conformers,
        open_mask=open_mask,
        axis=axis / axis_norm,
        window=read_window(),
        frozen_entities=entities,
        frozen_pdbs=pdbs,
        freeze_date=freeze,
    )


def anm_hessian(coords: np.ndarray, cutoff: float = ANM_CUTOFF) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coordinates must have n x 3 shape, found {coords.shape}")
    delta = coords[:, None, :] - coords[None, :, :]
    r2 = np.sum(delta * delta, axis=2)
    pairs = np.where(np.triu((r2 <= cutoff * cutoff) & (r2 > 1e-12), 1))
    hessian = np.zeros((3 * len(coords), 3 * len(coords)), dtype=float)
    for i, j in zip(pairs[0], pairs[1]):
        block = np.outer(delta[i, j], delta[i, j]) / r2[i, j]
        si = slice(3 * i, 3 * i + 3)
        sj = slice(3 * j, 3 * j + 3)
        hessian[si, si] += block
        hessian[sj, sj] += block
        hessian[si, sj] -= block
        hessian[sj, si] -= block
    return hessian


def anm_modes(coords: np.ndarray, n_modes: int = MAX_MODES) -> tuple[np.ndarray, np.ndarray]:
    hessian = anm_hessian(coords)
    if eigsh is not None and csr_matrix is not None:
        # CRBN ANM has six rigid-body zero modes. Ask for a few extra smallest
        # eigenpairs, discard zeros, and retain the requested internal modes.
        k = min(hessian.shape[0] - 2, n_modes + 10)
        values, vectors = eigsh(csr_matrix(hessian), k=k, which="SM", tol=1e-8, maxiter=20000)
        order = np.argsort(values)
        values = values[order]
        vectors = vectors[:, order]
    elif scipy_eigh is not None:
        upper = min(hessian.shape[0] - 1, n_modes + 12)
        values, vectors = scipy_eigh(
            hessian, subset_by_index=(0, upper), check_finite=False, driver="evr"
        )
    else:
        values, vectors = np.linalg.eigh(hessian)
    keep = values > 1e-9
    values = values[keep][:n_modes]
    vectors = vectors[:, keep][:, :n_modes]
    if values.shape != (n_modes,) or vectors.shape != (hessian.shape[0], n_modes):
        raise ValueError(f"ANM produced only {len(values)} nonzero modes; need {n_modes}")
    return values, vectors


def overlap_metrics(axis: np.ndarray, values: np.ndarray, vectors: np.ndarray) -> dict[str, Any]:
    overlaps = np.abs(vectors.T @ axis)
    best20_index = int(np.argmax(overlaps[:20]))
    best60_index = int(np.argmax(overlaps[:60]))

    def local_gap(index: int) -> float:
        previous_gap = float(values[index] - values[index - 1]) if index > 0 else float("nan")
        next_gap = float(values[index + 1] - values[index]) if index + 1 < len(values) else float("nan")
        finite = [gap for gap in (previous_gap, next_gap) if np.isfinite(gap)]
        return min(finite) if finite else float("nan")

    result: dict[str, Any] = {
        "mode1_overlap": float(overlaps[0]),
        "mode1_eigenvalue": float(values[0]),
        "mode1_next_eigenvalue_gap": float(values[1] - values[0]),
        "best20_overlap": float(overlaps[best20_index]),
        "best20_rank": best20_index + 1,
        "best20_eigenvalue": float(values[best20_index]),
        "best20_local_eigenvalue_gap": local_gap(best20_index),
        "best60_overlap": float(overlaps[best60_index]),
        "best60_rank": best60_index + 1,
        "best60_eigenvalue": float(values[best60_index]),
        "best60_local_eigenvalue_gap": local_gap(best60_index),
        "top3_subspace_projection": float(np.sqrt(np.square(overlaps[:3]).sum())),
        "top10_subspace_projection": float(np.sqrt(np.square(overlaps[:10]).sum())),
        "top20_subspace_projection": float(np.sqrt(np.square(overlaps[:20]).sum())),
    }
    return result


def frozen_per_reference(
    frame: FrozenFrame,
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    mode_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, coords, is_open in zip(frame.labels, frame.conformers, frame.open_mask):
        values, vectors = anm_modes(coords)
        mode_cache[label] = (values, vectors)
        row = {
            "pdb": label,
            "state": "open" if bool(is_open) else "closed",
            "cutoff_A": f"{ANM_CUTOFF:.1f}",
            "n_modes": MAX_MODES,
        }
        row.update(overlap_metrics(frame.axis, values, vectors))
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"  frozen ANM references {len(rows)}/70", flush=True)
    return rows, mode_cache


def open_closed_pair_comparison(
    frame: FrozenFrame, mode_cache: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> list[dict[str, Any]]:
    open_indices = [index for index, flag in enumerate(frame.open_mask) if bool(flag)]
    closed_indices = [index for index, flag in enumerate(frame.open_mask) if not bool(flag)]
    fixed_values, fixed_vectors = mode_cache["8CVP"]
    rows: list[dict[str, Any]] = []
    for open_index in open_indices:
        for closed_index in closed_indices:
            pair_axis = (frame.conformers[open_index] - frame.conformers[closed_index]).reshape(-1)
            pair_axis /= np.linalg.norm(pair_axis)
            own_values, own_vectors = mode_cache[frame.labels[open_index]]
            for basis, values, vectors in (
                ("fixed_8CVP", fixed_values, fixed_vectors),
                ("own_open", own_values, own_vectors),
            ):
                row = {
                    "open_pdb": frame.labels[open_index],
                    "closed_pdb": frame.labels[closed_index],
                    "basis": basis,
                    "basis_pdb": "8CVP" if basis == "fixed_8CVP" else frame.labels[open_index],
                    "cutoff_A": f"{ANM_CUTOFF:.1f}",
                }
                row.update(overlap_metrics(pair_axis, values, vectors))
                rows.append(row)
    return rows


def align_to_frozen_frame(coords: np.ndarray) -> np.ndarray:
    _window, mean, _pc1, _closed_mean, _open_mean = load_reference()
    return kabsch_apply(np.asarray(coords, dtype=float), mean)


def temporal_mode_rankings(
    score_inputs: Mapping[str, tuple[str, np.ndarray, Mapping[str, Any]]],
    frame: FrozenFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id in sorted(score_inputs):
        chain, coords, inventory_row = score_inputs[entity_id]
        aligned = align_to_frozen_frame(coords)
        values, vectors = anm_modes(aligned)
        row = {
            "pdb_entity": entity_id,
            "pdb_id": inventory_row["pdb_id"],
            "chain": chain,
            "basis": "own_new_structure",
            "axis": "frozen_70_open_minus_closed",
            "cutoff_A": f"{ANM_CUTOFF:.1f}",
            "n_window_positions": int(aligned.shape[0]),
            "release_date": inventory_row["release_date"],
            "resolution_A": inventory_row["resolution_A"],
            "title": inventory_row["title"],
        }
        row.update(overlap_metrics(frame.axis, values, vectors))
        rows.append(row)
    return rows


def pca_refit_sensitivity(
    score_inputs: Mapping[str, tuple[str, np.ndarray, Mapping[str, Any]]],
    frame: FrozenFrame,
) -> dict[str, Any]:
    if not score_inputs:
        return {"n_conformers": 70, "note": "no newer eligible structures available"}
    new_ids = sorted(score_inputs)
    new_coords = np.asarray([score_inputs[entity_id][1] for entity_id in new_ids], dtype=float)
    combined, _reference = superpose(np.concatenate([frame.conformers, new_coords], axis=0))
    labels = frame.labels + [validate_polymer_entity_id(entity_id)[0] for entity_id in new_ids]
    _mean, pcv, _pcw, variance_ratio, scores = pca(combined)
    pc1 = pcv[:, 0]
    pc1_scores = scores[:, 0] / np.sqrt(combined.shape[1])
    if abs(float(pc1_scores.min())) > abs(float(pc1_scores.max())):
        pc1 = -pc1
        pc1_scores = -pc1_scores
    ordered = np.sort(pc1_scores)[::-1]
    split = int(np.argmax(ordered[:15][:-1] - ordered[:15][1:])) + 1
    threshold = float(0.5 * (ordered[split - 1] + ordered[split]))
    open_mask = pc1_scores >= threshold
    difference = combined[open_mask].mean(0) - combined[~open_mask].mean(0)
    axis = difference.reshape(-1)
    axis /= np.linalg.norm(axis)
    values, vectors = anm_modes(combined[labels.index("8CVP")])
    metrics = overlap_metrics(axis, values, vectors)
    return {
        "scope": (
            "sensitivity only: PCA refit after adding current newer eligible structures; "
            "frozen 70-coordinate scoring remains primary"
        ),
        "n_conformers": int(combined.shape[0]),
        "n_residues": int(combined.shape[1]),
        "new_entities_added": new_ids,
        "pc1_variance_fraction": float(variance_ratio[0]),
        "open_count": int(open_mask.sum()),
        "closed_count": int((~open_mask).sum()),
        "pc1_open_threshold": threshold,
        "open_members": sorted(label for label, flag in zip(labels, open_mask) if bool(flag)),
        "anm_8cvp_mode1_overlap": metrics["mode1_overlap"],
        "anm_8cvp_best20_overlap": metrics["best20_overlap"],
        "anm_8cvp_best20_rank": metrics["best20_rank"],
        "anm_8cvp_best60_overlap": metrics["best60_overlap"],
        "anm_8cvp_best60_rank": metrics["best60_rank"],
        "anm_8cvp_top10_subspace_projection": metrics["top10_subspace_projection"],
    }


def rcsb_search_body() -> dict[str, Any]:
    return {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers."
                    "reference_sequence_identifiers.database_accession"
                ),
                "operator": "exact_match",
                "value": CRBN_ACCESSION,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {"return_all_hits": True},
    }


def request_json(url: str, payload: Mapping[str, Any], timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def live_search() -> dict[str, Any]:
    body = rcsb_search_body()
    response = request_json(SEARCH_ENDPOINT, body)
    identifiers = sorted(str(row["identifier"]).upper() for row in response.get("result_set", []))
    for identifier in identifiers:
        validate_polymer_entity_id(identifier)
    return {
        "queried_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint_url": SEARCH_ENDPOINT,
        "replay_url": SEARCH_ENDPOINT + "?json=" + urllib.parse.quote(json.dumps(body, separators=(",", ":"))),
        "query": body,
        "query_sha256": sha256_bytes(stable_json_bytes(body)),
        "total_count": int(response.get("total_count", len(identifiers))),
        "entities": identifiers,
        "result_sha256": sha256_bytes(stable_json_bytes(identifiers)),
        "raw_response_sha256": sha256_bytes(stable_json_bytes(response)),
    }


GRAPHQL_QUERY = """
query($ids:[String!]!){
  polymer_entities(entity_ids:$ids){
    rcsb_id
    entity_poly{rcsb_sample_sequence_length}
    rcsb_polymer_entity{pdbx_description pdbx_mutation}
    rcsb_polymer_entity_container_identifiers{
      auth_asym_ids
      reference_sequence_identifiers{database_accession}
    }
    entry{
      rcsb_id
      struct{title}
      rcsb_accession_info{initial_release_date deposit_date}
      exptl{method}
      rcsb_entry_info{resolution_combined}
      refine{pdbx_method_to_determine_struct pdbx_starting_model}
      symmetry{space_group_name_H_M}
      rcsb_primary_citation{pdbx_database_id_DOI title}
      polymer_entities{
        entity_poly{rcsb_sample_sequence_length}
        rcsb_polymer_entity{pdbx_description pdbx_mutation}
        rcsb_polymer_entity_container_identifiers{
          auth_asym_ids
          reference_sequence_identifiers{database_accession}
        }
      }
      nonpolymer_entities{nonpolymer_comp{chem_comp{id formula_weight}}}
    }
  }
}
"""


def chunks(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def fetch_metadata(entity_ids: Sequence[str]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for batch in chunks(entity_ids, 50):
        response = request_json(GRAPHQL_ENDPOINT, {"query": GRAPHQL_QUERY, "variables": {"ids": batch}})
        if response.get("errors"):
            raise RuntimeError(f"RCSB GraphQL errors: {response['errors']}")
        for record in (response.get("data") or {}).get("polymer_entities") or []:
            records[str(record["rcsb_id"]).upper()] = record
    missing = sorted(set(entity_ids) - set(records))
    if missing:
        raise RuntimeError(f"RCSB GraphQL omitted entities: {missing}")
    return records


def download_cif(pdb: str, structure_dir: Path, *, offline: bool) -> tuple[Path | None, str | None]:
    pdb = validate_pdb_id(pdb)
    path = structure_dir / f"{pdb}.cif.gz"
    if path.is_file():
        return path, None
    if offline:
        return None, f"offline mode and no cached mmCIF at {path}"
    structure_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DOWNLOAD_ENDPOINT.format(pdb=pdb),
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        gzip.decompress(payload)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    path.write_bytes(payload)
    return path, None


def read_cif(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")


def iso_date(value: Any) -> str:
    if value in (None, "", "?"):
        return ""
    text = str(value)
    return text[:10]


def first_resolution(entry: Mapping[str, Any]) -> float | None:
    values = ((entry.get("rcsb_entry_info") or {}).get("resolution_combined") or [])
    numeric = [float(value) for value in values if value is not None]
    return min(numeric) if numeric else None


def methods(entry: Mapping[str, Any]) -> str:
    return ";".join(str(row.get("method", "")) for row in entry.get("exptl") or [] if row)


def ligand_ids(entry: Mapping[str, Any]) -> str:
    ids: list[str] = []
    for entity in entry.get("nonpolymer_entities") or []:
        chem = (((entity or {}).get("nonpolymer_comp") or {}).get("chem_comp") or {})
        ligand = chem.get("id")
        if ligand:
            ids.append(str(ligand))
    return ";".join(sorted(set(ids)))


def partner_descriptions(entry: Mapping[str, Any]) -> tuple[bool, str]:
    ddb1_present = False
    partners: list[str] = []
    for entity in entry.get("polymer_entities") or []:
        description = ((entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "")
        accessions = {
            str(record.get("database_accession", "")).upper()
            for record in (
                (entity.get("rcsb_polymer_entity_container_identifiers") or {}).get(
                    "reference_sequence_identifiers"
                )
                or []
            )
        }
        length = ((entity.get("entity_poly") or {}).get("rcsb_sample_sequence_length") or "")
        if CRBN_ACCESSION in accessions:
            continue
        if "Q16531" in accessions or "dna damage-binding protein 1" in description.lower():
            ddb1_present = True
            continue
        if description:
            partners.append(f"{description}({length}aa)" if length != "" else description)
    return ddb1_present, ";".join(partners)


def primary_chain(auth_chains: Sequence[str], pdb: str, overrides: Mapping[str, str]) -> str | None:
    chains = sorted(str(chain) for chain in auth_chains)
    if not chains:
        return None
    override = overrides.get(pdb.upper())
    if override in chains:
        return str(override)
    return chains[0]


def load_chain_map() -> dict[str, str]:
    if not CHAIN_MAP.is_file():
        return {}
    value = json.loads(CHAIN_MAP.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{CHAIN_MAP}: expected JSON object")
    return {str(key).upper(): str(val) for key, val in value.items()}


def selected_entity_accessions(record: Mapping[str, Any]) -> set[str]:
    identifiers = record.get("rcsb_polymer_entity_container_identifiers") or {}
    return {
        str(item.get("database_accession", "")).upper()
        for item in identifiers.get("reference_sequence_identifiers") or []
        if item.get("database_accession")
    }


def chain_inventory(
    query_record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    frame: FrozenFrame,
    structure_dir: Path,
    *,
    offline: bool,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, np.ndarray, dict[str, Any]]], list[dict[str, Any]]]:
    overrides = load_chain_map()
    rows: list[dict[str, Any]] = []
    score_inputs: dict[str, tuple[str, np.ndarray, dict[str, Any]]] = {}
    downloads: list[dict[str, Any]] = []
    unique_pdbs = sorted({validate_polymer_entity_id(entity_id)[0] for entity_id in query_record["entities"]})
    cif_texts: dict[str, str | None] = {}
    ca_by_pdb: dict[str, dict[str, dict[int, tuple[float, float, float]]]] = {}
    range_groups: dict[str, Any] = {}
    deletion_groups: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for pdb in unique_pdbs:
        path, error = download_cif(pdb, structure_dir, offline=offline)
        downloads.append(
            {
                "pdb": pdb,
                "path": "" if path is None else display_path(path),
                "compressed_sha256": "" if path is None else sha256_file(path),
                "error": error or "",
            }
        )
        if path is None:
            cif_texts[pdb] = None
            errors[pdb] = error or "mmCIF unavailable"
            continue
        try:
            text = read_cif(path)
            cif_texts[pdb] = text
            ca_by_pdb[pdb] = parse_ca(text)
            range_groups[pdb] = accession_range_groups(text, CRBN_ACCESSION, required=False)
            deletion_groups[pdb] = accession_deletion_range_groups(text, CRBN_ACCESSION)
        except Exception as exc:
            cif_texts[pdb] = None
            errors[pdb] = f"{type(exc).__name__}: {exc}"

    for entity_id in query_record["entities"]:
        pdb, _entity_number = validate_polymer_entity_id(entity_id)
        record = metadata[entity_id]
        entry = record.get("entry") or {}
        resolution = first_resolution(entry)
        auth_chains = [
            str(chain)
            for chain in (
                (record.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids")
                or []
            )
        ]
        selected = primary_chain(auth_chains, pdb, overrides)
        entity_accessions = selected_entity_accessions(record)
        exact_human = CRBN_ACCESSION in entity_accessions
        release_date = iso_date((entry.get("rcsb_accession_info") or {}).get("initial_release_date"))
        deposit_date = iso_date((entry.get("rcsb_accession_info") or {}).get("deposit_date"))
        is_new = entity_id not in frame.frozen_entities
        ddb1_present, partners = partner_descriptions(entry)
        chain_pool = auth_chains or [""]
        for chain in chain_pool:
            resolved = 0
            missing: list[int] = list(map(int, frame.window))
            if pdb in ca_by_pdb and chain in ca_by_pdb[pdb]:
                missing = [int(residue) for residue in frame.window if int(residue) not in ca_by_pdb[pdb][chain]]
                resolved = len(frame.window) - len(missing)
            reasons: list[str] = []
            if pdb in errors:
                reasons.append(errors[pdb])
            if not exact_human:
                reasons.append(f"entity is not exact {CRBN_ACCESSION}")
            if chain != selected:
                reasons.append("non-primary CRBN copy under the one-primary-chain curation rule")
            if resolved != len(frame.window):
                reasons.append(f"primary window coverage {resolved}/{len(frame.window)}")
            if resolution is None:
                reasons.append("no reported resolution")
            elif resolution > 4.0:
                reasons.append(f"resolution {resolution:.2f} A > 4.0 A")
            primary_eligible = (
                exact_human
                and chain == selected
                and resolved == len(frame.window)
                and resolution is not None
                and resolution <= 4.0
                and pdb not in errors
            )
            row = {
                "pdb_entity": entity_id,
                "pdb_id": pdb,
                "chain": chain,
                "is_primary_chain": int(chain == selected),
                "is_new_since_frozen_query": int(is_new),
                "release_date": release_date,
                "deposit_date": deposit_date,
                "seq_length": (record.get("entity_poly") or {}).get("rcsb_sample_sequence_length", ""),
                "method": methods(entry),
                "resolution_A": "" if resolution is None else f"{resolution:.3f}",
                "title": (entry.get("struct") or {}).get("title", ""),
                "primary_citation_doi": (entry.get("rcsb_primary_citation") or {}).get(
                    "pdbx_database_id_DOI", ""
                )
                or "",
                "mutation": (record.get("rcsb_polymer_entity") or {}).get("pdbx_mutation", "") or "",
                "q96sw2_auth_chains": ";".join(auth_chains),
                "q96sw2_mapping_groups": json.dumps(range_groups.get(pdb, []), separators=(",", ":")),
                "q96sw2_deletion_groups": json.dumps(deletion_groups.get(pdb, []), separators=(",", ":")),
                "n_window_resolved": resolved,
                "missing_residues": ";".join(str(value) for value in missing),
                "ligands": ligand_ids(entry),
                "ddb1_present": int(ddb1_present),
                "other_polymer_partners": partners,
                "eligible_primary_current": int(primary_eligible),
                "eligible_newer_for_frozen_score": int(primary_eligible and is_new),
                "exclusion_reason": "" if primary_eligible else "; ".join(reasons),
            }
            rows.append(row)
            if row["eligible_newer_for_frozen_score"]:
                coords = np.asarray([ca_by_pdb[pdb][chain][int(residue)] for residue in frame.window], dtype=float)
                score_inputs[entity_id] = (chain, coords, row)
    return rows, score_inputs, downloads


def score_newer(score_inputs: Mapping[str, tuple[str, np.ndarray, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id in sorted(score_inputs):
        chain, coords, inventory_row = score_inputs[entity_id]
        pc1_score, coordinate = closure_score(coords)
        row = {
            "pdb_entity": entity_id,
            "pdb_id": inventory_row["pdb_id"],
            "chain": chain,
            "release_date": inventory_row["release_date"],
            "resolution_A": inventory_row["resolution_A"],
            "n_window_positions": int(coords.shape[0]),
            "pc1_score": f"{pc1_score:.6f}",
            "closure_coordinate": f"{coordinate:.6f}",
            "frozen_closed_band_max": FROZEN_CLOSED_THRESHOLD,
            "frozen_open_band_min": FROZEN_OPEN_THRESHOLD,
            "frozen_state_call": classify(coordinate),
            "title": inventory_row["title"],
            "ligands": inventory_row["ligands"],
            "other_polymer_partners": inventory_row["other_polymer_partners"],
        }
        rows.append(row)
    return rows


def load_or_fetch_live_query(output_dir: Path, *, offline: bool) -> dict[str, Any]:
    path = output_dir / OUTPUT_FILES["query"]
    if offline:
        if not path.is_file():
            raise FileNotFoundError(f"offline mode requires {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    query_record = live_search()
    write_json(path, query_record)
    return query_record


def load_or_fetch_metadata(
    output_dir: Path, entity_ids: Sequence[str], *, offline: bool
) -> dict[str, Any]:
    path = output_dir / OUTPUT_FILES["metadata"]
    if offline:
        if not path.is_file():
            raise FileNotFoundError(f"offline mode requires {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value.get("records") if isinstance(value, dict) else None
        if not isinstance(records, dict):
            raise ValueError(f"{path}: expected object with records")
        missing = sorted(set(entity_ids) - set(records))
        if missing:
            raise ValueError(f"{path}: missing metadata for {missing}")
        return records
    records = fetch_metadata(entity_ids)
    write_json(
        path,
        {
            "queried_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "endpoint_url": GRAPHQL_ENDPOINT,
            "entity_count": len(records),
            "entity_ids_sha256": sha256_bytes(stable_json_bytes(sorted(entity_ids))),
            "records_sha256": sha256_bytes(stable_json_bytes(records)),
            "records": records,
        },
    )
    return records


def manifest_for_outputs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.is_file():
            rows.append(
                {
                    "path": display_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def run(output_dir: Path, structure_dir: Path, *, offline: bool) -> dict[str, Any]:
    start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frozen_frame()

    per_reference_rows, mode_cache = frozen_per_reference(frame)
    pair_rows = open_closed_pair_comparison(frame, mode_cache)
    query_record = load_or_fetch_live_query(output_dir, offline=offline)
    metadata = load_or_fetch_metadata(output_dir, query_record["entities"], offline=offline)
    inventory_rows, score_inputs, downloads = chain_inventory(
        query_record, metadata, frame, structure_dir, offline=offline
    )
    score_rows = score_newer(score_inputs)
    temporal_mode_rows = temporal_mode_rankings(score_inputs, frame)
    sensitivity = pca_refit_sensitivity(score_inputs, frame)

    per_reference_path = output_dir / OUTPUT_FILES["per_reference"]
    pair_path = output_dir / OUTPUT_FILES["pairs"]
    inventory_path = output_dir / OUTPUT_FILES["inventory"]
    scores_path = output_dir / OUTPUT_FILES["scores"]
    temporal_modes_path = output_dir / OUTPUT_FILES["temporal_modes"]
    summary_path = output_dir / OUTPUT_FILES["summary"]

    per_reference_fields = [
        "pdb",
        "state",
        "cutoff_A",
        "n_modes",
        "mode1_overlap",
        "mode1_eigenvalue",
        "mode1_next_eigenvalue_gap",
        "best20_overlap",
        "best20_rank",
        "best20_eigenvalue",
        "best20_local_eigenvalue_gap",
        "best60_overlap",
        "best60_rank",
        "best60_eigenvalue",
        "best60_local_eigenvalue_gap",
        "top3_subspace_projection",
        "top10_subspace_projection",
        "top20_subspace_projection",
    ]
    pair_fields = [
        "open_pdb",
        "closed_pdb",
        "basis",
        "basis_pdb",
        "cutoff_A",
        "mode1_overlap",
        "mode1_eigenvalue",
        "mode1_next_eigenvalue_gap",
        "best20_overlap",
        "best20_rank",
        "best20_eigenvalue",
        "best20_local_eigenvalue_gap",
        "best60_overlap",
        "best60_rank",
        "best60_eigenvalue",
        "best60_local_eigenvalue_gap",
        "top3_subspace_projection",
        "top10_subspace_projection",
        "top20_subspace_projection",
    ]
    inventory_fields = [
        "pdb_entity",
        "pdb_id",
        "chain",
        "is_primary_chain",
        "is_new_since_frozen_query",
        "release_date",
        "deposit_date",
        "seq_length",
        "method",
        "resolution_A",
        "title",
        "primary_citation_doi",
        "mutation",
        "q96sw2_auth_chains",
        "q96sw2_mapping_groups",
        "q96sw2_deletion_groups",
        "n_window_resolved",
        "missing_residues",
        "ligands",
        "ddb1_present",
        "other_polymer_partners",
        "eligible_primary_current",
        "eligible_newer_for_frozen_score",
        "exclusion_reason",
    ]
    score_fields = [
        "pdb_entity",
        "pdb_id",
        "chain",
        "release_date",
        "resolution_A",
        "n_window_positions",
        "pc1_score",
        "closure_coordinate",
        "frozen_closed_band_max",
        "frozen_open_band_min",
        "frozen_state_call",
        "title",
        "ligands",
        "other_polymer_partners",
    ]
    temporal_mode_fields = [
        "pdb_entity",
        "pdb_id",
        "chain",
        "basis",
        "axis",
        "cutoff_A",
        "n_window_positions",
        "release_date",
        "resolution_A",
        "title",
        "mode1_overlap",
        "mode1_eigenvalue",
        "mode1_next_eigenvalue_gap",
        "best20_overlap",
        "best20_rank",
        "best20_eigenvalue",
        "best20_local_eigenvalue_gap",
        "best60_overlap",
        "best60_rank",
        "best60_eigenvalue",
        "best60_local_eigenvalue_gap",
        "top3_subspace_projection",
        "top10_subspace_projection",
        "top20_subspace_projection",
    ]

    write_csv(per_reference_path, per_reference_rows, per_reference_fields)
    write_csv(pair_path, pair_rows, pair_fields)
    write_csv(inventory_path, inventory_rows, inventory_fields)
    write_csv(scores_path, score_rows, score_fields)
    write_csv(temporal_modes_path, temporal_mode_rows, temporal_mode_fields)

    eligible_primary = [row for row in inventory_rows if int(row["eligible_primary_current"]) == 1]
    newer_primary = [row for row in inventory_rows if int(row["eligible_newer_for_frozen_score"]) == 1]
    score_states = {row["pdb_entity"]: row["frozen_state_call"] for row in score_rows}
    acquisition_errors = [row for row in downloads if row["error"]]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.monotonic() - start, 3),
        "offline": offline,
        "frozen": {
            "freeze_date": frame.freeze_date.isoformat(),
            "n_entities": len(frame.frozen_entities),
            "n_conformers": len(frame.labels),
            "n_open": int(frame.open_mask.sum()),
            "n_closed": int((~frame.open_mask).sum()),
            "closed_threshold": FROZEN_CLOSED_THRESHOLD,
            "open_threshold": FROZEN_OPEN_THRESHOLD,
        },
        "live_query": {
            "queried_at_utc": query_record.get("queried_at_utc", ""),
            "endpoint_url": query_record.get("endpoint_url", ""),
            "replay_url": query_record.get("replay_url", ""),
            "total_entities": int(query_record["total_count"]),
            "result_sha256": query_record["result_sha256"],
            "new_entity_count_vs_frozen": len(set(query_record["entities"]) - frame.frozen_entities),
        },
        "source_code": {
            "script": {
                "path": display_path(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
            "tests": {
                "path": "tests/test_strengthen_ensemble.py",
                "sha256": sha256_file(ROOT / "tests" / "test_strengthen_ensemble.py"),
            },
        },
        "current_primary_eligibility": {
            "eligible_primary_chains": len(eligible_primary),
            "newer_eligible_primary_chains": len(newer_primary),
            "newer_eligible_entities": [row["pdb_entity"] for row in newer_primary],
            "newer_frozen_score_states": score_states,
        },
        "temporal_mode_rankings": {
            "rows": len(temporal_mode_rows),
            "axis": "frozen_70_open_minus_closed",
            "basis": "own_new_structure",
            "note": "No 70+new PCA refit is used for these rankings.",
        },
        "pca_refit_sensitivity": sensitivity,
        "historical_postfreeze_baseline_note": (
            "The earlier four eligible entries 12BP, 9OPJ, 9V0C, and 9V0E are "
            "treated only as the previous post-freeze baseline; the current live "
            "inventory is recomputed from RCSB."
        ),
        "anm_references": {
            "per_reference_rows": len(per_reference_rows),
            "pair_rows": len(pair_rows),
            "pair_definition": "5 frozen open structures x 65 frozen closed structures x 2 bases",
            "open_reference_rank1": sum(
                1 for row in per_reference_rows if row["state"] == "open" and row["best20_rank"] == 1
            ),
            "closed_reference_rank1": sum(
                1 for row in per_reference_rows if row["state"] == "closed" and row["best20_rank"] == 1
            ),
        },
        "structure_manifest": downloads,
        "acquisition_gaps": acquisition_errors,
    }
    output_paths = [
        per_reference_path,
        pair_path,
        inventory_path,
        scores_path,
        temporal_modes_path,
        output_dir / OUTPUT_FILES["query"],
        output_dir / OUTPUT_FILES["metadata"],
    ]
    summary["output_manifest"] = manifest_for_outputs(output_paths)
    write_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--offline", action="store_true", help="reuse existing query/mmCIF artifacts")
    parser.add_argument(
        "--config",
        type=Path,
        help="optional JSON object with output_dir, structure_dir, and offline keys",
    )
    return parser.parse_args(argv)


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    value = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--config must point to a JSON object")
    if "output_dir" in value:
        args.output_dir = Path(value["output_dir"])
    if "structure_dir" in value:
        args.structure_dir = Path(value["structure_dir"])
    if "offline" in value:
        args.offline = bool(value["offline"])
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = apply_config(parse_args(argv))
    summary = run(args.output_dir, args.structure_dir, offline=bool(args.offline))
    print(
        "strengthen ensemble OK: "
        f"live_entities={summary['live_query']['total_entities']}, "
        f"newer_eligible={summary['current_primary_eligibility']['newer_eligible_primary_chains']}, "
        f"per_reference_rows={summary['anm_references']['per_reference_rows']}, "
        f"pair_rows={summary['anm_references']['pair_rows']}"
    )
    if summary["acquisition_gaps"]:
        print(f"acquisition gaps: {len(summary['acquisition_gaps'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
