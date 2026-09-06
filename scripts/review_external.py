#!/usr/bin/env python3
"""Frozen public-data comparisons for the CRBN review-response extension.

Only public coordinates, source documents and endpoint-specific observations
are used. Biased simulation coordinates are a computational comparator, not
experimental validation. No code obtained from a foreign archive is executed.
"""
from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import csv
import difflib
import gzip
import hashlib
import io
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import directional_external as previous
import strengthen_ensemble as census
import strengthen_external as source_tables
from softmode_lib import kabsch, kabsch_apply

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "119_crbn_directional_mechanics_20260906"
UA = "CRBN-review-response/1.0"
ZENODO_API = "https://zenodo.org/api/records/16459122"
MIN_FREE_BYTES = 100 * 1024**2
OCONNOR_S4_VISUAL_SOURCE_SHA256 = "0ff21510969676d824094cf9b5db9138055557d82e4eb3895c513cd606753c65"


def sha256(path: Path) -> str:
    return previous.sha256_file(path)


def write_json(path: Path, data: Any) -> None:
    previous.write_json(path, data)


def write_csv(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    previous.write_csv(path, rows, fields)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def relative(path: Path, package: Path) -> str:
    try:
        return path.resolve().relative_to(package.resolve()).as_posix()
    except ValueError:
        return path.resolve().relative_to(ROOT).as_posix()


def stage_cached(source: Path, destination: Path) -> None:
    """Stage immutable input without duplicating its disk blocks if possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"staged input mismatch: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def source_input(destination: Path, url: str, offline=False) -> None:
    """Private history is an optional cache, never a required public input."""
    if destination.is_file(): return
    for src in (ROOT/"data/directional_reference_inputs/external"/destination.name,
                OLD/"data/external"/destination.name,
                ROOT/"data/_cif_cache"/destination.name):
        if src.is_file():
            stage_cached(src,destination)
            return
    if offline: raise FileNotFoundError(f"offline public source absent: {destination}")
    status,body,headers,error=request(url)
    if status!=200: raise RuntimeError(f"source acquisition failed: {url}: {status}: {error}")
    if destination.suffix==".pdf" and not body.startswith(b"%PDF"): raise ValueError("source response is not PDF")
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(body)
    write_json(destination.with_suffix(destination.suffix+".source.json"),{
        "url":url,"acquired_utc":previous.utc_now(),"http_status":status,"sha256":sha256(destination),"bytes":len(body)})


def request(url: str, *, limit: int = 64 * 1024**2, timeout: int = 90,
            headers: dict[str, str] | None = None) -> tuple[int | None, bytes, dict, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError(f"response exceeds acquisition limit: {url}")
            return response.status, body, dict(response.headers), ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(min(limit, 65536)), dict(exc.headers), str(exc)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return None, b"", {}, str(exc)


def acquire_json(url: str, path: Path, offline: bool) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    if offline:
        raise FileNotFoundError(f"offline input absent: {path}")
    status, body, headers, error = request(url)
    if status != 200:
        raise RuntimeError(f"{url}: HTTP {status}: {error}")
    value = json.loads(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    write_json(path.with_suffix(path.suffix + ".source.json"), {
        "url": url, "acquired_utc": previous.utc_now(), "http_status": status,
        "bytes": len(body), "sha256": sha256(path), "etag": headers.get("ETag", ""),
    })
    return value


class RemoteZipReader(io.RawIOBase):
    """Seekable range access with a bounded in-memory block cache.

    Content-Range is validated on every request; servers that silently ignore
    Range cannot make a partial response masquerade as an acquired ZIP.
    """
    def __init__(self, url: str, size: int, block_size: int = 1024**2):
        self.url, self.size, self.position = url, int(size), 0
        self.block_size = block_size
        self.cache: collections.OrderedDict[int, bytes] = collections.OrderedDict()
        self.transferred = 0
        self.requests: list[dict] = []

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("negative ZIP seek")
        self.position = int(position)
        return self.position

    def _block(self, block: int) -> bytes:
        if block in self.cache:
            self.cache.move_to_end(block)
            return self.cache[block]
        start = block * self.block_size
        end = min(self.size, start + self.block_size) - 1
        for _attempt in range(3):
            status, data, headers, error = request(
                self.url, limit=self.block_size,
                headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if status == 429:
                time.sleep(min(30, int(headers.get("Retry-After", "15"))))
                continue
            if status != 206 or headers.get("Content-Range", headers.get("content-range")) != f"bytes {start}-{end}/{self.size}":
                raise RuntimeError(f"invalid HTTP range for ZIP: {status}, {headers}, {error}")
            if len(data) != end - start + 1:
                raise ValueError("truncated ZIP byte range")
            break
        else:
            raise RuntimeError("Zenodo range requests remain rate limited")
        self.transferred += len(data)
        self.requests.append({"start": start, "end": end, "sha256": hashlib.sha256(data).hexdigest()})
        self.cache[block] = data
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return data

    def read(self, n=-1):
        n = self.size - self.position if n < 0 else min(n, self.size - self.position)
        if n <= 0: return b""
        chunks = []
        while n:
            block, offset = divmod(self.position, self.block_size)
            data = self._block(block)
            chunk = data[offset:offset + n]
            if not chunk: raise EOFError("empty ZIP range")
            chunks.append(chunk)
            self.position += len(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def open_member(self, archive, member: str):
        """Stream one compressed ZIP member through the standard ZIP reader.

        A single ranged connection avoids one new TLS request for every cache
        block in a large trajectory. ZipExtFile performs standard decompression
        and validates the uncompressed CRC; this does not implement XTC decoding.
        """
        info=archive.getinfo(member)
        if info.flag_bits & 1: raise ValueError("encrypted archive member is unsupported")
        self.seek(info.header_offset)
        header=self.read(30)
        if header[:4]!=b"PK\x03\x04": raise ValueError("invalid local ZIP member header")
        name_len,extra_len=struct.unpack("<HH",header[26:30])
        start=info.header_offset+30+name_len+extra_len
        stream=ArchiveRangeStream(self.url,start,info.compress_size,self.size)
        return zipfile.ZipExtFile(stream,"r",info,close_fileobj=True)


class ArchiveRangeStream(io.RawIOBase):
    """Bounded, resumable byte stream for one compressed archive member."""
    def __init__(self,url,start,length,total_size):
        self.url=url;self.start=int(start);self.length=int(length)
        self.total_size=int(total_size);self.received=0;self.response=None;self.failures=0

    def readable(self):return True
    def seekable(self):return False

    def _connect(self):
        first=self.start+self.received;last=self.start+self.length-1
        req=urllib.request.Request(self.url,headers={"User-Agent":UA,"Range":f"bytes={first}-{last}"})
        response=urllib.request.urlopen(req,timeout=120)
        expected=f"bytes {first}-{last}/{self.total_size}"
        if response.status!=206 or response.headers.get("Content-Range")!=expected:
            response.close();raise ValueError("archive member range was not honored")
        self.response=response

    def read(self,n=-1):
        remaining=self.length-self.received
        if remaining==0:return b""
        n=min(remaining,1024**2 if n<0 else n,1024**2)
        while True:
            try:
                if self.response is None:self._connect()
                block=self.response.read1(n)
                if not block:raise EOFError("truncated compressed archive member")
                self.received+=len(block)
                return block
            except (OSError,urllib.error.URLError,EOFError) as exc:
                if self.response is not None:self.response.close();self.response=None
                self.failures+=1
                if self.failures>3:raise RuntimeError("archive member stream failed after three resumptions") from exc

    def close(self):
        if self.response is not None:self.response.close()
        super().close()

def safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and "\\" not in name


def stream_archive_digest(url: str, expected_bytes: int, expected_md5: str, output: Path) -> dict:
    """Verify the whole archive while retaining no archive-sized local file."""
    if output.is_file():
        result = json.loads(output.read_text())
        if result.get("md5") == expected_md5 and result.get("bytes") == expected_bytes:
            return result
    started = previous.utc_now()
    md5, sha = hashlib.md5(), hashlib.sha256()
    count = 0
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"whole-archive stream returned HTTP {response.status}")
            for chunk in iter(lambda: response.read(4 * 1024**2), b""):
                md5.update(chunk); sha.update(chunk); count += len(chunk)
                if count > expected_bytes:
                    raise ValueError("archive longer than frozen metadata")
        if count != expected_bytes or md5.hexdigest() != expected_md5:
            raise ValueError("archive bytes or MD5 disagree with frozen record")
        result = {"status": "whole_archive_stream_verified", "url": url, "started_utc": started,
                  "finished_utc": previous.utc_now(), "bytes": count, "md5": md5.hexdigest(),
                  "sha256": sha.hexdigest(), "archive_retained": False,
                  "storage_policy": "verified full byte stream; retained only selected compact inputs"}
    except Exception as exc:
        result = {"status": "whole_archive_stream_incomplete", "url": url,
                  "started_utc": started, "finished_utc": previous.utc_now(),
                  "bytes_received": count, "archive_retained": False,
                  "error": f"{type(exc).__name__}: {exc}"}
    write_json(output, result)
    return result


def acquisition_inventory(cfg: dict, out: Path, data: Path, offline: bool) -> dict:
    """Refresh the frozen census under its original inclusion rules."""
    target = out / "rcsb"
    target.mkdir(parents=True, exist_ok=True)
    structure_dir = data / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    query_path = target / census.OUTPUT_FILES["query"]
    metadata_path = target / census.OUTPUT_FILES["metadata"]
    query = census.load_or_fetch_live_query(target, offline=offline or query_path.is_file())
    metadata = census.load_or_fetch_metadata(target, query["entities"], offline=offline or metadata_path.is_file())
    for pdb in sorted({a.split("_")[0] for a in query["entities"]}):
        dst = structure_dir / f"{pdb}.cif.gz"
        if dst.exists(): continue
        for src in (ROOT / "data" / "_cif_cache" / dst.name,
                    ROOT / "118_csbj_strengthening_20260905/data/structures" / dst.name):
            if src.is_file():
                stage_cached(src, dst)
                break
    frame = census.load_frozen_frame()
    version_path=target/"temporal_coordinate_version_audit.json"
    if not version_path.is_file():
        if offline:raise FileNotFoundError("offline temporal coordinate version audit absent")
        newer=sorted({entity.split("_")[0] for entity in set(query["entities"])-frame.frozen_entities})
        def verify_current_cif(pdb):
            url=census.DOWNLOAD_ENDPOINT.format(pdb=pdb)
            status,body,headers,error=request(url)
            if status!=200:raise RuntimeError(f"temporal mmCIF version check failed: {pdb}: {status}: {error}")
            text=gzip.decompress(body)
            if b"_atom_site." not in text:raise ValueError("temporal CIF response lacks atom data")
            destination=structure_dir/f"{pdb}.cif.gz"
            old_hash=sha256(destination) if destination.exists() else ""
            new_hash=hashlib.sha256(body).hexdigest()
            if old_hash!=new_hash:
                # Replace this task's link; never overwrite the shared cache inode.
                with tempfile.NamedTemporaryFile(dir=structure_dir,delete=False) as staged:
                    staged.write(body);staged_path=Path(staged.name)
                staged_path.replace(destination)
            return {"pdb_id":pdb,"url":url,"checked_utc":previous.utc_now(),"http_status":status,
                    "last_modified":headers.get("Last-Modified",""),"etag":headers.get("ETag",""),
                    "sha256":new_hash,"uncompressed_sha256":hashlib.sha256(text).hexdigest(),
                    "cached_source_sha256":old_hash,"identical_to_cached_source":old_hash==new_hash}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            version_rows=list(executor.map(verify_current_cif,newer))
        write_json(version_path,version_rows)
        write_csv(target/"temporal_coordinate_version_audit.csv",version_rows)
    inventory, inputs, downloads = census.chain_inventory(query, metadata, frame, structure_dir, offline=offline)
    reference_studies = set()
    for row in inventory:
        if row["pdb_id"] in cfg["apo_references"]:
            reference_studies.add(str(row["primary_citation_doi"]).lower())
    for row in inventory:
        row["study_id"] = row["primary_citation_doi"] or f"unassigned:{row['pdb_id']}"
        row["construct_description"] = f"length={row['seq_length']}; mutations={row['mutation']}; mapping={row['q96sw2_mapping_groups']}"
        row["comparison_role"] = "frozen_census" if not row["is_new_since_frozen_query"] else "temporal_evaluation"
    scores = census.score_newer(inputs)
    rankings = census.temporal_mode_rankings(inputs, frame)
    by_id = {r["pdb_entity"]: r for r in inventory if r["is_primary_chain"]}
    independent = []
    for row in scores:
        src = by_id[row["pdb_entity"]]
        row.update({"study_id": src["study_id"], "construct_description": src["construct_description"],
                    "independent_of_apo_study": bool(src["primary_citation_doi"] and str(src["primary_citation_doi"]).lower() not in reference_studies)})
        if row["frozen_state_call"] == "open" and row["independent_of_apo_study"]:
            independent.append(row)
    write_csv(target / "chain_inventory.csv", inventory)
    write_csv(target / "eligible_frozen_scores.csv", scores)
    write_csv(target / "eligible_own_basis_mode_rankings.csv", rankings)
    for download in downloads:
        path = structure_dir / (download["pdb"] + ".cif.gz")
        download.update({"url": census.DOWNLOAD_ENDPOINT.format(pdb=download["pdb"]),
                         "path": relative(path, out.parent.parent) if path.is_file() else "",
                         "bytes": path.stat().st_size if path.is_file() else "",
                         "checked_utc": query["queried_at_utc"],
                         "source_version": "compressed mmCIF hash; query acquisition snapshot"})
    write_csv(target / "structure_sources.csv", downloads)
    summary = {"query_utc": query["queried_at_utc"], "entity_count": len(query["entities"]),
               "new_since_original_query": len(set(query["entities"]) - frame.frozen_entities),
               "eligible_temporal_count": len(scores), "eligible_temporal_pdbs": [r["pdb_id"] for r in scores],
               "eligible_temporal_state_counts": dict(collections.Counter(r["frozen_state_call"] for r in scores)),
               "independent_open_count": len(independent), "independent_open_rows": independent,
               "independent_open_static_status": "not_applicable_no_new_independent_open" if not independent else "requires_partner_coverage_evaluation",
               "original_70_frame_unchanged": True, "refit_pca_used_for_primary": False}
    write_json(target / "summary.json", summary)
    return summary


def oconnor_availability(cfg: dict, out: Path, data: Path, offline: bool) -> list[dict]:
    path = out / "oconnor_accessibility.json"
    cached=json.loads(path.read_text()) if path.is_file() else []
    if offline and not cached: raise FileNotFoundError(path)
    urls = [(p, "pdb_mmcif", f"https://files.rcsb.org/download/{p}.cif") for p in cfg["external_extension"]["oconnor_pdb"]]
    urls += [(p, "emdb_metadata", f"https://www.ebi.ac.uk/emdb/api/entry/{p}") for p in cfg["external_extension"]["oconnor_emdb"]]
    urls += [(p, "emdb_map", f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{p}/map/emd_{p.split('-')[1]}.map.gz") for p in cfg["external_extension"]["oconnor_emdb"]]
    rows = []
    def check(item):
        accession, kind, url = item
        status, body, headers, error = request(url, timeout=45, headers={"Range":"bytes=0-4095"} if kind=="emdb_map" else None)
        usable = status == 200 and bool(body)
        if usable and kind == "pdb_mmcif": usable = b"_atom_site." in body and body.startswith(b"data_")
        release_status=""
        if usable and kind == "emdb_metadata":
            try:
                meta=json.loads(body)
                release_status=meta.get("admin",{}).get("current_status",{}).get("code",{}).get("valueOf_","")
                usable = "map" in meta and release_status=="REL"
            except (ValueError, TypeError): usable = False
        if kind=="emdb_map": usable=status in (200,206) and body.startswith(b"\x1f\x8b")
        destination = data / "access_responses" / f"{accession}_{kind}.response"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        return {"accession": accession, "kind": kind, "url": url, "checked_utc": previous.utc_now(),
                "http_status": status, "usable_content": usable, "response_bytes": len(body),
                "response_sha256": sha256(destination), "response_path": relative(destination, out.parent.parent),
                "content_type": headers.get("Content-Type", ""), "error": error,
                "emdb_release_status":release_status,
                "interpretation": "accession mention alone is not download evidence"}
    known={(r["accession"],r["kind"]):r for r in cached}
    pending=[u for u in urls if (u[0],u[1]) not in known]
    if offline and pending: raise FileNotFoundError("offline map accessibility snapshot incomplete")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        new_rows = list(pool.map(check, pending))
    known.update({(r["accession"],r["kind"]):r for r in new_rows})
    rows=[known[(u[0],u[1])] for u in urls]
    for row in rows:
        if row["kind"]=="emdb_metadata":
            body=(data/"access_responses"/f"{row['accession']}_emdb_metadata.response").read_bytes()
            try: row["emdb_release_status"]=json.loads(body).get("admin",{}).get("current_status",{}).get("code",{}).get("valueOf_","")
            except (ValueError,TypeError):row["emdb_release_status"]=""
    write_json(path, rows); write_csv(out / "oconnor_accessibility.csv", rows)
    return rows


def spatial_audit(out: Path, data: Path, candidates: list[dict], offline=False) -> dict:
    cif = data / "9SFM.cif.gz"
    source_input(cif,previous.RCSB_CIF_URL,offline)
    text = previous.load_cif_text(cif)
    atoms, residues, qa = previous.structure_contacts(text, 4.5)
    candidate_rows = previous.candidate_overlap_rows(candidates, residues, text)
    reverse = previous.independent_contact_residue_audit(text, 4.5)
    write_csv(out / "9sfm_a1ceg_atom_contacts.csv", atoms)
    write_csv(out / "9sfm_a1ceg_residue_contacts.csv", residues)
    write_csv(out / "9sfm_all_candidate_spatial_comparison.csv", candidate_rows)
    # Each 15-A CA network edge is paired with the heavy-atom distance to the
    # SAME DDB1 residue. This is not a chemical-bond count.
    open_cif = data / "8CVP.cif.gz"
    source_input(open_cif,"https://files.rcsb.org/download/8CVP.cif.gz",offline)
    all_atoms, open_qa = previous.selected_atoms(previous.load_cif_text(open_cif))
    by_residue: dict[tuple[str, int], list] = {}
    for atom in all_atoms:
        if atom.get("group_PDB") != "ATOM" or str(atom.get("pdbx_PDB_model_num", "1")) != "1": continue
        if atom.get("type_symbol", "").upper() in {"H", "D"}: continue
        try: key = (atom["auth_asym_id"], int(atom["auth_seq_id"]))
        except (ValueError, KeyError): continue
        by_residue.setdefault(key, []).append(atom)
    # 8CVP Q96SW2 is chain B and Q16531 is chain A; assert rather than infer.
    open_text = previous.load_cif_text(open_cif)
    previous.chain_uniprot_mapping(open_text, "Q96SW2", "B")
    previous.chain_uniprot_mapping(open_text, "Q16531", "A")
    edges = []; nearest = []
    for residue in (221, 222, 339):
        ra = by_residue[("B", residue)]
        ca = next(a["xyz_array"] for a in ra if a["label_atom_id"] == "CA")
        best_all = (float("inf"), None, None, None)
        for (chain, partner), da in by_residue.items():
            if chain != "A": continue
            dc = [a["xyz_array"] for a in da if a["label_atom_id"] == "CA"]
            if not dc: continue
            ca_distance = float(np.linalg.norm(ca-dc[0]))
            rxyz, dxyz = np.stack([a["xyz_array"] for a in ra]), np.stack([a["xyz_array"] for a in da])
            distances = np.linalg.norm(rxyz[:,None,:]-dxyz[None,:,:],axis=2)
            i,j = np.unravel_index(np.argmin(distances), distances.shape)
            hd = float(distances[i,j])
            if hd < best_all[0]: best_all = (hd, partner, ra[i]["label_atom_id"], da[j]["label_atom_id"])
            if ca_distance <= 15:
                edges.append({"pdb":"8CVP","crbn_chain":"B","crbn_residue":residue,
                              "crbn_resname":ra[0]["label_comp_id"],"ddb1_chain":"A","ddb1_residue":partner,
                              "ddb1_resname":da[0]["label_comp_id"],"CA_distance_A":ca_distance,
                              "same_pair_min_heavy_atom_distance_A":hd,"crbn_atom":ra[i]["label_atom_id"],
                              "ddb1_atom":da[j]["label_atom_id"],"network_cutoff_A":15,
                              "atomic_contact_within_4p5A":hd<=4.5})
        nearest.append({"residue":residue,"resname":ra[0]["label_comp_id"],
                        "network_spring_count":sum(a["crbn_residue"]==residue for a in edges),
                        "min_heavy_atom_distance_to_any_DDB1_A":best_all[0],"nearest_DDB1_residue":best_all[1],
                        "nearest_CRBN_atom":best_all[2],"nearest_DDB1_atom":best_all[3]})
    write_csv(out / "8cvp_network_vs_atomic_contacts.csv", edges)
    write_csv(out / "8cvp_selected_residue_contact_summary.csv", nearest)
    summary={"candidate_count":len(candidate_rows),"9sfm_contact_count":len(residues),
             "9sfm_contact_residues":[r["uniprot_residue"] for r in residues],
             "9sfm_independent_audit":reverse,"8cvp_selected_residues":nearest,
             "role":"atomic spatial context; not a mutation or functional validation",
             "source_sha256":{"8CVP":sha256(open_cif),"9SFM":sha256(cif)}}
    write_json(out / "spatial_audit.json",summary)
    return summary


def candidate_inputs(data: Path) -> list[dict]:
    for name in ("candidate_universe.csv", "legacy_robustness.csv"):
        path=data / name
        if not path.exists(): stage_cached(ROOT / "data/directional_reference_inputs" / name, path)
    result=previous.load_candidate_rows(data / "candidate_universe.csv")
    if len(result)!=142: raise ValueError("candidate universe must remain 142 groups")
    return result


def pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def variant_audit(out: Path, data: Path, candidates: list[dict], offline=False) -> dict:
    """Recheck every tested identity against primary tables and endpoint types."""
    main=data/"Chrisochoidou_2025_Blood.pdf"
    supplement=data/"BLOOD_BLD-2024-025861-mmc1.pdf"
    oconnor=data/"OConnor_2025_media-1_supplement.pdf"
    for destination,url in ((main,previous.BLOOD_PDF_URL),(supplement,previous.BLOOD_SUPPLEMENT_URL),
                            (oconnor,source_tables.OCONNOR_SUPPLEMENT)):
        source_input(destination,url,offline)
    mp,sp,op=pdf_pages(main),pdf_pages(supplement),pdf_pages(oconnor)
    # Identity comes from the sequenced plasmids, independently checked against
    # the functional table. Primer-table labels contain two documented typos.
    tested=set(re.findall(r"p\.\s*([A-Z][0-9]+[A-Z])", "\n".join(sp[10:12])))
    table3=set(re.findall(r"p\.\s*([A-Z][0-9]+[A-Z])", "\n".join(sp[7:9])))
    expected={r[1] for r in previous.BLOOD_VARIANTS}
    if tested!=expected or table3!=expected:
        raise ValueError(f"Blood tested-variant identity discrepancy: sequencing={tested}, table3={table3}")
    main_text=" ".join(" ".join(mp).split())
    if not all(term in main_text for term in ("W415X", "Trp (W)", "Gly (G)")):
        raise ValueError("W415 patient-versus-experiment distinction not supported by main caption")
    identities=[]; endpoints=[]
    base=previous.blood_variant_rows(candidates)
    core=set(map(int,census.read_window()))
    for row in base:
        mutation=row["experimentally_tested_symbol"]
        position=int(re.search(r"\d+",mutation).group())
        actual_window="inside_269_window" if position in core else "outside_269_window"
        evidence_page=8 if mutation not in {"W386A","H397Y","W415G"} else 9
        mismatch=("Supplemental Table 4 labels Y386A although TGG-to-GCG and sequenced Tables 5A/5B establish W386A" if mutation=="W386A" else
                  "Main Figure 1 reports patient W415X; experimental Tables 3/5 establish W415G (TGG-to-GGG); Table 2 uses W415G and primer Table 4 retains W415X" if mutation=="W415G" else "")
        identity={"study":"Chrisochoidou_2025_Blood","tested_substitution":mutation,
                  "patient_reported_symbol": "not_patient_variant_positive_control" if mutation=="W386A" else "W415X" if mutation=="W415G" else mutation,
                  "sequencing_verified":mutation in tested,"functional_table_verified":mutation in table3,
                  "identity_evidence":"Supplementary Tables 3, 5A and 5B; W415 distinction: main Figure 1 caption",
                  "source_label_discrepancy":mismatch,"primary_269_window":actual_window,
                  "previous_inventory_window":row["primary_269_window"],
                  "previous_window_label_corrected":actual_window!=row["primary_269_window"],
                  "candidate_contact_classes":row["candidate_contact_classes"],
                  "stable_apo_candidate_overlap":row["stable_apo_candidate_overlap"],
                  "functional_class_from_table3":row["cell_response_endpoint"],
                  "source_main_sha256":sha256(main),"source_supplement_sha256":sha256(supplement)}
        identities.append(identity)
        for endpoint,measurement,status,observation,location in (
            ("binding","direct ligand affinity","not_separately_measured","No variant-resolved affinity measurement extracted; structural predictions of binding are not binding data","Methods and Supplementary Table 3"),
            ("abundance","CRBN immunoblot expression control","qualitative_experiment","Stable re-expression to similar CRBN levels was checked; this is not a folding or thermal-stability measurement","Main Methods; Supplementary Figure 2"),
            ("folding_stability","direct folding or thermal stability","not_separately_measured","Destabilization proposed from structure is not an experimentally measured folding endpoint","Supplementary Table 3 prediction column"),
            ("degradation","neosubstrate immunoblot after 24 h","plotted_experiment_not_digitized","Compound-specific blots and plotted quantification retained as source evidence; no common numerical effect imputed from the combined functional class","Main Figures 2-4 and corresponding supplementary immunoblots"),
            ("cell_response","CellTiter-Blue viability after 5 days","qualitative_from_published_table",row["cell_response_endpoint"],f"Supplementary Table 3, page {evidence_page}; main Figures 2-4; Supplementary Figure 6"),
        ):
            endpoints.append({"study":"Chrisochoidou_2025_Blood","tested_substitution":mutation,
                              "endpoint_class":endpoint,"measurement":measurement,"evidence_status":status,
                              "observation":observation,"value":"","unit":"","estimated_from_figure":False,
                              "source_location":location,"system":"CRBN-reconstituted myeloma cells; main MM1.s knockout clones, protein-level KMS11 validation",
                              "compound_conditions":"lenalidomide; pomalidomide; iberdomide; mezigdomide; assay-specific doses retained in source",
                              "candidate_contact_classes":row["candidate_contact_classes"],
                              "stable_candidate_overlap":row["stable_apo_candidate_overlap"],
                              "interpretation":"endpoint-specific retrospective comparison; no substitution-to-spring equivalence"})
    write_csv(out/"blood_tested_variant_identity_audit.csv",identities)
    write_csv(out/"blood_endpoint_evidence.csv",endpoints)
    # The existing source table is numerical, but numbers are rechecked against
    # the indicated primary PDF page instead of trusting the prior inventory.
    numerical=[{k:str(v) for k,v in row.items()} for row in source_tables.oconnor_quantitative_rows()]
    qc=[]
    for row in numerical:
        page=int(row["pdf_page"])
        normalized=op[page-1].replace("−","-").replace("–","-").replace("\u2009","")
        tokens=re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",normalized)
        value=row["value"]
        try:
            v=float(value)
            present=any(np.isclose(v,float(t),rtol=1e-7,atol=1e-7) for t in tokens)
        except ValueError:
            present=value=="" or re.sub(r"\s","",value) in re.sub(r"\s","",normalized)
        qc.append({**row,"source_sha256":sha256(oconnor),"page_numeric_token_present":bool(present),
                   "independent_visual_row_verified":row["table_id"]=="S4" and sha256(oconnor)==OCONNOR_S4_VISUAL_SOURCE_SHA256,
                   "audit_pass":bool(present) or (row["table_id"]=="S4" and sha256(oconnor)==OCONNOR_S4_VISUAL_SOURCE_SHA256),
                   "audit_scope":"S4: all Tm and delta-Tm cells checked on the rendered primary page 13; other tables: page-level numeric presence with retained source-table pairing",
                   "endpoint_class":"folding_stability" if row["assay_type"]=="folding_dsf" else "binding" if row["assay_type"].startswith("binding") else "solution_compaction"})
    write_csv(out/"oconnor_measurement_source_audit.csv",qc)
    curated_inventory=data/"oconnor_variant_inventory.csv"
    write_csv(curated_inventory,source_tables.oconnor_variant_inventory_rows())
    variants=previous.oconnor_reuse_rows(curated_inventory,candidates)
    for row in variants:
        positions=[int(n) for n in re.findall(r"\d+",row["variant"])]
        row["previous_inventory_window"]=row["primary_269_window"]
        row["primary_269_window"]="inside_269_window" if positions and all(p in core for p in positions) else "outside_269_window" if positions and all(p not in core for p in positions) else "mixed_window" if positions else "wild_type"
        row["source_supplement_sha256"]=sha256(oconnor)
        row["source_text_contains_variant"]=all(m in " ".join(op) for m in row["variant"].split())
        row["construct_scope"]="engineered CRBNmidi; binding/DSF/SAXS endpoints are distinct"
    write_csv(out/"oconnor_all_variant_comparison.csv",variants)
    source_inventory=[]
    for path,url in ((main,previous.BLOOD_PDF_URL),(supplement,previous.BLOOD_SUPPLEMENT_URL),
                     (oconnor,"https://pmc.ncbi.nlm.nih.gov/articles/instance/12767645/bin/media-1.pdf")):
        source_inventory.append({"file":relative(path,out.parent.parent),"url":url,"sha256":sha256(path),
                                 "bytes":path.stat().st_size,"acquisition_role":"reuse of previously acquired public primary source; original bytes unchanged"})
    write_csv(out/"variant_primary_sources.csv",source_inventory)
    summary={"blood_tested_variant_count":len(identities),"blood_table3_matches_sequencing":tested==table3,
             "blood_endpoint_rows":len(endpoints),"blood_stable_candidate_overlap_count":sum(r["stable_apo_candidate_overlap"] for r in identities),
             "blood_source_label_discrepancies":[r["tested_substitution"] for r in identities if r["source_label_discrepancy"]],
             "blood_window_labels_corrected":[r["tested_substitution"] for r in identities if r["previous_window_label_corrected"]],
             "oconnor_measurement_rows":len(qc),"oconnor_numeric_token_mismatches":sum(not r["page_numeric_token_present"] for r in qc),
             "oconnor_visual_tableS4_verified_rows":sum(r["independent_visual_row_verified"] for r in qc),
             "oconnor_source_audit_failures":sum(not r["audit_pass"] for r in qc),
             "oconnor_variants":len(variants),"oconnor_stable_candidate_overlap_count":sum(bool(r["stable_apo_candidate_overlap"]) for r in variants),
             "no_cross_assay_pooled_statistics":True,"no_mutation_to_spring_mapping":True}
    write_json(out/"variant_audit_summary.json",summary)
    return summary


def zenodo_index(cfg: dict, out: Path, data: Path, offline: bool):
    metadata=acquire_json(ZENODO_API, data/"zenodo_16459122.json",offline)
    expected=cfg["external_extension"]
    records=[r for r in metadata["files"] if r["key"]==expected["zenodo_file"]]
    if len(records)!=1: raise ValueError("expected a unique frozen Zenodo archive")
    record=records[0]
    if record["size"]!=expected["zenodo_bytes"] or record["checksum"]!="md5:"+expected["zenodo_md5"]:
        raise ValueError("Zenodo archive differs from the approved data identity")
    index_path=out/"zenodo_zip_index.json"
    if index_path.exists(): return record,json.loads(index_path.read_text())
    if offline: raise FileNotFoundError(index_path)
    reader=RemoteZipReader(record["links"]["self"],record["size"])
    with zipfile.ZipFile(reader) as archive:
        rows=[{"member":info.filename,"bytes":info.file_size,"compressed_bytes":info.compress_size,
               "crc32":f"{info.CRC:08x}","compression_type":info.compress_type,
               "header_offset":info.header_offset,"is_directory":info.is_dir(),
               "safe_path":safe_member(info.filename)} for info in archive.infolist()]
    if any(not r["safe_path"] for r in rows): raise ValueError("unsafe archive path")
    write_json(index_path,rows);write_csv(out/"zenodo_zip_index.csv",rows)
    write_json(out/"zenodo_index_acquisition.json",{
        "source":ZENODO_API,"archive_url":record["links"]["self"],"checked_utc":previous.utc_now(),
        "entry_count":len(rows),"archive_bytes":record["size"],
        "advertised_md5":expected["zenodo_md5"],"range_bytes_received":reader.transferred,
        "range_requests":reader.requests,"whole_archive_hash_verified":False})
    return record,rows


def zenodo_member(archive, member: str, data: Path) -> bytes:
    """Retain compressed source bytes for compact PDB and text inputs."""
    if not safe_member(member): raise ValueError("unsafe member path")
    path=data/"zenodo_members"/(member+".gz")
    if path.exists():
        provenance=json.loads(path.with_suffix(path.suffix+".source.json").read_text())
        if sha256(path)!=provenance["retained_gzip_sha256"]:
            raise ValueError("retained source member checksum mismatch")
        content=gzip.decompress(path.read_bytes())
        if hashlib.sha256(content).hexdigest()!=provenance["sha256"]:
            raise ValueError("retained source member content checksum mismatch")
        return content
    if archive is None: raise FileNotFoundError(f"offline member absent: {member}")
    info=archive.getinfo(member)
    if info.file_size > 50*1024**2: raise ValueError("compact member exceeds 50 MiB limit")
    content=archive.read(member)
    encoded=gzip.compress(content,mtime=0)
    if shutil.disk_usage(data).free-len(encoded)<MIN_FREE_BYTES:
        raise OSError("insufficient disk reserve for compact member cache")
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as staged:
        staged.write(encoded);staged_name=Path(staged.name)
    staged_name.replace(path)
    write_json(path.with_suffix(path.suffix+".source.json"),{
        "member":member,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest(),
        "zip_crc32_verified":True,"crc32":f"{info.CRC:08x}","retained_gzip_sha256":sha256(path),
        "acquired_utc":previous.utc_now()})
    return content


AA3=dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),"ARNDCQEGHILKMFPSTWYV"))
AA3.update(HIE="H",HID="H",HIP="H",ASH="D",GLH="E",LYN="K",CYX="C",CYM="C",MSE="M")


def pdb_frames(content: bytes):
    atoms=[];model=0
    for line in content.decode().splitlines():
        if line.startswith("MODEL"):
            if atoms: yield model,atoms; atoms=[]
            model+=1
        elif line.startswith("ENDMDL"):
            if atoms: yield max(model,1),atoms;atoms=[]
        elif line.startswith(("ATOM  ","HETATM")):
            if len(line)<54: raise ValueError("truncated PDB atom line")
            atoms.append({"atom":line[12:16].strip(),"resname":line[17:20].strip(),
                          "chain":line[21],"resnum":int(line[22:26]),"icode":line[26],
                          "xyz":np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]),
                          "altloc":line[16],"serial":int(line[6:11])})
    if atoms: yield max(model,1),atoms


def topology_mapping(atoms: list[dict], data: Path, offline: bool = False) -> dict:
    ca=[(i,a) for i,a in enumerate(atoms) if a["atom"]=="CA" and a["resname"] in AA3 and a["altloc"] in (" ","A")]
    sequence="".join(AA3[a["resname"]] for _,a in ca)
    maps={};segments={}
    for accession in ("Q96SW2","Q16531"):
        fasta=data/(accession+".fasta")
        source_input(fasta,f"https://rest.uniprot.org/uniprotkb/{accession}.fasta",offline)
        canonical="".join(fasta.read_text().splitlines()[1:])
        match=difflib.SequenceMatcher(a=canonical,b=sequence,autojunk=False)
        blocks=[b for b in match.get_matching_blocks() if b.size>=4]
        # Short coincidental peptides must not assign DDB1 to CRBN-only files.
        if max((b.size for b in blocks),default=0)<50:blocks=[]
        maps[accession]={b.a+j+1:b.b+j for b in blocks for j in range(b.size)}
        segments[accession]=[{"canonical_start":b.a+1,"topology_CA_start":b.b+1,"length":b.size} for b in blocks]
    overlap=set(maps["Q96SW2"].values())&set(maps["Q16531"].values())
    if overlap: raise ValueError("ambiguous CRBN/DDB1 sequence assignment")
    core=list(map(int,census.read_window()))
    missing=[r for r in core if r not in maps["Q96SW2"]]
    return {"atom_count":len(atoms),"ca_atom_indices":np.array([i for i,_ in ca],dtype=np.int32),
            "ca_topology":ca,"maps":maps,"segments":segments,"missing_core":missing,
            "core_ca_indices":np.array([maps["Q96SW2"][r] for r in core if r in maps["Q96SW2"]],dtype=int),
            "ddb1_ca_indices":np.array(list(maps["Q16531"].values()),dtype=int),
            "crbn_ca_indices":np.array(list(maps["Q96SW2"].values()),dtype=int)}


def unwrap_protein_ca(ca: np.ndarray, mapping: dict, box: np.ndarray | None):
    result=np.array(ca,dtype=float,copy=True)
    if box is None or not np.isfinite(box).all() or abs(np.linalg.det(box))<1e-6: return result
    inv=np.linalg.inv(box)
    for field in ("crbn_ca_indices","ddb1_ca_indices"):
        indices=mapping[field]
        for first,second in zip(indices[:-1],indices[1:]):
            difference=result[second]-result[first]
            result[second]-=np.round(difference@inv)@box
    ddb1=mapping["ddb1_ca_indices"]
    if len(ddb1):
        core=mapping["core_ca_indices"];window=census.read_window()
        hb=core[(window>=187)&(window<=317)]
        delta=result[ddb1].mean(0)-result[hb].mean(0)
        result[ddb1]-=np.round(delta@inv)@box
    return result


class FrameScorer:
    def __init__(self,mapping: dict):
        if mapping["missing_core"]: raise ValueError(f"fixed core missing: {mapping['missing_core']}")
        self.mapping=mapping
        self.window,self.mean,self.pc1,self.closed,self.open=census.load_reference()
        self.hb=(self.window>=187)&(self.window<=317)
        self.tbd=self.window>=318;self.ntd=self.window<187
        self.reference_ddb1=None;self.reference_tbd=None

    def geometry(self, ca):
        """Report peptide geometry without selecting frames for agreement."""
        rows={}
        for accession,label in (("Q96SW2","CRBN"),("Q16531","DDB1")):
            positions=self.mapping.get("maps",{}).get(accession,{})
            pairs=[(positions[r],positions[r+1]) for r in positions if r+1 in positions]
            if not pairs: continue
            distances=np.linalg.norm(ca[[p[0] for p in pairs]]-ca[[p[1] for p in pairs]],axis=1)
            rows.update({label+"_adjacent_CA_min_A":float(distances.min()),
                         label+"_adjacent_CA_max_A":float(distances.max()),
                         label+"_adjacent_CA_below_2p5A":int(np.count_nonzero(distances<2.5)),
                         label+"_adjacent_CA_above_4p5A":int(np.count_nonzero(distances>4.5))})
            if accession=="Q96SW2":
                core=set(map(int,self.window))
                core_pairs=[(positions[r],positions[r+1]) for r in positions if r in core and r+1 in core and r+1 in positions]
                core_distances=np.linalg.norm(ca[[p[0] for p in core_pairs]]-ca[[p[1] for p in core_pairs]],axis=1)
                rows.update(CRBN_core_adjacent_CA_min_A=float(core_distances.min()),
                            CRBN_core_adjacent_CA_max_A=float(core_distances.max()),
                            CRBN_core_adjacent_CA_below_2p5A=int(np.count_nonzero(core_distances<2.5)))
        return rows

    def score(self,ca,box=None):
        ca=unwrap_protein_ca(ca,self.mapping,box)
        if not np.isfinite(ca).all(): raise ValueError("nonfinite trajectory coordinate")
        core=ca[self.mapping["core_ca_indices"]]
        aligned=kabsch_apply(core,self.mean)
        raw=float((aligned-self.mean).ravel()@self.pc1/np.sqrt(len(self.window)))
        coordinate=(raw-self.closed)/(self.open-self.closed)
        rotation,translation,hb_rmsd=kabsch(core[self.hb],self.mean[self.hb])
        anchored=ca@rotation.T+translation
        ddb1=anchored[self.mapping["ddb1_ca_indices"]]
        tbd=anchored[self.mapping["core_ca_indices"]][self.tbd]
        if self.reference_ddb1 is None:self.reference_ddb1=ddb1.copy();self.reference_tbd=tbd.copy()
        row={"closure_coordinate":coordinate,"frozen_pc1_score":raw,
             "HB_fit_RMSD_to_frozen_mean_A":hb_rmsd,**self.geometry(ca),
             "CRBN_core_RMSD_to_frozen_mean_A":float(np.sqrt(np.mean(np.sum((aligned-self.mean)**2,axis=1)))),
             "NTD_TBD_centroid_distance_A":float(np.linalg.norm(core[self.ntd].mean(0)-core[self.tbd].mean(0))),
             "DDB1_observed_mapped_CA_count":len(ddb1),"DDB1_body_translation_A":"","DDB1_body_rotation_deg":"","DDB1_internal_RMSD_A":""}
        for label,mobile,reference in (("DDB1",ddb1,self.reference_ddb1),("TBD",tbd,self.reference_tbd)):
            if len(mobile)>=3:
                rr,tt,rmsd=kabsch(mobile,reference)
                row[label+"_body_translation_A"]=float(np.linalg.norm(mobile.mean(0)-reference.mean(0)))
                row[label+"_body_rotation_deg"]=float(np.degrees(np.arccos(np.clip((np.trace(rr)-1)/2,-1,1))))
                row[label+"_internal_RMSD_A"]=rmsd
        return row


def read_exact(stream,n:int) -> bytes:
    chunks=[];remaining=n
    while remaining:
        b=stream.read(remaining)
        if not b:raise EOFError("truncated XTC record")
        chunks.append(b);remaining-=len(b)
    return b"".join(chunks)


def xtc_records(stream):
    """Frame boundaries only; standard MDTraj performs all decompression."""
    while True:
        prefix=stream.read(4)
        if not prefix:return
        if len(prefix)!=4:raise EOFError("truncated XTC magic")
        header=prefix+read_exact(stream,52)
        magic,natoms,step=struct.unpack(">iii",header[:12])
        coordinate_count=struct.unpack(">i",header[52:56])[0]
        if magic!=1995 or natoms!=coordinate_count or not 0<natoms<10_000_000:
            raise ValueError("unsupported or malformed XTC frame header")
        if natoms<=9:
            yield header+read_exact(stream,natoms*12)
        else:
            compressed=read_exact(stream,36)
            count=struct.unpack(">i",compressed[-4:])[0]
            if not 0<count<50*1024**2:raise ValueError("XTC compressed frame size outside bounded spool limit")
            yield header+compressed+read_exact(stream,(count+3)//4*4)


def xtc_batches(stream,mapping:dict,data:Path,batch_bytes=8*1024**2):
    from mdtraj.formats import XTCTrajectoryFile
    buffer=bytearray();count=0
    for record in xtc_records(stream):
        buffer.extend(record);count+=1
        if len(buffer)>=batch_bytes:
            with tempfile.NamedTemporaryFile(suffix=".xtc",dir=data) as tmp:
                tmp.write(buffer);tmp.flush()
                with XTCTrajectoryFile(tmp.name) as reader:
                    xyz,t,step,box=reader.read(atom_indices=mapping["ca_atom_indices"])
                if len(xyz)!=count:raise ValueError("standard XTC reader disagrees with framed batch")
                yield xyz*10,t,step,box*10 if box is not None else [None]*len(xyz)
            buffer.clear();count=0
    if buffer:
        with tempfile.NamedTemporaryFile(suffix=".xtc",dir=data) as tmp:
            tmp.write(buffer);tmp.flush()
            with XTCTrajectoryFile(tmp.name) as reader:
                xyz,t,step,box=reader.read(atom_indices=mapping["ca_atom_indices"])
            if len(xyz)!=count:raise ValueError("standard XTC reader disagrees with final framed batch")
            yield xyz*10,t,step,box*10 if box is not None else [None]*len(xyz)


def trajectory_topology_candidates(member:str,index:list[dict]) -> list[str]:
    directory=PurePosixPath(member).parent
    stem=PurePosixPath(member).stem
    same={r["member"] for r in index if r["member"].endswith(".pdb") and PurePosixPath(r["member"]).parent==directory}
    priorities=[str(directory/(stem+".pdb")),str(directory/"simul-top.pdb"),str(directory/"topol.pdb")]
    if "/Simulations/Apo/" in member:
        priorities=["rMDautoencoderGitHubRepo/Simulations/Apo/topol.pdb"]+priorities
        if "traj-path-" in member: priorities=["rMDautoencoderGitHubRepo/Simulations/Apo/path-refine-topol.pdb"]+priorities
    priorities+=sorted(same,key=lambda s:("final_state" in s,"minim" in s,s))
    return list(dict.fromkeys(p for p in priorities if any(r["member"]==p for r in index)))


def trajectory_role(member:str) -> tuple[str,str]:
    if member.endswith("aligned-10kframes.xtc"):
        return "duplicate_alignment","Concatenation and rigid alignment of the two 5000-frame apo trajectories, documented in Apo/README"
    if "/Simulations/Apo/traj-path-refine" in member:
        return "duplicate_path","Copy of the path trajectory in path-meta-eABF/closed; ZIP size and CRC matched"
    if "simul-chainB-" in member:
        return "duplicate_CRBN_subset","CRBN-only autoimaged subset of simul.xtc in the same directory, documented by README/cpptraj commands"
    if member.endswith("prot-only.xtc"):
        return "processed_coordinate_comparator","Protein-only named representation in the same simulation directory; analyzed separately because exact processing provenance was not established"
    if member.endswith("relax.xtc"):
        return "equilibration_coordinate_comparator","Public equilibration frames, kept separate from production and path-derived frames"
    if "traj-path-refine" in member:
        return "path_selected_coordinate_comparator","Path-selected frames; selection and bias preclude population or kinetic interpretation"
    return "biased_simulation_coordinate_comparator","All frames of this public simulation; no equilibrium, kinetic or experimental-validation claims"


def predicted_path_comparison(cfg:dict,out:Path,data:Path,offline=False,archive=None):
    record,index=zenodo_index(cfg,out,data,offline)
    groups={
        "autoencoder_raw":sorted([r["member"] for r in index if "/rMD-autoencoder-predicted-path-frames/" in r["member"] and r["member"].endswith(".pdb")],key=lambda s:int(re.search(r"_(\d+)\.pdb",s).group(1))),
        "rosetta_relaxed":sorted([r["member"] for r in index if "/Rosetta-Relax/" in r["member"] and r["member"].endswith(".pdb")],key=lambda s:int(re.search(r"relaxed(\d+)\.pdb",s).group(1))),
        "final_refined":["rMDautoencoderGitHubRepo/Refinement/Final-cleanup/path_all.pdb"],
    }
    rows=[];summaries=[];mapping_rows=[]
    for group,members in groups.items():
        scorer=None;number=0;group_rows=[];status="eligible";reason=""
        for member in members:
            payload=zenodo_member(archive,member,data)
            for source_model,atoms in pdb_frames(payload):
                mapping=topology_mapping(atoms,data,offline)
                if mapping["missing_core"]:status="excluded";reason=f"core missing {mapping['missing_core']}";break
                if scorer is None:
                    scorer=FrameScorer(mapping)
                    mapping_rows.append({"trajectory_id":group,"member":member,"CA_count":len(mapping["ca_atom_indices"]),
                                         "CRBN_mapped_count":len(mapping["crbn_ca_indices"]),"DDB1_mapped_count":len(mapping["ddb1_ca_indices"]),
                                         "core_count":len(mapping["core_ca_indices"]),"sequence_mapping":mapping["segments"]})
                elif mapping["maps"]!=scorer.mapping["maps"]:
                    raise ValueError(f"canonical CA order changes within compact path group: {group}: {member}")
                xyz=np.stack([a["xyz"] for _,a in mapping["ca_topology"]])
                number+=1
                group_rows.append({"trajectory_id":group,"frame":number,"source_member":member,"source_model":source_model,
                                   "coordinate_role":"model_predicted_refined_path",**scorer.score(xyz)})
            if status=="excluded":break
        if status=="eligible":rows+=group_rows
        geometry_flag=any(r.get("CRBN_adjacent_CA_below_2p5A",0)>0 for r in group_rows)
        adoption="descriptive_computational_structure_comparison" if group=="final_refined" and not geometry_flag else "intermediate_model_geometry_diagnostic_only"
        for r in group_rows:r["quantitative_adoption"]=adoption
        summaries.append({"trajectory_id":group,"role":"model_predicted_refined_path","status":status,
                          "frames_analyzed":len(group_rows),"reason":reason,"DDB1_scope":"absent from these CRBN-only path coordinates",
                          "canonical_CA_mapping_checked_for_every_frame":True,
                          "quantitative_adoption":adoption,
                          "geometry_diagnostic":"Adjacent canonical C-alpha distances <2.5 A flag severely compressed peptide geometry; post-acquisition source-quality diagnostic, not a prespecified hypothesis test",
                          "frames_with_severe_CA_compression":sum(r.get("CRBN_adjacent_CA_below_2p5A",0)>0 for r in group_rows),
                          "frames_with_severe_core_CA_compression":sum(r.get("CRBN_core_adjacent_CA_below_2p5A",0)>0 for r in group_rows),
                          "adoption_basis":"Source workflow final-cleanup representation is the primary path comparator; earlier representations are related intermediate model stages",
                          "closure_coordinate_min":min((r["closure_coordinate"] for r in group_rows),default=None),
                          "closure_coordinate_max":max((r["closure_coordinate"] for r in group_rows),default=None)})
    write_csv(out/"zenodo_predicted_path_metrics.csv",rows)
    write_json(out/"zenodo_predicted_path_mapping.json",mapping_rows)
    write_json(out/"zenodo_predicted_path_summary.json",{
        "groups":summaries,"frames":len(rows),"role":"biased computational path comparison; no independent biological validation",
        "mapping_rule":"exact canonical sequence blocks >=4 residues, with at least one >=50-residue identity block; complete fixed269 required",
        "raw_sources_retained":True,"offline_raw_PDB_recomputation":True})
    return rows,summaries


def scientific_fingerprint(cfg:dict,data:Path):
    """Identify the scientific operations and frozen inputs independently of I/O."""
    functions=(FrameScorer,topology_mapping,unwrap_protein_ca,pdb_frames,xtc_records,
               xtc_batches,read_exact,kabsch,kabsch_apply,census.load_reference,census.read_window)
    source={}
    for obj in functions:
        tree=ast.parse(inspect.getsource(obj))
        # Python 3.12 adds empty type_params fields. Normalize earlier ASTs so
        # the same scientific source has the same identity across runtimes.
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and "type_params" not in node._fields:
                node._fields=(*node._fields,"type_params");node.type_params=[]
        source[obj.__name__]=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    paths=[census.WINDOW,census.PCA_INPUT,data/"Q96SW2.fasta",data/"Q16531.fasta"]
    evidence={"formula_AST_sha256":source,"frozen_input_sha256":{p.name:sha256(p) for p in paths},
              "configuration":{"crbn_position_count":cfg["crbn_position_count"],
                               "domain_bounds":cfg["contact"]["domain_bounds"]},
              "residue_code_mapping":AA3,"XTC_decoder":"MDTraj 1.11.0; XTC nm converted to Angstrom"}
    payload=json.dumps(evidence,sort_keys=True,separators=(",",":")).encode()
    return {"sha256":hashlib.sha256(payload).hexdigest(),"components":evidence}


def trajectory_source_condition(member:str,ligands:list[str]):
    ligand_label="apo" if not ligands else "ligand-containing ("+", ".join(ligands)+")"
    if member.endswith("relax.xtc"):
        return ligand_label+"; relaxation stage (bias flag for this execution not established)"
    if "/Simulations/Apo/" in member:return "three-CV meta-eABF; "+ligand_label
    if "traj-path-refine" in member:return "path-selected coordinates from path-CV meta-eABF; "+ligand_label
    if member.endswith("prot-only.xtc"):
        return "processed protein-only coordinates from path-CV meta-eABF directory; "+ligand_label
    return "path-CV meta-eABF; "+ligand_label


def topology_identity(member,data,offline=False):
    content=zenodo_member(None,member,data)
    atoms=next(pdb_frames(content))[1]
    mapping=topology_mapping(atoms,data,offline)
    identity={accession:{str(k):int(v) for k,v in values.items()} for accession,values in mapping["maps"].items()}
    encoded=json.dumps(identity,sort_keys=True,separators=(",",":")).encode()
    ligand_components=sorted({a["resname"] for a in atoms if a["resname"] not in AA3 and a["resname"] not in {"WAT","HOH","Na+","Cl-","NA","CL","ZN"}})
    structural_metal_components=sorted({a["resname"] for a in atoms if a["resname"]=="ZN"})
    return {"topology_sha256":hashlib.sha256(content).hexdigest(),
            "canonical_CA_mapping_sha256":hashlib.sha256(encoded).hexdigest(),
            "source_ligand_components":ligand_components,"structural_metal_components":structural_metal_components,"mapping":mapping}


def checkpoint_is_verified(old,table,out,cfg,data,info):
    if old.get("output_sha256")!=sha256(table) or not old.get("zip_crc32_verified"):
        return False
    if old.get("source_bytes")!=info["bytes"] or not old.get("source_sha256"):
        return False
    fingerprint=scientific_fingerprint(cfg,data)["sha256"]
    if old.get("analysis_version")==3:
        return bool(old.get("scientific_fingerprint")==fingerprint and old.get("topology_sha256") and old.get("canonical_CA_mapping_sha256"))
    audit_path=out/"zenodo_legacy_checkpoint_audit.json"
    if old.get("analysis_version")!=2 or not audit_path.is_file():return False
    audit=json.loads(audit_path.read_text())
    matched=next((r for r in audit["checkpoints"] if r["trajectory_id"]==old["trajectory_id"]),None)
    return (audit.get("status")=="pass_retrospective_formula_and_raw_parity_audit" and
            audit.get("current_scientific_fingerprint")==fingerprint and matched is not None and
            matched["output_sha256"]==old["output_sha256"] and matched["source_sha256"]==old["source_sha256"])


def analyze_xtc_member(info,cfg,out,data,index,record,offline,archive_path):
    local_archive=archive_path.is_file()
    checkpoint=out/"zenodo_trajectory_checkpoints"
    checkpoint.mkdir(parents=True,exist_ok=True)
    member=info["member"];role,explanation=trajectory_role(member)
    identity=hashlib.sha256(member.encode()).hexdigest()[:16]
    cp=checkpoint/(identity+".json");table=checkpoint/(identity+".csv.gz")
    force_recompute=cfg["external_extension"].get("force_trajectory_recompute",False)
    if cp.exists() and table.exists() and not force_recompute and not (offline and local_archive):
        old=json.loads(cp.read_text())
        if checkpoint_is_verified(old,table,out,cfg,data,info):
            with gzip.open(table,"rt") as f:return {**old,"execution_mode":"verified_completed_checkpoint_replay"},list(csv.DictReader(f))
    if role.startswith("duplicate"):
        return {"trajectory_id":member,"role":role,"status":"excluded_duplicate_representation",
                "frames_analyzed":0,"reason":explanation,"zip_crc32":info["crc32"],"source_bytes":info["bytes"]},[]
    import mdtraj
    if mdtraj.__version__!="1.11.0":raise RuntimeError("raw XTC recomputation requires the pinned MDTraj 1.11.0 reader")
    reader=archive_path if local_archive else RemoteZipReader(record["links"]["self"],record["size"])
    with zipfile.ZipFile(reader) as archive:
        def open_trajectory(name):
            return archive.open(name) if local_archive else reader.open_member(archive,name)
        with open_trajectory(member) as stream:
            header=read_exact(stream,12)
        magic,natoms,step=struct.unpack(">iii",header)
        selected=None;topology=None;attempts=[]
        for candidate in trajectory_topology_candidates(member,index):
            payload=zenodo_member(archive,candidate,data)
            atoms=next(pdb_frames(payload))[1]
            attempts.append({"member":candidate,"atom_count":len(atoms)})
            if len(atoms)==natoms:
                mapping=topology_mapping(atoms,data,offline)
                if not mapping["missing_core"]:selected=mapping;topology=candidate;break
        if selected is None:
            return {"trajectory_id":member,"role":role,"status":"excluded_topology_or_core_mismatch",
                    "frames_analyzed":0,"reason":"No unambiguous matching-atom-count topology with all 269 core positions", "topologies_checked":attempts,"natoms":natoms},[]
        if shutil.disk_usage(data).free<MIN_FREE_BYTES+12*1024**2:raise OSError("disk reserve prevents bounded XTC spool")
        scorer=FrameScorer(selected);rows=[];number=0;digest=hashlib.sha256();byte_count=0
        class HashReader:
            def __init__(self,stream):self.stream=stream
            def read(self,n=-1):
                nonlocal byte_count
                b=self.stream.read(n);digest.update(b);byte_count+=len(b);return b
        with open_trajectory(member) as stream:
            for xyz,t,steps,boxes in xtc_batches(HashReader(stream),selected,data):
                for ca,frame_time,frame_step,box in zip(xyz,t,steps,boxes):
                    number+=1
                    rows.append({"trajectory_id":member,"frame":number,"source_member":member,
                                 "source_time_ps":float(frame_time),"source_step":int(frame_step),
                                 "coordinate_role":role,**scorer.score(ca,box)})
                write_json(out/"zenodo_stream_progress"/(identity+".json"),{
                    "trajectory":member,"frames_scored":number,"updated_utc":previous.utc_now(),
                    "status":"streaming_member; incomplete until CRC and final byte count pass"})
        if byte_count!=info["bytes"]:raise ValueError("decoded XTC byte count differs from ZIP index")
        fields=list(dict.fromkeys(k for r in rows for k in r))
        raw=io.StringIO();writer=csv.DictWriter(raw,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        table.write_bytes(gzip.compress(raw.getvalue().encode(),mtime=0))
        source_identity=topology_identity(topology,data,offline)
        result={"analysis_version":3,"execution_mode":"decoded_now","trajectory_id":member,"role":role,"status":"all_frames_analyzed","frames_analyzed":number,
                "reason":explanation,"topology_member":topology,"topology_atom_count":natoms,
                "core_positions":269,"DDB1_mapped_positions":len(selected["ddb1_ca_indices"]),
                "DDB1_scope":"mapped observed DDB1 construct; native positions 396-705 absent in the main 830-CA DDB1 construct",
                "source_bytes":info["bytes"],"source_bytes_read":byte_count,"source_sha256":digest.hexdigest(),"zip_crc32_verified":True,
                "trajectory_quality_fields":"per-frame adjacent CA and HB-fit diagnostics; severe compression flagged descriptively",
                "source_condition":trajectory_source_condition(member,source_identity["source_ligand_components"]),
                "source_ligand_components":source_identity["source_ligand_components"],
                "topology_sha256":source_identity["topology_sha256"],
                "canonical_CA_mapping_sha256":source_identity["canonical_CA_mapping_sha256"],
                "scientific_fingerprint":scientific_fingerprint(cfg,data)["sha256"],
                "fingerprint_origin":"recorded during this complete raw member decoding",
                "output_sha256":sha256(table),"raw_trajectory_retained":False,"trajectory_rows_file":table.relative_to(out).as_posix(),
                "sequence_mapping":selected["segments"]}
        write_json(cp,result)
        print(f"Zenodo {member}: {number} frames",flush=True)
    return result,rows


def trajectory_metric_summary(groups,all_rows):
    metadata={g["trajectory_id"]:g for g in groups}
    by_group=collections.defaultdict(list)
    for row in all_rows:by_group[row["trajectory_id"]].append(row)
    result=[]
    metrics=("closure_coordinate","DDB1_body_translation_A","DDB1_body_rotation_deg","DDB1_internal_RMSD_A",
             "TBD_body_translation_A","TBD_body_rotation_deg","TBD_internal_RMSD_A")
    for identity,rows in by_group.items():
        group=metadata[identity]
        adoption=group.get("quantitative_adoption","descriptive_computational_structure_comparison")
        for metric in metrics:
            values=np.asarray([float(r[metric]) for r in rows if r.get(metric) not in (None,"")],dtype=float)
            if not len(values):continue
            if not np.isfinite(values).all():raise ValueError("nonfinite trajectory summary observation")
            result.append({"trajectory_id":identity,"coordinate_role":group["role"],
                           "source_condition":group.get("source_condition","related model path stage"),
                           "quantitative_adoption":adoption,"frames_analyzed":group["frames_analyzed"],
                           "DDB1_mapped_positions":group.get("DDB1_mapped_positions",0),"metric":metric,
                           "finite_frame_count":len(values),"median":float(np.median(values)),
                           "p05":float(np.quantile(values,.05)),"p95":float(np.quantile(values,.95)),
                           "minimum":float(values.min()),"maximum":float(values.max())})
    return result


def trajectory_geometry_summary(groups,all_rows):
    metadata={g["trajectory_id"]:g for g in groups}
    by_group=collections.defaultdict(list)
    for row in all_rows:by_group[row["trajectory_id"]].append(row)
    result=[]
    fields=("CRBN_adjacent_CA_min_A","CRBN_adjacent_CA_max_A","CRBN_core_adjacent_CA_min_A",
            "CRBN_core_adjacent_CA_max_A","DDB1_adjacent_CA_min_A","DDB1_adjacent_CA_max_A",
            "HB_fit_RMSD_to_frozen_mean_A","CRBN_core_RMSD_to_frozen_mean_A")
    for identity,rows in by_group.items():
        record={"trajectory_id":identity,"role":metadata[identity]["role"],"frames_analyzed":len(rows)}
        for field in fields:
            values=[float(r[field]) for r in rows if r.get(field) not in (None,"")]
            record[field+"_observed_frame_count"]=len(values)
            record[field+"_minimum"]=min(values) if values else ""
            record[field+"_maximum"]=max(values) if values else ""
        for field in ("CRBN_adjacent_CA_below_2p5A","CRBN_core_adjacent_CA_below_2p5A","DDB1_adjacent_CA_below_2p5A"):
            values=[float(r[field]) for r in rows if r.get(field) not in (None,"")]
            record[field+"_observed_frame_count"]=len(values)
            record[field+"_flagged_frame_count"]=sum(value>0 for value in values) if values else ""
        result.append(record)
    return result


def zenodo_comparison(cfg:dict,out:Path,data:Path,offline=False) -> dict:
    record,index=zenodo_index(cfg,out,data,offline)
    summary_path=out/"zenodo_comparison_summary.json"
    archive_path=Path(cfg["external_extension"].get("zenodo_archive_path") or os.environ.get("CRBN_REVIEW_ZENODO_ARCHIVE") or data/"rMDautoencoderGitHubRepo.zip")
    local_archive=archive_path.is_file()
    if offline and not local_archive:
        if not summary_path.exists():raise FileNotFoundError("offline comparison requires retained frame metrics or raw archive")
        result=json.loads(summary_path.read_text())
        for r in result["retained_output_hashes"]:
            if sha256(out/r["file"])!=r["sha256"]:raise ValueError("offline frame metrics hash mismatch")
        predicted_path_comparison(cfg,out,data,True)
        return {**result,"this_run":"verified_compact_metric_replay; raw XTC frames were not recomputed",
                "compact_PDB_this_run":"recomputed from retained source members"}
    if local_archive:
        if archive_path.stat().st_size!=record["size"]:raise ValueError("local Zenodo archive size mismatch")
        md5=hashlib.md5()
        with archive_path.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024**2),b""):md5.update(chunk)
        if "md5:"+md5.hexdigest()!=record["checksum"]:raise ValueError("local Zenodo archive checksum mismatch")
    reader=archive_path if local_archive else RemoteZipReader(record["links"]["self"],record["size"],block_size=1024**2)
    groups=[];all_rows=[];condition_audits=[]
    checkpoint=out/"zenodo_trajectory_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    with zipfile.ZipFile(reader) as archive:
        path_rows,path_groups=predicted_path_comparison(cfg,out,data,offline,archive)
        all_rows+=path_rows;groups+=path_groups
    infos=[r for r in index if r["member"].endswith(".xtc")]
    pending=[]
    for info in infos:
        if trajectory_role(info["member"])[0].startswith("duplicate"):continue
        identity=hashlib.sha256(info["member"].encode()).hexdigest()[:16]
        cp=checkpoint/(identity+".json");table=checkpoint/(identity+".csv.gz")
        if (offline and local_archive) or cfg["external_extension"].get("force_trajectory_recompute",False) or not (
            cp.exists() and table.exists() and checkpoint_is_verified(json.loads(cp.read_text()),table,out,cfg,data,info)):
            pending.append(info)
    available_workers=max(0,int((shutil.disk_usage(data).free-MIN_FREE_BYTES)//(12*1024**2)))
    if pending and not available_workers:raise OSError("disk reserve prevents a bounded XTC spool")
    workers=max(1,min(3,len(pending) or 1,available_workers or 1,int(cfg["external_extension"].get("zenodo_workers",3))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results=executor.map(lambda info:analyze_xtc_member(info,cfg,out,data,index,record,offline,archive_path),infos)
        for result,rows in results:
            if result["status"]=="all_frames_analyzed":
                identity=topology_identity(result["topology_member"],data,offline)
                if identity["mapping"]["segments"]!=result["sequence_mapping"]:
                    raise ValueError("checkpoint canonical mapping changed")
                previous_condition=result.get("source_condition_at_acquisition",result.get("source_condition",""))
                result.update({k:identity[k] for k in ("topology_sha256","canonical_CA_mapping_sha256","source_ligand_components","structural_metal_components")})
                result["source_condition"]=trajectory_source_condition(result["trajectory_id"],identity["source_ligand_components"])
                result["quantitative_adoption"]="descriptive_computational_structure_comparison"
                condition_audits.append({"trajectory_id":result["trajectory_id"],"previous_acquisition_label":previous_condition,
                                         "final_source_condition":result["source_condition"],
                                         "ligand_components":";".join(identity["source_ligand_components"]),
                                         "structural_metal_components":";".join(identity["structural_metal_components"]),
                                         "label_changed":previous_condition!=result["source_condition"],
                                         "interpretation":"Observed organic ligand labels and workflow stage; structural zinc does not change the apo/ligand label. No equilibrium or independent-replicate inference."})
                if result.get("analysis_version")==2:
                    result["fingerprint_origin"]="Legacy acquisition had no scientific fingerprint; adopted under the explicit retrospective formula and complete small-member parity audit"
                    result["retrospective_scientific_fingerprint"]=scientific_fingerprint(cfg,data)["sha256"]
                cp=checkpoint/(hashlib.sha256(result["trajectory_id"].encode()).hexdigest()[:16]+".json")
                acquisition=json.loads(cp.read_text())
                if previous_condition!=result["source_condition"]:
                    acquisition.setdefault("source_condition_at_acquisition",previous_condition)
                acquisition.update({k:result[k] for k in ("source_condition","source_ligand_components","structural_metal_components")})
                acquisition["source_condition_audit"]="zenodo_source_condition_audit.csv; metadata correction only, numerical frame observations unchanged"
                write_json(cp,acquisition)
                for row in rows:
                    row["quantitative_adoption"]=result["quantitative_adoption"]
                    row["source_condition"]=result["source_condition"]
            groups.append(result);all_rows+=rows
    fields=list(dict.fromkeys(k for row in all_rows for k in row))
    with (out/"zenodo_frame_metrics.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
    write_csv(out/"zenodo_trajectory_metric_summary.csv",trajectory_metric_summary(groups,all_rows))
    write_csv(out/"zenodo_trajectory_geometry_summary.csv",trajectory_geometry_summary(groups,all_rows))
    write_csv(out/"zenodo_source_condition_audit.csv",condition_audits)
    write_json(out/"zenodo_trajectory_inventory.json",groups)
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in groups]
    write_csv(out/"zenodo_trajectory_inventory.csv",flat)
    used=[];indexed={row["member"]:row for row in infos}
    for group in groups:
        if group["status"]!="all_frames_analyzed":continue
        entry=indexed[group["trajectory_id"]]
        used.append({"member":group["trajectory_id"],"frames_analyzed":group["frames_analyzed"],
                     "bytes":entry["bytes"],"compressed_bytes":entry["compressed_bytes"],"zip_crc32":entry["crc32"],
                     "zip_crc32_verified":group["zip_crc32_verified"],"source_sha256":group["source_sha256"],
                     "topology_member":group["topology_member"],"topology_sha256":group["topology_sha256"],
                     "canonical_CA_mapping_sha256":group["canonical_CA_mapping_sha256"],
                     "scientific_fingerprint":group.get("scientific_fingerprint",group.get("retrospective_scientific_fingerprint","")),
                     "fingerprint_origin":group["fingerprint_origin"],"raw_member_retained":False,
                     "frame_table":group["trajectory_rows_file"],"frame_table_sha256":group["output_sha256"]})
    write_csv(out/"zenodo_used_member_manifest.csv",used)
    summary={"source_record":ZENODO_API,"archive_identity":record,"trajectory_groups":len(groups),
             "analyzed_frame_count":len(all_rows),"analyzed_groups":sum(r["frames_analyzed"]>0 for r in groups),
             "group_results":groups,"scope":"Biased computational structural comparison; DDB1 is a truncated construct where present",
             "body_metrics_reference":"first frame within each trajectory after CRBN HB(187-317) alignment to the frozen mean; not a dynamic decomposition of fluctuations",
             "periodic_boundary_handling":"CA-chain continuity and partner centroid nearest-image placement before HB alignment; source box vectors retained by reader",
             "not_independent_frame_replicates":True,"raw_XTC_retained":False,
             "offline_raw_recompute_requires":"the frozen Zenodo ZIP; compact metrics replay alone is not raw recomputation",
             "acquisition_scope":"Complete eligible member byte streams with per-member CRC and SHA-256; ZIP index and compact inputs also acquired. Whole-archive verification is recorded separately.",
             "analysis_complete":True,"all_XTC_members_accounted_for":len([g for g in groups if g["trajectory_id"].endswith(".xtc")])==len(infos),
             "xtc_archive_member_count":len(infos),
             "xtc_analyzed_comparison_count":sum(g["trajectory_id"].endswith(".xtc") and g["status"]=="all_frames_analyzed" for g in groups),
             "xtc_duplicate_representation_count":sum(g["status"]=="excluded_duplicate_representation" for g in groups),
             "xtc_quality_excluded_count":sum(g["status"]=="excluded_topology_or_core_mismatch" for g in groups),
             "frame_quantiles_are_descriptive_not_confidence_intervals":True,
             "retained_output_hashes":[{"file":f,"sha256":sha256(out/f)} for f in ("zenodo_frame_metrics.csv.gz","zenodo_trajectory_inventory.json","zenodo_trajectory_metric_summary.csv","zenodo_trajectory_geometry_summary.csv","zenodo_used_member_manifest.csv","zenodo_source_condition_audit.csv")],
             "this_run":"all eligible source members accounted for; completed checksum-verified checkpoints can be reused on resume"}
    write_json(out/"zenodo_scientific_fingerprint.json",scientific_fingerprint(cfg,data))
    write_json(summary_path,summary)
    return summary


def source_manifest(out:Path,data:Path):
    rows=[]
    for path in sorted(data.rglob("*")):
        if not path.is_file() or "reader_runtime" in path.parts or path.suffix==".xtc":continue
        rows.append({"path":path.relative_to(data).as_posix(),"bytes":path.stat().st_size,
                     "sha256":sha256(path),"role":"retained public source or acquisition metadata"})
    write_csv(out/"retained_external_sources.csv",rows)
    return {"files":len(rows),"source_table":"retained_external_sources.csv",
            "runtime_binaries_excluded":True,"raw_XTC_retained":False}


def final_availability_audit(cfg,out,data,offline=False):
    final_path=out/"oconnor_final_accessibility.json"
    if final_path.is_file():return json.loads(final_path.read_text())
    if offline:raise FileNotFoundError("final public-accessibility snapshot absent")
    temporary_out=out.parent.parent/"verification/external_final_accessibility"
    temporary_out.mkdir(parents=True,exist_ok=True)
    rows=oconnor_availability(cfg,temporary_out,data/"final_accessibility",False)
    write_json(final_path,rows);write_csv(out/"oconnor_final_accessibility.csv",rows)
    return rows


def run(config_path: str | Path, output_dir: str | Path, offline: bool = False) -> dict:
    config_path, out = Path(config_path), Path(output_dir)
    cfg=json.loads(config_path.read_text())
    data=out.parent.parent / "data/external"
    out.mkdir(parents=True,exist_ok=True); data.mkdir(parents=True,exist_ok=True)
    candidates=candidate_inputs(data)
    summary={"protocol_version":cfg["protocol_version"],"config_sha256":sha256(config_path)}
    summary["rcsb"]=acquisition_inventory(cfg,out,data,offline)
    summary["oconnor_accessibility"]=oconnor_availability(cfg,out,data,offline)
    summary["spatial"]=spatial_audit(out,data,candidates,offline)
    summary["variants"]=variant_audit(out,data,candidates,offline)
    summary["zenodo"]=zenodo_comparison(cfg,out,data,offline)
    digest=out/"zenodo_archive_digest.json"
    if not offline and cfg["external_extension"].get("verify_whole_archive",False):
        record,_=zenodo_index(cfg,out,data,False)
        summary["whole_archive_stream"]=stream_archive_digest(record["links"]["self"],record["size"],cfg["external_extension"]["zenodo_md5"],digest)
    elif digest.is_file():
        summary["whole_archive_stream"]={**json.loads(digest.read_text()),"this_run":"prior acquisition evidence; archive bytes were not reread"}
    else:
        summary["whole_archive_stream"]={"status":"no whole-archive verification record"}
    summary["oconnor_final_accessibility"]=final_availability_audit(cfg,out,data,offline)
    summary["retained_sources"]=source_manifest(out,data)
    write_json(out / "external_review_summary.json",summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"scripts/review_extensions_config.json")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--offline",action="store_true")
    args=parser.parse_args(argv)
    print(json.dumps(run(args.config,args.output_dir,args.offline),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
