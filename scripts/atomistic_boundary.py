"""Small OpenMM boundary-condition builders; not a validated CRBN MD protocol.

Coordinates are plain arrays in nm, masses in Da, and energies in kJ/mol.
Every public builder clones its input System.  No caller-owned System is mutated.
The six core restraints are a finite-k approximation to the static core gauge.
The tetrahedral virtual-site body has exactly six mechanical degrees of freedom
up to the OpenMM distance-constraint tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
from typing import Any

import numpy as np
import openmm as mm
from openmm import unit

try:
    from . import directional_mechanics as dm
except ImportError:  # direct scripts/ import
    import directional_mechanics as dm


# Common-platform force buffers use a 2**-32 fixed-point unit even in double
# precision. Scaling a linear CV and inversely scaling its outer derivative
# suppresses amplification of that inner-buffer error by a large gauge k.
# B is orthonormal, so scaled CV forces are <= 2**16 and cannot overflow the
# signed 64-bit inner buffer (which has 31 integer bits).
GAUGE_CV_SCALE = 2**16
_LINEAR_FORM = "bx*(x1-x0)+by*(y1-y0)+bz*(z1-z0)"
_LINEAR_PARAMETERS = ("bx", "by", "bz", "x0", "y0", "z0")


@dataclass
class BoundaryResult:
    system: mm.System
    positions_nm: np.ndarray | None
    metadata: dict[str, Any]


def _indices(system, indices, minimum=1):
    raw = list(indices)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in raw):
        raise ValueError("Particle indices must be integers")
    ids = [int(i) for i in raw]
    if len(ids) < minimum or len(set(ids)) != len(ids):
        raise ValueError("Particle indices must be unique and sufficiently numerous")
    if any(i < 0 or i >= system.getNumParticles() for i in ids):
        raise ValueError("Particle index outside System")
    return ids


def _xyz(values, count):
    if unit.is_quantity(values):
        values = values.value_in_unit(unit.nanometer)
    a = np.asarray(values, dtype=float)
    if a.shape != (count, 3) or not np.isfinite(a).all():
        raise ValueError(f"Expected finite ({count}, 3) coordinates in nm")
    return a.copy()


def _clone(system):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def _mass(system, i):
    return float(system.getParticleMass(i).value_in_unit(unit.dalton))


def _inertia(x, masses):
    center = np.average(x, axis=0, weights=masses)
    r = x - center
    tensor = np.eye(3) * np.sum(masses * np.sum(r*r, axis=1)) - (r*masses[:, None]).T @ r
    return center, tensor


def _gauge_expression(prefix):
    return (f"0.5*{prefix}_gauge_k/({GAUGE_CV_SCALE}^2)*("
            + "+".join(f"g{i}^2" for i in range(6)) + ")")


def _linear_shape(force, expression, global_names=()):
    return (type(force) is mm.CustomCompoundBondForce
            and force.getNumParticlesPerBond() == 1
            and force.getEnergyFunction() == expression
            and force.getNumTabulatedFunctions() == 0
            and tuple(force.getPerBondParameterName(i) for i in range(force.getNumPerBondParameters()))
            == _LINEAR_PARAMETERS
            and tuple(force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters()))
            == tuple(global_names))


def _our_gauge(force):
    if type(force) is not mm.CustomCVForce or not force.getName().startswith("atomistic_boundary:gauge:"):
        return False
    prefix = force.getName().rsplit(":", 1)[1]
    return (force.getEnergyFunction() == _gauge_expression(prefix) and force.getNumCollectiveVariables() == 6
            and all(force.getCollectiveVariableName(i) == f"g{i}"
                    and _linear_shape(force.getCollectiveVariable(i), f"{GAUGE_CV_SCALE}*({_LINEAR_FORM})")
                    for i in range(6)))


def _our_probe(force):
    if not force.getName().startswith("atomistic_boundary:probe:"):
        return False
    prefix = force.getName().rsplit(":", 1)[1]
    return _linear_shape(force, f"-{prefix}_force_h*({_LINEAR_FORM})", [f"{prefix}_force_h"])


def _linear_force(ids, ref, coefficients, expression):
    # Unlike CustomExternalForce (float per-particle parameter buffers), this
    # kernel selects double per-bond buffers when the platform precision is
    # double. One one-particle bond contributes each term of the same sum.
    # OpenMM 8.4 CommonKernels.cpp:1274 vs :1391; ComputeParameterSet.h:51.
    force = mm.CustomCompoundBondForce(1, expression)
    for name in _LINEAR_PARAMETERS:
        force.addPerBondParameter(name)
    for j, i in enumerate(ids):
        force.addBond([i], [*coefficients[3*j:3*j+3], *ref[j]])
    return force


def _preflight_body(system, body_indices):
    ids = _indices(system, body_indices)
    body = set(ids)
    supported = (mm.HarmonicBondForce, mm.HarmonicAngleForce,
                 mm.PeriodicTorsionForce, mm.RBTorsionForce,
                 mm.NonbondedForce, mm.CMMotionRemover,
                 mm.MonteCarloBarostat, mm.CustomExternalForce)
    for force in system.getForces():
        if type(force) not in supported and not _our_gauge(force) and not _our_probe(force):
            raise ValueError(f"Unsupported force for body conversion: {type(force).__name__}")
        if type(force) is mm.NonbondedForce and force.getNumParticles() != system.getNumParticles():
            raise ValueError("NonbondedForce particle count differs from System")
        if type(force) is mm.MonteCarloBarostat and force.getFrequency() != 0:
            raise ValueError("Boundary validation requires NVT: disable the barostat first")
    removed = []
    for k in range(system.getNumConstraints()):
        i, j, _ = system.getConstraintParameters(k)
        if (i in body) != (j in body):
            raise ValueError("A distance constraint crosses the body boundary")
        if i in body:
            removed.append(k)
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            site = system.getVirtualSite(i)
            parents = {site.getParticle(j) for j in range(site.getNumParticles())}
            if i in body or parents.intersection(body):
                raise ValueError("Existing virtual sites involving the body are unsupported")
    masses = np.array([_mass(system, i) for i in ids])
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("All input body particles must have positive finite mass")
    return ids, removed, masses


def add_core_gauge_and_probe(system, core_indices, reference_nm, q,
                             gauge_k, force_h, *, prefix="core"):
    """Return a cloned System with a rank-six restraint and fixed linear probe.

    q is a unit ambient vector (3*n or n*3), or unit internal coefficients
    (3*n-6) in dm.internal_basis(reference_core_nm).  It is never renormalized.
    reference_nm may contain all System particles or just the ordered core.
    Context parameters are <prefix>_gauge_k (kJ/mol/nm^2) and
    <prefix>_force_h (kJ/mol/nm).  The probe is -h*q.(x-reference).
    No minimum-image wrapping is performed: supply a coherent molecular image.
    Use Reference or GPU Precision=double. Single/mixed precision does not
    preserve the double coefficients. Stored gauge CV values are scaled by
    GAUGE_CV_SCALE; the potential and physical parameters are unchanged.
    """
    ids = _indices(system, core_indices, minimum=4)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prefix):
        raise ValueError("Invalid parameter prefix")
    if not np.isfinite([gauge_k, force_h]).all() or gauge_k < 0:
        raise ValueError("gauge_k must be nonnegative and both parameters finite")
    if any(not np.isfinite(_mass(system, i)) or _mass(system, i) <= 0
           or system.isVirtualSite(i) for i in ids):
        raise ValueError("The measured core must consist of massive, nonvirtual particles")
    if unit.is_quantity(reference_nm):
        reference_nm = reference_nm.value_in_unit(unit.nanometer)
    ref = np.asarray(reference_nm, dtype=float)
    if ref.shape == (system.getNumParticles(), 3):
        ref = _xyz(ref, system.getNumParticles())[ids]
    else:
        ref = _xyz(ref, len(ids))
    B = dm.rigid_basis(ref)
    v = np.asarray(q, dtype=float).reshape(-1)
    if v.size == 3*len(ids)-6:
        v = dm.internal_basis(ref) @ v
    if v.size != 3*len(ids) or not np.isfinite(v).all():
        raise ValueError("q has invalid size or nonfinite values")
    if abs(np.linalg.norm(v)-1) > 1e-10 or np.linalg.norm(B.T@v) > 1e-10:
        raise ValueError("q must already be unit norm and core-internal")
    names = {f"{prefix}_gauge_k", f"{prefix}_force_h"}
    for force in system.getForces():
        if hasattr(force, "getNumGlobalParameters"):
            existing = {force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters())}
            if names.intersection(existing):
                raise ValueError("Boundary global parameter name already exists")
    gauge = mm.CustomCVForce(_gauge_expression(prefix))
    gauge.setName(f"atomistic_boundary:gauge:{prefix}")
    gauge.addGlobalParameter(f"{prefix}_gauge_k", float(gauge_k))
    for a in range(6):
        cv = _linear_force(ids, ref, B[:, a], f"{GAUGE_CV_SCALE}*({_LINEAR_FORM})")
        gauge.addCollectiveVariable(f"g{a}", cv)
    probe = _linear_force(ids, ref, v, f"-{prefix}_force_h*({_LINEAR_FORM})")
    probe.setName(f"atomistic_boundary:probe:{prefix}")
    probe.addGlobalParameter(f"{prefix}_force_h", float(force_h))
    out = _clone(system)
    out.addForce(gauge)
    out.addForce(probe)
    return BoundaryResult(out, None, {
        "kind": "finite_rank6_gauge_and_linear_probe", "approximation": True,
        "core_indices": ids, "gauge_rank": 6, "measured_dimension": 3*len(ids)-6,
        "reference_core_nm": ref.tolist(), "rigid_basis": B.tolist(),
        "q_ambient": v.tolist(), "gauge_parameter": f"{prefix}_gauge_k",
        "probe_parameter": f"{prefix}_force_h", "gauge_units": "kJ/mol/nm^2",
        "probe_units": "kJ/mol/nm", "requires_gauge_strength_convergence": True,
        "coefficient_storage": "CustomCompoundBondForce per-bond; double requires Precision=double",
        "gauge_cv_scale": GAUGE_CV_SCALE,
        "gpu_force_quantum_kj_mol_nm": 2.**-32,
        "gpu_force_quantization_bound": "2^-32*(7+abs(gauge_k)*sum(abs(B.T*delta))/gauge_cv_scale) per component; excludes floating arithmetic",
    })


def make_fixed_body(system, body_indices):
    """Clone, remove internal body constraints, and set its masses to zero."""
    ids, removed, masses = _preflight_body(system, body_indices)
    out = _clone(system)
    for k in reversed(removed):
        out.removeConstraint(k)
    for i in ids:
        out.setParticleMass(i, 0)
    return BoundaryResult(out, None, {
        "kind": "fixed_body", "body_indices": ids, "body_dof": 0,
        "original_mass_da": float(masses.sum()), "removed_constraint_indices": removed,
        "nvt_required": True,
    })


def make_rigid_body(system, positions_nm, body_indices):
    """Clone and replace one 3D body by a mass/inertia-matched rigid tetrahedron.

    All original body atoms become LocalCoordinatesSites.  They retain their
    interaction parameters, but their mass moves to four noninteracting anchors.
    This freezes ALL body atoms, including side chains; it is not a flexible
    backbone model.  Planar/linear or ill-conditioned bodies are rejected.
    """
    ids, removed, masses = _preflight_body(system, body_indices)
    if len(ids) < 4:
        raise ValueError("A rigid tetrahedron requires at least four body atoms")
    xyz = _xyz(positions_nm, system.getNumParticles())
    center, inertia = _inertia(xyz[ids], masses)
    principal, axes = np.linalg.eigh(inertia)
    total_mass = float(masses.sum())
    squared = (principal.sum()-2*principal)/(2*total_mass)
    if np.min(squared) <= 1e-10*np.max(squared) or np.max(squared) <= 0:
        raise ValueError("Body must be nonplanar and well conditioned")
    signs = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]])
    anchors = (signs*np.sqrt(squared)) @ axes.T + center
    anchor_center, anchor_inertia = _inertia(anchors, np.full(4, total_mass/4))
    if not np.allclose(anchor_inertia, inertia, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("Tetrahedron does not preserve the inertia tensor")
    ex = anchors[1]-anchors[0]
    ex /= np.linalg.norm(ex)
    ez = np.cross(ex, anchors[2]-anchors[0])
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    local = (xyz[ids]-anchors[0]) @ np.column_stack([ex, ey, ez])
    out = _clone(system)
    for k in reversed(removed):
        out.removeConstraint(k)
    anchor_ids = [out.addParticle(total_mass/4) for _ in range(4)]
    for force in out.getForces():
        if type(force) is mm.NonbondedForce:
            for _ in range(4):
                force.addParticle(0, 1, 0)
    for i, j in combinations(range(4), 2):
        out.addConstraint(anchor_ids[i], anchor_ids[j], float(np.linalg.norm(anchors[i]-anchors[j])))
    for i, position in zip(ids, local):
        out.setParticleMass(i, 0)
        out.setVirtualSite(i, mm.LocalCoordinatesSite(
            *anchor_ids[:3], mm.Vec3(1,0,0), mm.Vec3(-1,1,0),
            mm.Vec3(-1,0,1), mm.Vec3(*position)))
    return BoundaryResult(out, np.vstack([xyz, anchors]), {
        "kind": "exact_all_atom_rigid_body", "body_indices": ids,
        "anchor_indices": anchor_ids, "body_dof": 6, "anchor_constraints": 6,
        "original_mass_da": total_mass, "anchor_mass_da": total_mass/4,
        "center_of_mass_nm": center.tolist(), "anchor_center_of_mass_nm": anchor_center.tolist(),
        "inertia_da_nm2": inertia.tolist(), "anchor_inertia_da_nm2": anchor_inertia.tolist(),
        "principal_inertia_da_nm2": principal.tolist(),
        "removed_constraint_indices": removed, "nvt_required": True,
        "exactness": "Rigid geometry up to integrator constraint tolerance; engine validation required",
    })
