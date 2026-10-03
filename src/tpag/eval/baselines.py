"""Baselines for the TPAG replication package."""

from __future__ import annotations
from ..bound import compute_bound, multiplicative_certificate
from ..structure import Structure


def multiplicative_bound(structure: Structure, eps: dict[str, float]) -> float:
    """Independence/chain-rule certificate: sum over cuts of product of marginals."""
    return multiplicative_certificate(structure.cut_sets(), eps)


def boole_bound(structure: Structure, eps_upper: dict[str, float]) -> float:
    """Sound *ablation* of TPAG: marginal Fréchet per cut + Boole union, with **no**
    measured joints and **no** Hunter--Worsley correction (the naive conservative
    composition a careful practitioner would reach for)."""
    return compute_bound(
        structure.cut_sets(), eps_upper, pair_upper=None, cut_joint_lower=None
    ).b_boole


def frechet_joints_bound(
    structure: Structure,
    eps_upper: dict[str, float],
    pair_upper: dict[frozenset[str], float] | None,
) -> float:
    """Sound *ablation*: :func:`boole_bound` plus measured pairwise joints (Fréchet
    tightening *within* cuts), still without the Hunter--Worsley correction. Isolates
    the contribution of the measured co-violation matrix ``K``."""
    return compute_bound(
        structure.cut_sets(), eps_upper, pair_upper=pair_upper or None, cut_joint_lower=None
    ).b_boole
