"""Tools for studying identifiability of scaling-law extrapolation."""

from .certificate import (
    CertificateInterval,
    certificate_curve,
    certificate_interval,
    design_matrix,
    simultaneous_gaussian_multiplier,
)
from .spectral import (
    MatchedPair,
    asymptotic_tail,
    hidden_weight_for_tolerance,
    mixture_risk,
    tail_basis,
)

__all__ = [
    "CertificateInterval",
    "MatchedPair",
    "asymptotic_tail",
    "certificate_curve",
    "certificate_interval",
    "design_matrix",
    "hidden_weight_for_tolerance",
    "mixture_risk",
    "simultaneous_gaussian_multiplier",
    "tail_basis",
]
