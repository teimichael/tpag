"""Init for the TPAG replication package."""

from __future__ import annotations

__version__ = "0.1.0"

from .bound import BoundResult, compute_bound, multiplicative_certificate
from .certificate import Certificate, certify, certify_exact
from .containment import ContainmentController, GlobalContainment
from .contracts import Contract
from .estimation import EstimationModel, clopper_pearson_lower, clopper_pearson_upper
from .events import Event, EventKind, Trace
from .graph import OrchestrationGraph
from .monitors import Verdict
from .predicates import Predicate, PredicateKind
from .runtime import TPAGRuntime
from .structure import (
    Atom,
    ExplicitCutSets,
    Parallel,
    Series,
    check_adequacy,
    parallel,
    redundancy,
    series,
)
from .vectorclock import VectorClock, VectorClockEngine

__all__ = [
    "__version__",
    # framework
    "Contract",
    "Event",
    "EventKind",
    "Trace",
    "OrchestrationGraph",
    "Verdict",
    "Predicate",
    "PredicateKind",
    # structure
    "Atom",
    "Series",
    "Parallel",
    "ExplicitCutSets",
    "series",
    "parallel",
    "redundancy",
    "check_adequacy",
    # causality
    "VectorClock",
    "VectorClockEngine",
    # estimation + bound + certificate (the crown jewel)
    "clopper_pearson_upper",
    "clopper_pearson_lower",
    "EstimationModel",
    "compute_bound",
    "multiplicative_certificate",
    "BoundResult",
    "Certificate",
    "certify",
    "certify_exact",
    # runtime + containment
    "TPAGRuntime",
    "ContainmentController",
    "GlobalContainment",
]
