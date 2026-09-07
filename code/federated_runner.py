"""Federated experiment orchestration API.

The current client lifecycle, ModuleALA/ModuleLocal dispatch, aggregation,
validation, and serialization remain the unchanged runtime implementation.
This module is the explicit import boundary for future runner extraction.
"""

from experiment_runtime import federated

__all__ = ["federated"]
