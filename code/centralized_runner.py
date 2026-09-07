"""Centralized experiment orchestration API.

This compatibility boundary exposes the unchanged centralized runner from
the shared runtime. Keeping the implementation in one place prevents result,
early-stopping, and checkpoint semantics from diverging during Phase 2A.
"""

from experiment_runtime import centralized

__all__ = ["centralized"]
