"""
simulate_rl — minimal package for LLM comparison evaluation.

This is a stripped-down version of the full simulator package.
Only the StudentSimulator and PROFILES are exposed, because they
are the only pieces compare_llms.py needs.
"""

from simulate_rl.profiles import (  # noqa: F401
    PROFILES,
    StudentSimulator,
)

__all__ = ["PROFILES", "StudentSimulator"]
