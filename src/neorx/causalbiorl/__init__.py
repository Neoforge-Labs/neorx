"""
CausalBioRL — Causal Reinforcement Learning Environments for Biological System Control.

Provides Gymnasium-compatible RL environments that simulate biological systems
(gene expression, metabolic flux, cell populations) and causal RL agents that
use structural causal models as world models.

Reference:
    CausalBioRL: Causal Reinforcement Learning Environments for
    Biological System Control (2026).
"""

from neorx.causalbiorl.envs.registration import register_envs

from . import agents, causal, envs  # noqa: F401,E402
from .agents.causal_agent import CausalAgent  # noqa: F401,E402
from .envs.drug_discovery import DrugDiscoveryEnv  # noqa: F401,E402

__version__ = "0.1.0"
__all__ = ["envs", "agents", "causal", "DrugDiscoveryEnv", "CausalAgent"]

# Register Gymnasium environments on import
register_envs()
