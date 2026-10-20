"""
CFCL Gadepan Complex - First-Principles Chemical Engineering & Metallurgy Engine.
Provides rigorous thermodynamic equilibrium solvers, transient thermal stress equations,
Larson-Miller creep damage accumulators, and stoichiometric synthesis loop balances.
Sanitized DCS Telemetry under strict NDA.
"""

from src.physics.metallurgy import (
    TubeMetallurgyEngine,
    VonMisesStressResult,
    CreepDamageResult,
)
from src.physics.kinetics import (
    SMRKineticsEngine,
    EquilibriumSlipResult,
)
from src.physics.synthesis_loop import (
    AmmoniaSynthesisLoopEngine,
    LoopMaterialBalanceResult,
    UreaStoichiometricResult,
)

__all__ = [
    "TubeMetallurgyEngine",
    "VonMisesStressResult",
    "CreepDamageResult",
    "SMRKineticsEngine",
    "EquilibriumSlipResult",
    "AmmoniaSynthesisLoopEngine",
    "LoopMaterialBalanceResult",
    "UreaStoichiometricResult",
]
