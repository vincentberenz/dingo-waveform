from typing import TypeAlias

import lalsimulation as LS

Approximant: TypeAlias = int


def get_approximant(approximant: str) -> Approximant:
    return Approximant(LS.GetApproximantFromString(approximant))
