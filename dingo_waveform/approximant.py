from typing import NewType

import lalsimulation as LS

Approximant = NewType("Approximant", int)

TD_Approximant = Approximant(52)  # 'SEOBNRv4PHM'
FD_Approximant = Approximant(101)  # 'IMRPhenomXPHM'


def get_approximant(approximant: str) -> Approximant:
    return Approximant(LS.GetApproximantFromString(approximant))


def get_approximant_description(approximant: Approximant) -> str:
    return LS.GetStringFromApproximant(int(approximant))
