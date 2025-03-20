"""
This module defines a custom type for Approximant and provides functions to work with it.
"""

from typing import NewType

import lalsimulation as LS

Approximant = NewType("Approximant", int)

# Predefined Approximant values
TD_Approximant = Approximant(52)  # 'SEOBNRv4PHM'
FD_Approximant = Approximant(101)  # 'IMRPhenomXPHM'


def get_approximant(approximant: str) -> Approximant:
    """
    Converts a string representation of an approximant to its integer value.

    Parameters
    ----------
    approximant
        The string representation of the approximant.

    Returns
    -------
    The integer value of the approximant.
    """
    return Approximant(LS.GetApproximantFromString(approximant))


def get_approximant_description(approximant: Approximant) -> str:
    """
    Converts an integer value of an approximant to its string description.

    Parameters
    ----------
    approximant
        The integer value of the approximant.

    Returns
    -------
        The string description of the approximant.
    """
    return LS.GetStringFromApproximant(int(approximant))
