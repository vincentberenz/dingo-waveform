from typing import NewType

import lalsimulation as LS

Approximant = NewType("Approximant", str)

# Predefined Approximant values
TD_Approximant = "SEOBNRv4PHM"
FD_Approximant = "IMRPhenomXPHM"
SEOBNR_Approximants = (
    "SEOBNRv5HM", "SEOBNRv5EHM", "SEOBNRv5PHM"
)
ConditionedExtraTime_Approximant = (
    "SEOBNRv5HM", "SEOBNRv5PHM"   
)





def is_gwsignal_approximant(approximant: Approximant)->bool:
    try:
        LS.SimInspiralGetApproximantFromString(approximant)
        return True
    except: 
        return False


def get_approximant(approximant: Approximant) -> int:
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
    return LS.GetApproximantFromString(approximant)


def get_approximant_description(approximant: int) -> str:
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
