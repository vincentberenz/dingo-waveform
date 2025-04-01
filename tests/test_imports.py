import pytest

from dingo_waveform.approximant import Approximant
from dingo_waveform.imports import check_function_signature, import_entity
from dingo_waveform.polarizations import Polarization


def test_import_entity() -> None:

    # FrequencyDomain should be imported with success
    d, module, d_name = import_entity("dingo_waveform.domains.FrequencyDomain")
    assert d_name == "FrequencyDomain"
    assert module == "dingo_waveform.domains"
    assert d.__name__ == "FrequencyDomain"

    # get_approximant should be imported with success
    f, module, f_name = import_entity("dingo_waveform.approximant.get_approximant")
    assert f_name == "get_approximant"
    assert module == "dingo_waveform.approximant"
    assert check_function_signature(f, [Approximant], int)

    # this import should fail
    with pytest.raises(ImportError):
        import_entity("package.module.function")

    # this should fail as well, except if an italian starts to program domains.
    with pytest.raises(ImportError):
        import_entity("dingo_waveform.domains.RavioliDomain")


def test_check_function_signature() -> None:

    def fn(a: int, b: float) -> float:
        return a + b

    # correct signature for fn
    assert check_function_signature(fn, [int, float], float)

    # incorrect signature for fn
    assert not check_function_signature(fn, [int], str)

    # suitable transform for instances of Domain
    def transform(p: Polarization) -> Polarization:
        return p

    assert check_function_signature(transform, [Polarization], Polarization)

    # unsuitable transform for instances of Domain
    def not_transform(p: Polarization, a: int) -> Polarization:
        return p

    assert not check_function_signature(not_transform, [Polarization], Polarization)

    # suitable transform for instances of Domain, with optional kwarg
    def kwargs_transform(p: Polarization, a: int = 0) -> Polarization:
        return p

    assert check_function_signature(kwargs_transform, [Polarization], Polarization)
