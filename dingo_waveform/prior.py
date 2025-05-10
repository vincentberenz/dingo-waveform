from dataclasses import Field, asdict, dataclass, fields, make_dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from bilby.core.prior import Cosine, Sine, Uniform
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    fill_from_fixed_priors,
)
from bilby.gw.prior import BBHPriorDict

from .imports import read_file
from .logging import TableStr
from .waveform_parameters import WaveformParameters


class BBHExtrinsicPriorDict(BBHPriorDict):
    """
    Subclass of BBHPriorDict not requiring mass parameters.
    It supports methods for estimating the standardization parameters.

    TODO:
        * Add support for zenith/azimuth
        * Defaults?
    """

    def default_conversion_function(self, sample):
        # Overwrite the default_conversion_function of the superclass
        # BBHPriorDict. Convert sample to LAL binary black hole parameters.

        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)

        # The previous call sometimes adds phi_jl, phi_12 parameters. These are
        # not needed so they can be deleted.
        if "phi_jl" in out_sample.keys():
            del out_sample["phi_jl"]
        if "phi_12" in out_sample.keys():
            del out_sample["phi_12"]

        return out_sample

    def mean_std(
        self, keys: List[str], sample_size=50000, force_numerical=False
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate the mean and standard deviation over the prior.

        Parameters
        ----------
        keys
            A list of desired parameter names
        sample_size
            For nonanalytic priors, number of samples to use to estimate the
            result.
        force_numerical
            Whether to force a numerical estimation of result, even when
            analytic results are available (useful for testing)

        Returns
        -------
        The mean and standard deviation dictionaries.

        TODO: Fix for constrained priors. Shouldn't be an issue for extrinsic parameters.
        """
        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}

        if not force_numerical:
            # First try to calculate analytically (works for standard priors)
            estimation_keys = []
            for key in keys:
                p = self[key]
                # A few analytic cases
                if isinstance(p, Uniform):
                    mean[key] = (p.maximum + p.minimum) / 2.0
                    std[key] = np.sqrt((p.maximum - p.minimum) ** 2.0 / 12.0).item()
                elif isinstance(p, Sine) and p.minimum == 0.0 and p.maximum == np.pi:
                    mean[key] = np.pi / 2.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                elif (
                    isinstance(p, Cosine)
                    and p.minimum == -np.pi / 2
                    and p.maximum == np.pi / 2
                ):
                    mean[key] = 0.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                else:
                    estimation_keys.append(key)
        else:
            estimation_keys = keys

        # For remaining parameters, estimate numerically
        if len(estimation_keys) > 0:
            samples = self.sample_subset(keys, size=sample_size)
            samples = self.default_conversion_function(samples)
            for key in estimation_keys:
                if key in samples.keys():
                    mean[key] = np.mean(samples[key]).item()
                    std[key] = np.std(samples[key]).item()

        return mean, std


@dataclass
class ExtrinsicPriors(TableStr):
    dec: Union[str, float] = (
        "bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2)"
    )
    ra: Union[str, float] = (
        'bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic")'
    )
    geocent_time: Union[str, float] = (
        "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)"
    )
    psi: Union[str, float] = (
        'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic")'
    )
    luminosity_distance: Union[str, float] = (
        "bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0)"
    )

    def mean_std(self, keys: List[str], sample_size=50000, force_numerical=False):
        """
        Calculate the mean and standard deviation over the prior.

        Parameters
        ----------
        keys
            A list of desired parameter names
        sample_size
            For nonanalytic priors, number of samples to use to estimate the
            result.
        force_numerical
            Whether to force a numerical estimation of result, even when
            analytic results are available (useful for testing)

        Returns
        -------
        The mean and standard deviation dictionaries.
        """
        bbh_prior_dict = BBHExtrinsicPriorDict(asdict(self))  # type: ignore
        return bbh_prior_dict.mean_std(
            keys, sample_size=sample_size, force_numerical=force_numerical
        )

    def sample(self) -> WaveformParameters:
        """
        Generate a single sample of waveform parameters.

        Returns
        -------
        A single sample of waveform parameters.
        """
        return self.samples(1)[0]

    def samples(self, nb_samples) -> List[WaveformParameters]:
        """
        Generate multiple samples of waveform parameters.

        Parameters
        ----------
        nb_samples
            The number of samples to generate.

        Returns
        -------
        A list of waveform parameters samples.
        """
        # type ignore:
        # I could not find how to specify this mixin could be superclass
        # only for dataclass. (I tried to use Protocol, but no success)
        bbh_prior_dict = BBHExtrinsicPriorDict(asdict(self))  # type: ignore
        return [
            WaveformParameters(**bbh_prior_dict.sample()) for _ in range(nb_samples)
        ]


@dataclass
class IntrinsicPriors(TableStr):
    mass_1: Union[str, float] = (
        "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)"
    )
    mass_2: Union[str, float] = (
        "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)"
    )
    mass_ratio: Union[str, float] = (
        "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)"
    )
    chirp_mass: Union[str, float] = (
        "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)"
    )
    luminosity_distance: Union[str, float] = 1000.0
    theta_jn: Union[str, float] = "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)"
    phase: Union[str, float] = (
        'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")'
    )
    a_1: Union[str, float] = "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)"
    a_2: Union[str, float] = "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)"
    tilt_1: Union[str, float] = "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)"
    tilt_2: Union[str, float] = "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)"
    phi_12: Union[str, float] = (
        'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")'
    )
    phi_jl: Union[str, float] = (
        'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")'
    )
    geocent_time: Union[str, float] = 0.0

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "IntrinsicPriors":
        parameters = read_file(file_path)
        return cls(**parameters)

    def sample(self) -> WaveformParameters:
        """
        Generate a single sample of waveform parameters.

        Returns
        -------
        A single sample of waveform parameters.
        """
        return self.samples(1)[0]

    def samples(self, nb_samples: int) -> List[WaveformParameters]:
        """
        Generate multiple samples of waveform parameters.

        Parameters
        ----------
        nb_samples
            The number of samples to generate.

        Returns
        -------
        A list of waveform parameters samples.
        """
        # type ignore:
        # I could not find how to specify this mixin could be superclass
        # only for dataclass. (I tried to use Protocol, but no success)
        bbh_prior_dict = BBHPriorDict(asdict(self))  # type: ignore
        return [
            WaveformParameters(**bbh_prior_dict.sample()) for _ in range(nb_samples)
        ]


def _create_priors_dataclass() -> Type:
    # Create a dataclass combining fields from IntrinsicPriors and ExtrinsicPriors.
    # It is called when this module is imported first, so that the class 'Priors'
    # becomes part of the user API.

    # Get fields from both classes
    extrinsic_fields = list(fields(ExtrinsicPriors))
    intrinsic_fields = list(fields(IntrinsicPriors))

    # luminosity_distance and geocent_time are both intrinsic and extrinsic priors.
    # We use here the default value as an extrinsic prior.
    all_fields_ = extrinsic_fields + [
        f
        for f in intrinsic_fields
        if f.name not in [f_.name for f_ in extrinsic_fields]
    ]
    all_fields = [(f.name, f.type, f.default) for f in all_fields_]

    # will be used in the _PriorSampling._get_prior method right below
    _PriorType = TypeVar("_PriorType", IntrinsicPriors, ExtrinsicPriors)

    class _PriorSampling:

        # We are creating the Priors dataclass. The class we are constructing
        # will be inherating from _PriorSampling. This is a convenient way
        # to add methods to the Priors dataclass we are constructing.

        # _PriorType will be either ExtrinsicPriors or IntrinsicPriors.
        # Using the TypeVar _PriorType indicate to mypy that if the target type
        # is ExtrinsicPriors, then the return type is an instance of ExtrinsicPriors
        # (same for IntrinsicPriors).
        def _get_prior(self, target_type: Type[_PriorType]) -> _PriorType:

            # type ignore:
            # we know that 'self' will be an instance of Prior, which is a dataclass
            # (i.e. asdict will be supported). mypy does not detect this.
            d_ = asdict(self)  # type: ignore
            d = {
                k: v
                for k, v in d_.items()
                if k in [f.name for f in fields(target_type)]
            }
            return target_type(**d)

        def get_intrinsic_priors(self) -> IntrinsicPriors:
            """
            Get the intrinsic priors.

            Returns
            -------
            The instance of IntrinsicPriors corresponding to this
            instance of Priors.
            """
            return self._get_prior(IntrinsicPriors)

        def get_extrinsic_priors(self) -> ExtrinsicPriors:
            """
            Get the extrinsic priors.

            Returns
            -------
            The instance of ExtrinsicPriors corresponding to this
            instance of Priors.
            """
            return self._get_prior(ExtrinsicPriors)

        def sample(self) -> WaveformParameters:
            """
            Generate a single sample of waveform parameters.

            Returns
            -------
            A single sample of waveform parameters.
            """
            return self.samples(1)[0]

        def samples(self, nb_samples: int) -> List[WaveformParameters]:
            """
            Generate multiple samples of waveform parameters.

            Parameters
            ----------
            nb_samples
                The number of samples to generate.

            Returns
            -------
            A list of waveform parameters samples.
            """
            ip = self.get_intrinsic_priors()
            ep = self.get_extrinsic_priors()
            intrinsic_wfs: List[WaveformParameters] = ip.samples(nb_samples)
            extrinsic_wfs: List[WaveformParameters] = ep.samples(nb_samples)

            def _get_wf(
                intrinsic_wf: WaveformParameters, extrinsic_wf: WaveformParameters
            ):
                """
                Combine intrinsic and extrinsic waveform parameters.

                Parameters
                ----------
                intrinsic_wf
                    The intrinsic waveform parameters.
                extrinsic_wf
                    The extrinsic waveform parameters.

                Returns
                -------
                The combined waveform parameters.
                """
                intrinsic_dict = asdict(intrinsic_wf)
                extrinsic_dict = asdict(extrinsic_wf)
                priors_dict = intrinsic_dict
                for k, v in extrinsic_dict.items():
                    if v is not None:
                        priors_dict[k] = v
                return WaveformParameters(**priors_dict)

            return [_get_wf(iw, ew) for iw, ew in zip(intrinsic_wfs, extrinsic_wfs)]

    # this 'constructs' the dataclass 'Priors'.
    # This class will be called 'Priors' (first argument), it will have the fields
    # 'all_fields' (combined fiels of extrinsic and intrinsic priors) and will
    # inheritate from TableStr (for table representation) and _PriorSample (for the
    # methods 'sample' and 'samples')
    return make_dataclass("Priors", all_fields, bases=(TableStr, _PriorSampling))


# Priors is a dataclass that have the combined fields of IntrinsicPriors and ExtrinsicPriors.
# It also has the 'sample' method which returns a list of WaveformParameters.
# Once created here (which happens when this module is imported), user can create instances of
# Priors.
# This class is created programmatically at runtime. I could not find a simpler way
# to do this (did I miss something ?)
Priors = _create_priors_dataclass()
"""
Dataclass that includes all fields of the IntrinsicPriors and the ExtrinsicPriors dataclasses.
"""


def prior_split(
    waveform_parameters: WaveformParameters,
    intrinsic_luminosity_distance: Optional[float] = 100.0,
    intrinsic_geocent_time: Optional[float] = 0.0,
) -> Tuple[WaveformParameters, WaveformParameters]:
    """
    Split waveform parameters into intrinsic and extrinsic components.

    Parameters
    ----------
    waveform_parameters
        The waveform parameters to split.
    intrinsic_luminosity_distance
        The intrinsic luminosity distance.
    intrinsic_geocent_time
        The intrinsic geocent time.

    Returns
    -------
    The intrinsic and extrinsic waveform parameters.
    """
    intrinsic_keys = [f.name for f in fields(IntrinsicPriors)]
    extrinsic_keys = [f.name for f in fields(ExtrinsicPriors)]
    waveform_dict = asdict(waveform_parameters)
    intrinsic_dict = {k: v for k, v in waveform_dict.items() if k in intrinsic_keys}
    extrinsic_dict = {k: v for k, v in waveform_dict.items() if k in extrinsic_keys}
    if intrinsic_luminosity_distance is not None:
        intrinsic_dict["luminosity_distance"] = intrinsic_luminosity_distance
    if intrinsic_geocent_time is not None:
        intrinsic_dict["geocent_time"] = intrinsic_geocent_time
    return WaveformParameters(**intrinsic_dict), WaveformParameters(**extrinsic_dict)


def build_prior_with_defaults(
    prior_settings: Union[IntrinsicPriors, Mapping[str, Union[str, float]]],
) -> BBHPriorDict:
    """
    Generate BBHPriorDict based on dictionary of prior settings,
    allowing for default values.

    Parameters
    ----------
    prior_settings
        A dictionary containing prior definitions for intrinsic parameters
        Allowed values for each parameter are:
            * 'default' to use a default prior
            * a string for a custom prior, e.g.,
               "Uniform(minimum=10.0, maximum=80.0, name=None, latex_label=None, unit=None, boundary=None)"

    Returns
    -------
    The BBHPriorDict with the specified prior settings.
    """
    prior_settings_: IntrinsicPriors
    if isinstance(prior_settings, dict):
        prior_settings = {
            k: v for k, v in prior_settings.items() if v is not None and v != "default"
        }
        prior_settings_ = IntrinsicPriors(**prior_settings)
    else:
        prior_settings_ = cast(IntrinsicPriors, prior_settings)
    return BBHPriorDict(asdict(prior_settings_))


default_extrinsic_dict = asdict(ExtrinsicPriors())
"""
Default extrinsic priors dictionary.

This dictionary contains the default prior settings for extrinsic parameters.
"""

default_intrinsic_dict = asdict(IntrinsicPriors())
"""
Default intrinsic priors dictionary.

This dictionary contains the default prior settings for intrinsic parameters.
"""

default_inference_parameters = [
    "chirp_mass",
    "mass_ratio",
    "phase",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "luminosity_distance",
    "geocent_time",
    "ra",
    "dec",
    "psi",
]
"""
Default inference parameters list.

This list contains the default parameters to be inferred during the analysis.
"""
