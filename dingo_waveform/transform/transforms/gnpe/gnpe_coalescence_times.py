"""
GNPECoalescenceTimes: GNPE transform for detector coalescence times.

This transform applies GNPE to detector coalescence times, enabling
exact or approximate time-translation equivariance in neural posterior estimation.

Reference: https://arxiv.org/abs/2111.13139
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from dingo_waveform.transform.transforms.gnpe.gnpe_base import GNPEBase, GNPEBaseConfig


@dataclass(frozen=True)
class GNPECoalescenceTimesConfig(GNPEBaseConfig):
    """
    Configuration for GNPECoalescenceTimes transform.

    Attributes
    ----------
    ifo_list : List[str]
        List of interferometer names
    kernel : str
        Bilby prior specification string for time perturbations
        (applied to all interferometers)
    exact_global_equivariance : bool
        Whether to enforce exact global time translation symmetry. Default True.
    inference : bool
        Whether in inference mode (True) or training mode (False). Default False.
    """

    ifo_list: List[str]
    kernel: str
    exact_global_equivariance: bool = True
    inference: bool = False

    def __post_init__(self) -> None:
        """Validate and initialize derived attributes."""
        if not isinstance(self.ifo_list, list):
            raise TypeError(f"ifo_list must be a list, got {type(self.ifo_list)}")
        if len(self.ifo_list) == 0:
            raise ValueError("ifo_list cannot be empty")

        if not isinstance(self.kernel, str):
            raise TypeError(f"kernel must be a string, got {type(self.kernel)}")

        if not isinstance(self.exact_global_equivariance, bool):
            raise TypeError(
                f"exact_global_equivariance must be bool, got {type(self.exact_global_equivariance)}"
            )
        if not isinstance(self.inference, bool):
            raise TypeError(f"inference must be bool, got {type(self.inference)}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GNPECoalescenceTimesConfig':
        """Create config from dictionary."""
        ifo_list = config_dict['ifo_list']

        # Convert InterferometerList to list of names
        if not isinstance(ifo_list, list):
            ifo_list = [ifo.name for ifo in ifo_list]

        # Build kernel_dict and operators from ifo_list
        ifo_time_labels = [f"{ifo}_time" for ifo in ifo_list]
        kernel_dict = {k: config_dict['kernel'] for k in ifo_time_labels}
        operators = {k: "+" for k in ifo_time_labels}

        # Create config with base class attributes
        config = cls(
            ifo_list=ifo_list,
            kernel=config_dict['kernel'],
            exact_global_equivariance=config_dict.get('exact_global_equivariance', True),
            inference=config_dict.get('inference', False),
            kernel_dict=kernel_dict,
            operators=operators
        )
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'ifo_list': self.ifo_list,
            'kernel': self.kernel,
            'exact_global_equivariance': self.exact_global_equivariance,
            'inference': self.inference
        }


class GNPECoalescenceTimes(GNPEBase):
    """
    GNPE transformation for detector coalescence times.

    This transform generates proxy detector times by adding perturbations
    from the GNPE kernel. The proxies are then subtracted from the true
    detector times, standardizing time shifts to the kernel range.

    Key features:
    - **Approximate equivariance**: Proxies condition the network
    - **Exact equivariance**: Optional exact global time translation symmetry
    - **Training/Inference modes**: Different behavior for each

    Examples
    --------
    >>> from bilby.gw.detector import InterferometerList
    >>> from dingo_waveform.transform.transforms.gnpe import (
    ...     GNPECoalescenceTimes,
    ...     GNPECoalescenceTimesConfig
    ... )
    >>>
    >>> # Training mode with exact equivariance
    >>> config = GNPECoalescenceTimesConfig(
    ...     ifo_list=['H1', 'L1', 'V1'],
    ...     kernel='Uniform(minimum=-0.1, maximum=0.1)',
    ...     exact_global_equivariance=True,
    ...     inference=False
    ... )
    >>> transform = GNPECoalescenceTimes.from_config(config)
    >>>
    >>> sample = {
    ...     'extrinsic_parameters': {
    ...         'H1_time': 1234567890.0,
    ...         'L1_time': 1234567890.01,
    ...         'V1_time': 1234567890.02,
    ...         'geocent_time': 1234567890.0
    ...     }
    ... }
    >>> result = transform(sample)
    >>> # Proxies added, detector times modified
    >>> 'H1_time_proxy' in result['extrinsic_parameters']
    True

    Notes
    -----
    **Training mode** (inference=False):
    - Assumes detector times are in extrinsic_parameters (not yet applied to data)
    - Computes: detector_time_new = detector_time - proxy
    - Data time-shifting happens once with these modified times

    **Inference mode** (inference=True):
    - Data already at detector times
    - Computes: detector_time_new = -proxy
    - Time shift is only by the proxy offset

    **Exact global equivariance** (exact_global_equivariance=True):
    1. First detector proxy (typically H1) defines global time shift
    2. This proxy is NOT explicitly conditioned on
    3. Subtract it from geocent_time (must be undone with PostCorrectGeocentTime)
    4. Subtract it from other proxies to get relative proxies
    5. Network conditions on relative proxies: {detector}_time_proxy_relative

    See Also
    --------
    PostCorrectGeocentTime : Undoes geocent_time correction at inference
    GetDetectorTimes : Computes detector times from sky position
    """

    def __init__(self, config: GNPECoalescenceTimesConfig):
        """Initialize GNPE coalescence times transform."""
        # First validate config completely
        object.__setattr__(config, 'kernel_dict', self._build_kernel_dict(config))
        object.__setattr__(config, 'operators', self._build_operators(config))

        super().__init__(config)

        self.ifo_time_labels = [f"{ifo}_time" for ifo in config.ifo_list]

        if config.exact_global_equivariance:
            # Remove first proxy from context (it's the "preferred" proxy)
            # Remaining proxies are conditioned as relative to the first
            del self.context_parameters[0]
            self.context_parameters = [p + "_relative" for p in self.context_parameters]

    @staticmethod
    def _build_kernel_dict(config: GNPECoalescenceTimesConfig) -> Dict[str, str]:
        """Build kernel_dict from ifo_list and kernel."""
        ifo_time_labels = [f"{ifo}_time" for ifo in config.ifo_list]
        return {k: config.kernel for k in ifo_time_labels}

    @staticmethod
    def _build_operators(config: GNPECoalescenceTimesConfig) -> Dict[str, str]:
        """Build operators dict (all additive for time)."""
        ifo_time_labels = [f"{ifo}_time" for ifo in config.ifo_list]
        return {k: "+" for k in ifo_time_labels}

    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Apply GNPE coalescence time transform.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample with 'extrinsic_parameters' dict

        Returns
        -------
        Dict[str, Any]
            Sample with GNPE proxies and modified detector times in
            'extrinsic_parameters'
        """
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        # Check if proxies already exist (e.g., from unconditional init network)
        # Otherwise, sample them
        if set(self.proxy_list).issubset(extrinsic_parameters.keys()):
            new_parameters = {p: extrinsic_parameters[p] for p in self.proxy_list}
        else:
            new_parameters = self.sample_proxies(extrinsic_parameters)

        # Modify detector times based on proxies
        if not self.config.inference:
            # Training: detector_time_new = detector_time - proxy
            for k in self.ifo_time_labels:
                new_parameters[k] = (
                    -new_parameters[k + "_proxy"] + extrinsic_parameters[k]
                )
        else:
            # Inference: detector_time_new = -proxy
            for k in self.ifo_time_labels:
                new_parameters[k] = -new_parameters[k + "_proxy"]

        # Apply exact global time shift equivariance
        if self.config.exact_global_equivariance:
            # First proxy defines global time shift
            dt = new_parameters[self.ifo_time_labels[0] + "_proxy"]

            if not self.config.inference:
                # Training: subtract from geocent_time
                if "geocent_time" not in extrinsic_parameters:
                    raise KeyError(
                        "geocent_time must be in extrinsic_parameters during training "
                        "when exact_global_equivariance=True"
                    )
                new_parameters["geocent_time"] = (
                    extrinsic_parameters["geocent_time"] - dt
                )
            else:
                # Inference: set geocent_time to -dt
                new_parameters["geocent_time"] = -dt

            # Compute relative proxies (relative to first detector)
            for k in self.ifo_time_labels[1:]:
                new_parameters[k + "_proxy_relative"] = (
                    new_parameters[k + "_proxy"] - dt
                )

        # Update extrinsic_parameters with all new values
        extrinsic_parameters.update(new_parameters)
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample
