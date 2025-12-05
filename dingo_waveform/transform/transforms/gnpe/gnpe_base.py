"""
GNPEBase: Abstract base class for Group-Equivariant Neural Posterior Estimation.

This implements GNPE for approximate equivariances. Subclasses implement
specific group transformations (e.g., time translations, rotations).

Reference: https://arxiv.org/abs/2111.13139
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import torch
from dingo_waveform.transform.base import WaveformTransform, WaveformTransformConfig
from dingo_waveform.transform.types import GroupOperator, ParameterValue, TransformSample


@dataclass(frozen=True)
class GNPEBaseConfig(WaveformTransformConfig):
    """
    Configuration for GNPEBase transform.

    Attributes
    ----------
    kernel_dict : Dict[str, Any]
        Dictionary specifying kernel priors for perturbations.
        Maps parameter names to bilby prior specifications.
    operators : Dict[str, GroupOperator]
        Dictionary specifying group operators ('+' or 'x') for each parameter
    """

    kernel_dict: Dict[str, Any]
    operators: Dict[str, GroupOperator]

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.kernel_dict, dict):
            raise TypeError(f"kernel_dict must be a dict, got {type(self.kernel_dict)}")
        if not isinstance(self.operators, dict):
            raise TypeError(f"operators must be a dict, got {type(self.operators)}")

        if len(self.kernel_dict) == 0:
            raise ValueError("kernel_dict cannot be empty")

        # Check operators are valid
        for k, op in self.operators.items():
            if op not in ['+', 'x']:
                raise ValueError(
                    f"Operator for {k} must be '+' or 'x', got '{op}'"
                )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GNPEBaseConfig':
        """Create config from dictionary."""
        return cls(
            kernel_dict=config_dict['kernel_dict'],
            operators=config_dict['operators']
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'kernel_dict': self.kernel_dict,
            'operators': self.operators
        }


class GNPEBase(WaveformTransform[GNPEBaseConfig], ABC):
    """
    Abstract base class for Group-Equivariant Neural Posterior Estimation.

    GNPE leverages group symmetries to improve neural posterior estimation.
    This base class implements core GNPE functionality for approximate
    equivariances. Subclasses implement specific group transformations.

    Key concepts:
    - **Proxy parameters**: Perturbed versions of true parameters (g_hat ~ p(g_hat | g))
    - **Kernel**: Prior distribution for perturbations
    - **Group operators**: '+' (additive) or 'x' (multiplicative)

    Examples
    --------
    Subclasses must implement __call__() to apply specific transformations.
    See GNPECoalescenceTimes for a concrete example.

    Notes
    -----
    Reference: https://arxiv.org/abs/2111.13139

    The GNPE framework enables:
    1. Data augmentation via group transformations
    2. Improved network conditioning via proxy parameters
    3. Exact or approximate equivariance to group symmetries
    """

    def __init__(self, config: GNPEBaseConfig):
        """Initialize GNPE base transform."""
        super().__init__(config)

        from bilby.core.prior import PriorDict

        self.kernel = PriorDict(config.kernel_dict)
        self.proxy_list = [k + "_proxy" for k in config.kernel_dict.keys()]
        self.context_parameters = self.proxy_list.copy()
        self.input_parameter_names = list(self.kernel.keys())

    @abstractmethod
    def __call__(self, input_sample: TransformSample) -> TransformSample:  # type: ignore[override]
        """
        Apply GNPE transform.

        Must be implemented by subclasses.

        Parameters
        ----------
        input_sample : Dict[str, Any]
            Input sample

        Returns
        -------
        Dict[str, Any]
            Transformed sample with GNPE proxies
        """
        pass

    def sample_proxies(self, input_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate proxy parameters by perturbing input parameters.

        Samples proxy parameters g_hat ~ p(g_hat | g) using the kernel:
            g_hat = g ⊕ ε, where ε ~ kernel and ⊕ is the group operator

        Parameters
        ----------
        input_parameters : Dict[str, Any]
            Initial parameter values. Values can be floats (training) or
            torch Tensors (inference).

        Returns
        -------
        Dict[str, Any]
            Dictionary of proxy parameters with keys '{param}_proxy'

        Examples
        --------
        >>> # Training mode (scalar values)
        >>> params = {'geocent_time': 0.0, 'H1_time': 0.0}
        >>> proxies = transform.sample_proxies(params)
        >>> proxies.keys()
        dict_keys(['geocent_time_proxy', 'H1_time_proxy'])

        >>> # Inference mode (batched tensors)
        >>> params = {'geocent_time': torch.zeros(100)}
        >>> proxies = transform.sample_proxies(params)
        >>> proxies['geocent_time_proxy'].shape
        torch.Size([100])
        """
        proxies = {}
        for k in self.kernel:
            if k not in input_parameters:
                raise KeyError(
                    f"Input parameters missing key '{k}' required for GNPE."
                )
            g = input_parameters[k]
            g_hat = self.perturb(g, k)
            proxies[k + "_proxy"] = g_hat
        return proxies

    def perturb(self, g: ParameterValue, k: str) -> ParameterValue:
        """
        Generate proxy variable by perturbing parameter g.

        Parameters
        ----------
        g : ParameterValue
            Initial parameter value (float, np.ndarray, or torch.Tensor)
        k : str
            Parameter name (determines operator)

        Returns
        -------
        ParameterValue
            Proxy variable g_hat = g ⊕ ε (same type as g)
        """
        # Sample epsilon from kernel with correct type
        if type(g) == torch.Tensor:
            epsilon = self.kernel[k].sample(len(g))
            epsilon = torch.tensor(epsilon, dtype=g.dtype, device=g.device)
        elif type(g) in [np.float64, float]:
            epsilon = self.kernel[k].sample()
        elif type(g) == np.ndarray:
            epsilon = self.kernel[k].sample(len(g)).astype(g.dtype)
        else:
            raise NotImplementedError(f"Unsupported data type {type(g)}.")

        return self.multiply(g, epsilon, k)

    def multiply(self, a: ParameterValue, b: ParameterValue, k: str) -> ParameterValue:
        """
        Apply group multiplication operator.

        Parameters
        ----------
        a : ParameterValue
            First operand
        b : ParameterValue
            Second operand
        k : str
            Parameter name (determines operator)

        Returns
        -------
        ParameterValue
            Result of a ⊕ b

        Notes
        -----
        Supported operators:
        - '+': Additive group (a + b)
        - 'x': Multiplicative group (a * b)
        """
        op = self.config.operators[k]
        if op == "+":
            return a + b
        elif op == "x":
            return a * b
        else:
            raise NotImplementedError(f"Unsupported operator: {op}")

    def inverse(self, a: ParameterValue, k: str) -> ParameterValue:
        """
        Compute group inverse.

        Parameters
        ----------
        a : ParameterValue
            Element to invert
        k : str
            Parameter name (determines operator)

        Returns
        -------
        ParameterValue
            Group inverse of a

        Notes
        -----
        Inverses:
        - '+': -a
        - 'x': 1/a
        """
        op = self.config.operators[k]
        if op == "+":
            return -a
        elif op == "x":
            return 1 / a
        else:
            raise NotImplementedError(f"Unsupported operator: {op}")
