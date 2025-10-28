"""Waveform dataset class for storing and managing generated waveforms."""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import h5py
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


class WaveformDataset:
    """
    Container for waveform parameters and polarizations.

    Attributes
    ----------
    parameters : pd.DataFrame
        DataFrame containing waveform parameters (one row per waveform).
    polarizations : Dict[str, np.ndarray]
        Dictionary with 'h_plus' and 'h_cross' arrays of shape (num_samples, frequency_bins).
    settings : Dict
        Dictionary containing the settings used to generate the dataset.
    """

    def __init__(
        self,
        parameters: pd.DataFrame,
        polarizations: Dict[str, np.ndarray],
        settings: Optional[Dict] = None,
    ):
        """
        Initialize a WaveformDataset.

        Parameters
        ----------
        parameters
            DataFrame of waveform parameters.
        polarizations
            Dictionary with 'h_plus' and 'h_cross' polarization arrays.
        settings
            Optional dictionary of generation settings.
        """
        self.parameters = parameters
        self.polarizations = polarizations
        self.settings = settings or {}

        # Validate consistency
        self._validate()

    def _validate(self) -> None:
        """
        Validate internal consistency of the dataset.

        Raises
        ------
        ValueError
            If polarizations and parameters have inconsistent sizes.
        """
        num_params = len(self.parameters)
        num_h_plus = len(self.polarizations.get("h_plus", []))
        num_h_cross = len(self.polarizations.get("h_cross", []))

        if num_h_plus != num_params:
            raise ValueError(
                f"Mismatch: {num_params} parameter rows but {num_h_plus} h_plus waveforms"
            )
        if num_h_cross != num_params:
            raise ValueError(
                f"Mismatch: {num_params} parameter rows but {num_h_cross} h_cross waveforms"
            )

        _logger.debug(f"Dataset validated: {num_params} waveforms")

    def __len__(self) -> int:
        """Return the number of waveforms in the dataset."""
        return len(self.parameters)

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"WaveformDataset(num_waveforms={len(self)}, "
            f"num_parameters={len(self.parameters.columns)}, "
            f"waveform_length={self.polarizations['h_plus'].shape[-1]})"
        )

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save the dataset to an HDF5 file.

        Parameters
        ----------
        file_path
            Path where the HDF5 file will be saved.
        """
        file_path = Path(file_path)
        _logger.info(f"Saving dataset to {file_path}")

        with h5py.File(file_path, "w") as f:
            # Save polarizations
            for key, data in self.polarizations.items():
                f.create_dataset(key, data=data, compression="gzip")

            # Save parameters as a group with individual datasets
            params_group = f.create_group("parameters")
            for col in self.parameters.columns:
                params_group.create_dataset(col, data=self.parameters[col].values)

            # Save settings as attributes (flattened)
            settings_group = f.create_group("settings")
            self._save_dict_to_group(settings_group, self.settings)

        _logger.info(f"Dataset saved successfully ({len(self)} waveforms)")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "WaveformDataset":
        """
        Load a dataset from an HDF5 file.

        Parameters
        ----------
        file_path
            Path to the HDF5 file.

        Returns
        -------
        WaveformDataset
            Loaded dataset.
        """
        file_path = Path(file_path)
        _logger.info(f"Loading dataset from {file_path}")

        with h5py.File(file_path, "r") as f:
            # Load polarizations
            polarizations = {
                "h_plus": f["h_plus"][:],
                "h_cross": f["h_cross"][:],
            }

            # Load parameters
            params_group = f["parameters"]
            param_dict = {key: params_group[key][:] for key in params_group.keys()}
            parameters = pd.DataFrame(param_dict)

            # Load settings
            settings = cls._load_dict_from_group(f["settings"])

        _logger.info(f"Dataset loaded successfully ({len(parameters)} waveforms)")
        return cls(parameters, polarizations, settings)

    @staticmethod
    def _save_dict_to_group(group: h5py.Group, data: Dict) -> None:
        """
        Recursively save a dictionary to an HDF5 group.

        Parameters
        ----------
        group
            HDF5 group to save to.
        data
            Dictionary to save.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                WaveformDataset._save_dict_to_group(subgroup, value)
            elif value is None:
                # Store None as empty dataset with special attribute
                ds = group.create_dataset(key, data=h5py.Empty("f"))
                ds.attrs["is_none"] = True
            elif isinstance(value, (list, tuple)):
                # Convert to numpy array for storage
                group.create_dataset(key, data=np.array(value))
            elif isinstance(value, (str, int, float, bool, np.ndarray)):
                group.create_dataset(key, data=value)
            else:
                # Try to convert to string as fallback
                _logger.warning(
                    f"Converting unsupported type {type(value)} to string for key '{key}'"
                )
                group.create_dataset(key, data=str(value))

    @staticmethod
    def _load_dict_from_group(group: h5py.Group) -> Dict:
        """
        Recursively load a dictionary from an HDF5 group.

        Parameters
        ----------
        group
            HDF5 group to load from.

        Returns
        -------
        Dict
            Loaded dictionary.
        """
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = WaveformDataset._load_dict_from_group(item)
            elif isinstance(item, h5py.Dataset):
                # Check if this represents None
                if item.attrs.get("is_none", False):
                    result[key] = None
                else:
                    data = item[()]
                    # Convert bytes to string if needed
                    if isinstance(data, bytes):
                        result[key] = data.decode("utf-8")
                    else:
                        result[key] = data
        return result
