"""Waveform dataset class for storing and managing generated waveforms."""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union

import h5py
import numpy as np
import pandas as pd

from .polarizations import Polarizations
from .dataset_settings import DatasetSettings

_logger = logging.getLogger(__name__)


class WaveformDataset:
    """
    Container for waveform parameters and polarizations.

    Attributes
    ----------
    parameters : pd.DataFrame
        DataFrame containing waveform parameters (one row per waveform).
    polarizations : Polarizations
        Dataclass containing 'h_plus' and 'h_cross' arrays of shape (num_samples, frequency_bins).
    settings : Optional[DatasetSettings]
        Settings used to generate the dataset.
    """

    def __init__(
        self,
        parameters: pd.DataFrame,
        polarizations: Union[Polarizations, Dict[str, np.ndarray]],
        settings: Optional[Union[DatasetSettings, Dict]] = None,
    ):
        """
        Initialize a WaveformDataset.

        Parameters
        ----------
        parameters
            DataFrame of waveform parameters.
        polarizations
            Polarizations dataclass or dictionary with 'h_plus' and 'h_cross' arrays.
            Dictionary support maintained for backward compatibility.
        settings
            Optional DatasetSettings or dictionary of generation settings.
            Dictionary support maintained for backward compatibility.
        """
        self.parameters = parameters

        # Convert dict to Polarizations if needed (backward compatibility)
        if isinstance(polarizations, dict):
            self.polarizations = Polarizations(
                h_plus=polarizations["h_plus"],
                h_cross=polarizations["h_cross"],
            )
        else:
            self.polarizations = polarizations

        # Convert dict to DatasetSettings if needed (backward compatibility)
        if settings is None:
            self.settings = None
        elif isinstance(settings, dict):
            # Store as dict for now, but log a deprecation-style warning
            self.settings = settings
            _logger.debug("Received settings as dict; consider using DatasetSettings dataclass")
        else:
            self.settings = settings

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
        num_polarizations = len(self.polarizations)

        if num_polarizations != num_params:
            raise ValueError(
                f"Mismatch: {num_params} parameter rows but {num_polarizations} waveforms in polarizations"
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
            f"waveform_length={self.polarizations.num_frequency_bins})"
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
            f.create_dataset("h_plus", data=self.polarizations.h_plus, compression="gzip")
            f.create_dataset("h_cross", data=self.polarizations.h_cross, compression="gzip")

            # Save parameters as a group with individual datasets
            params_group = f.create_group("parameters")
            for col in self.parameters.columns:
                params_group.create_dataset(col, data=self.parameters[col].values)

            # Save settings as attributes (flattened)
            settings_group = f.create_group("settings")
            if self.settings is not None:
                # Convert DatasetSettings to dict if needed
                if isinstance(self.settings, DatasetSettings):
                    settings_dict = self.settings.to_dict()
                else:
                    settings_dict = self.settings
                self._save_dict_to_group(settings_group, settings_dict)

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
            polarizations = Polarizations(
                h_plus=f["h_plus"][:],
                h_cross=f["h_cross"][:],
            )

            # Load parameters
            params_group = f["parameters"]
            param_dict = {key: params_group[key][:] for key in params_group.keys()}
            parameters = pd.DataFrame(param_dict)

            # Load settings (keep as dict for now, could be converted to DatasetSettings later)
            settings = cls._load_dict_from_group(f["settings"]) if "settings" in f else None

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
