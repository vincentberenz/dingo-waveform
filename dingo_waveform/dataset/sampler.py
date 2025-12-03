"""Sampler class for applying transforms to waveform datasets."""

from typing import Iterator

from ..transform.transform import Transform


class Sampler:
    """
    Sampler for applying transforms to waveform datasets.

    Wraps a Transform instance and provides convenient methods for
    sampling transformed waveforms from a WaveformDataset.

    Attributes
    ----------
    transform : Transform
        Transform instance configured for SVD generation or inference
    dataset : WaveformDataset
        Reference to the source waveform dataset
    """

    def __init__(
        self,
        transform: Transform,
        dataset: "WaveformDataset",  # Forward reference to avoid circular import
    ):
        """
        Initialize Sampler with Transform and WaveformDataset.

        Parameters
        ----------
        transform : Transform
            Configured Transform instance
        dataset : WaveformDataset
            Source waveform dataset
        """
        self.transform = transform
        self.dataset = dataset

    def get_svd_iterator(self, **kwargs) -> Iterator:
        """
        Get iterator for SVD generation.

        Delegates to transform.get_svd_iterator().

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to get_svd_iterator()
            (e.g., batch_size, shuffle, num_workers)

        Returns
        -------
        Iterator
            Data iterator configured for SVD generation
        """
        return self.transform.get_svd_iterator(**kwargs)

    def get_training_iterator(self, **kwargs) -> Iterator:
        """
        Get iterator for training.

        Delegates to transform.get_training_iterator().

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to get_training_iterator()
            (e.g., batch_size, shuffle, num_workers)

        Returns
        -------
        Iterator
            Data iterator configured for training
        """
        return self.transform.get_training_iterator(**kwargs)

    def get_inference_transform_pre(self):
        """
        Get pre-processing transform for inference.

        Delegates to transform.get_inference_transform_pre().

        Returns
        -------
        Transform
            Pre-processing transform for inference
        """
        return self.transform.get_inference_transform_pre()

    def get_inference_transform_post(self):
        """
        Get post-processing transform for inference.

        Delegates to transform.get_inference_transform_post().

        Returns
        -------
        Transform
            Post-processing transform for inference
        """
        return self.transform.get_inference_transform_post()

    def __repr__(self) -> str:
        """Return string representation of Sampler."""
        return f"Sampler(dataset={self.dataset}, transform={self.transform})"
