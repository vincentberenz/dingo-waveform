"""Tests for parallel processing functions."""

import time

import numpy as np
import pandas as pd
import pytest

from dingo_waveform.svd import (
    ParallelConfig,
    SVDGenerationConfig,
    ValidationConfig,
    generate_svd_bases_from_dict,
    parallel_map,
)
from dingo_waveform.svd.parallel import ParallelSVDGenerator


# Module-level functions for pickling (required for multiprocessing)
def _square(x):
    """Square function for testing parallel_map."""
    return x**2


def _double(x):
    """Double function for testing parallel_map."""
    return x * 2


def _identity(x):
    """Identity function for testing parallel_map."""
    return x


def _slow_identity(x):
    """Slow identity with varying execution times for order testing."""
    time.sleep(0.001 * (x % 3))
    return x


class TestParallelMap:
    """Test parallel_map function."""

    def test_sequential_processing(self):
        """Test with num_workers=1 (sequential)."""
        items = list(range(10))
        config = ParallelConfig(num_workers=1)

        results = parallel_map(_square, items, config)

        assert results == [i**2 for i in range(10)]

    def test_parallel_processing(self):
        """Test with multiple workers."""
        items = list(range(20))
        config = ParallelConfig(num_workers=2)

        results = parallel_map(_square, items, config)

        # Results should be correct (order preserved)
        assert results == [i**2 for i in range(20)]

    def test_empty_items(self):
        """Test with empty list."""
        config = ParallelConfig(num_workers=2)

        results = parallel_map(_identity, [], config)

        assert results == []

    def test_single_item(self):
        """Test with single item."""
        config = ParallelConfig(num_workers=2)

        results = parallel_map(_double, [5], config)

        assert results == [10]

    def test_preserves_order(self):
        """Test that results preserve input order."""
        items = list(range(100))
        config = ParallelConfig(num_workers=4)

        results = parallel_map(_slow_identity, items, config)

        assert results == items

    def test_different_worker_counts(self):
        """Test with different numbers of workers."""
        items = list(range(10))

        expected = [i**2 for i in range(10)]

        for num_workers in [1, 2, 4]:
            config = ParallelConfig(num_workers=num_workers)
            results = parallel_map(_square, items, config)
            assert results == expected


class TestParallelSVDGenerator:
    """Test ParallelSVDGenerator class."""

    def test_add_batch(self):
        """Test adding batches."""
        config = ParallelConfig(num_workers=1, batch_size=10)
        generator = ParallelSVDGenerator(config)

        batch1 = np.random.randn(10, 50)
        batch2 = np.random.randn(10, 50)

        generator.add_batch(batch1)
        generator.add_batch(batch2)

        assert len(generator) == 2

    def test_concatenate(self):
        """Test concatenating batches."""
        config = ParallelConfig(num_workers=1, batch_size=10)
        generator = ParallelSVDGenerator(config)

        batch1 = np.random.randn(10, 50)
        batch2 = np.random.randn(15, 50)

        generator.add_batch(batch1)
        generator.add_batch(batch2)

        concatenated = generator.concatenate()

        assert concatenated.shape == (25, 50)
        assert np.allclose(concatenated[:10], batch1)
        assert np.allclose(concatenated[10:], batch2)

    def test_clear(self):
        """Test clearing the buffer."""
        config = ParallelConfig(num_workers=1)
        generator = ParallelSVDGenerator(config)

        generator.add_batch(np.random.randn(10, 50))
        generator.add_batch(np.random.randn(10, 50))

        assert len(generator) == 2

        generator.clear()

        assert len(generator) == 0

    def test_concatenate_empty_raises_error(self):
        """Test that concatenating empty buffer raises error."""
        config = ParallelConfig(num_workers=1)
        generator = ParallelSVDGenerator(config)

        with pytest.raises(ValueError, match="No data in buffer"):
            generator.concatenate()

    def test_len(self):
        """Test __len__ method."""
        config = ParallelConfig(num_workers=1)
        generator = ParallelSVDGenerator(config)

        assert len(generator) == 0

        generator.add_batch(np.random.randn(10, 50))
        assert len(generator) == 1

        generator.add_batch(np.random.randn(10, 50))
        assert len(generator) == 2

    def test_reuse_after_clear(self):
        """Test that generator can be reused after clear."""
        config = ParallelConfig(num_workers=1)
        generator = ParallelSVDGenerator(config)

        # First use
        generator.add_batch(np.random.randn(10, 50))
        data1 = generator.concatenate()
        generator.clear()

        # Second use
        generator.add_batch(np.random.randn(15, 50))
        data2 = generator.concatenate()

        assert data1.shape == (10, 50)
        assert data2.shape == (15, 50)


class TestGenerateSVDBasesFromDict:
    """Test generate_svd_bases_from_dict function."""

    def test_basic_multi_stream(self):
        """Test generating SVD bases for multiple streams."""
        np.random.seed(42)
        data_dict = {
            "H1": np.random.randn(100, 50),
            "L1": np.random.randn(100, 50),
        }

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config)

        assert "H1" in bases
        assert "L1" in bases
        assert bases["H1"].n_components == 10
        assert bases["L1"].n_components == 10

    def test_complex_data(self):
        """Test with complex-valued data."""
        np.random.seed(42)
        data_dict = {
            "H1": np.random.randn(100, 50) + 1j * np.random.randn(100, 50),
            "L1": np.random.randn(100, 50) + 1j * np.random.randn(100, 50),
        }

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config)

        assert np.iscomplexobj(bases["H1"].V)
        assert np.iscomplexobj(bases["L1"].V)

    def test_single_stream(self):
        """Test with single stream."""
        np.random.seed(42)
        data_dict = {"H1": np.random.randn(100, 50)}

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config)

        assert len(bases) == 1
        assert "H1" in bases

    def test_train_validation_split(self):
        """Test with training and validation split."""
        np.random.seed(42)
        data_dict = {
            "H1": np.random.randn(100, 50),
            "L1": np.random.randn(100, 50),
        }

        svd_config = SVDGenerationConfig(n_components=10, method="scipy")
        val_config = ValidationConfig(increment=5)

        bases = generate_svd_bases_from_dict(
            data_dict,
            svd_config,
            num_training_samples=80,
            num_validation_samples=20,
            validation_config=val_config,
        )

        # Check that validation was performed
        assert bases["H1"].mismatches is not None
        assert bases["L1"].mismatches is not None
        assert len(bases["H1"].mismatches) == 20

    def test_with_labels(self):
        """Test with parameter labels."""
        np.random.seed(42)
        data_dict = {
            "H1": np.random.randn(100, 50),
            "L1": np.random.randn(100, 50),
        }

        labels = pd.DataFrame(
            {
                "mass_1": np.random.uniform(10, 50, 100),
                "mass_2": np.random.uniform(10, 50, 100),
            }
        )

        svd_config = SVDGenerationConfig(n_components=10, method="scipy")
        val_config = ValidationConfig(increment=5)

        bases = generate_svd_bases_from_dict(
            data_dict,
            svd_config,
            num_training_samples=80,
            num_validation_samples=20,
            validation_config=val_config,
            labels=labels,
        )

        # Labels should be in validation results
        assert bases["H1"].mismatches is not None
        assert "mass_1" in bases["H1"].mismatches.columns
        assert "mass_2" in bases["H1"].mismatches.columns

    def test_no_validation(self):
        """Test without validation."""
        np.random.seed(42)
        data_dict = {"H1": np.random.randn(100, 50)}

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config)

        assert bases["H1"].mismatches is None

    def test_use_all_samples(self):
        """Test using all samples for training (no split)."""
        np.random.seed(42)
        data_dict = {"H1": np.random.randn(100, 50)}

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config)

        # Should have used all 100 samples for training
        assert bases["H1"].n_components == 10

    def test_verbose_output(self, capsys):
        """Test verbose output."""
        np.random.seed(42)
        data_dict = {"H1": np.random.randn(100, 50)}

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config, verbose=True)

        captured = capsys.readouterr()
        assert "Generating SVD bases" in captured.out
        assert "H1" in captured.out


class TestGenerateSVDBasesFromDictValidation:
    """Test input validation for generate_svd_bases_from_dict."""

    def test_mismatched_sample_counts(self):
        """Test that mismatched sample counts raise error."""
        data_dict = {
            "H1": np.random.randn(100, 50),
            "L1": np.random.randn(80, 50),  # Different number of samples
        }

        config = SVDGenerationConfig(n_components=10, method="scipy")

        with pytest.raises(ValueError, match="same number of samples"):
            generate_svd_bases_from_dict(data_dict, config)

    def test_too_many_samples_requested(self):
        """Test that requesting too many samples raises error."""
        data_dict = {"H1": np.random.randn(100, 50)}

        config = SVDGenerationConfig(n_components=10, method="scipy")

        with pytest.raises(ValueError, match="only.*samples available"):
            generate_svd_bases_from_dict(
                data_dict,
                config,
                num_training_samples=80,
                num_validation_samples=30,  # Total = 110 > 100
            )

    def test_validation_config_without_validation_samples(self):
        """Test that validation_config without validation_samples raises error."""
        data_dict = {"H1": np.random.randn(100, 50)}

        svd_config = SVDGenerationConfig(n_components=10, method="scipy")
        val_config = ValidationConfig(increment=5)

        with pytest.raises(ValueError, match="validation_config provided"):
            generate_svd_bases_from_dict(
                data_dict,
                svd_config,
                num_training_samples=100,
                num_validation_samples=None,  # No validation samples
                validation_config=val_config,
            )

    def test_empty_data_dict(self):
        """Test with empty data dictionary."""
        data_dict: dict[str, np.ndarray] = {}

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config)

        assert len(bases) == 0


class TestEdgeCases:
    """Test edge cases in parallel operations."""

    def test_many_streams(self):
        """Test with many streams."""
        np.random.seed(42)
        data_dict = {f"ifo_{i}": np.random.randn(50, 30) for i in range(10)}

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config)

        assert len(bases) == 10
        for i in range(10):
            assert f"ifo_{i}" in bases

    def test_different_feature_sizes_raises_error(self):
        """Test that different feature sizes raise error during compression."""
        # This is more of an integration test
        np.random.seed(42)
        data_dict = {
            "H1": np.random.randn(100, 50),
            "L1": np.random.randn(100, 50),
        }

        config = SVDGenerationConfig(n_components=10, method="scipy")
        bases = generate_svd_bases_from_dict(data_dict, config)

        # Try to compress data with wrong feature size
        wrong_size_data = np.random.randn(10, 40)  # 40 features instead of 50

        with pytest.raises((ValueError, Exception)):
            bases["H1"].compress(wrong_size_data)

    def test_single_sample_per_stream(self):
        """Test with minimal data."""
        data_dict = {
            "H1": np.random.randn(10, 50),
            "L1": np.random.randn(10, 50),
        }

        config = SVDGenerationConfig(n_components=5, method="scipy")
        bases = generate_svd_bases_from_dict(
            data_dict, config, num_training_samples=8, num_validation_samples=2
        )

        assert bases["H1"].n_components == 5
        # No validation performed without validation_config, so mismatches is None
        assert bases["H1"].mismatches is None


class TestIntegration:
    """Integration tests for parallel SVD generation."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: generate, validate, save, load."""
        import os
        import tempfile

        np.random.seed(42)
        data_dict = {
            "H1": np.random.randn(100, 50) + 1j * np.random.randn(100, 50),
            "L1": np.random.randn(100, 50) + 1j * np.random.randn(100, 50),
        }

        labels = pd.DataFrame({"param": np.arange(100)})

        # Generate with validation
        svd_config = SVDGenerationConfig(n_components=10, method="scipy")
        val_config = ValidationConfig(increment=5)

        bases = generate_svd_bases_from_dict(
            data_dict,
            svd_config,
            num_training_samples=80,
            num_validation_samples=20,
            validation_config=val_config,
            labels=labels,
            verbose=False,
        )

        # Save to files
        with tempfile.TemporaryDirectory() as tmpdir:
            h1_file = os.path.join(tmpdir, "svd_h1.h5")
            l1_file = os.path.join(tmpdir, "svd_l1.h5")

            bases["H1"].save(h1_file)
            bases["L1"].save(l1_file)

            # Load back
            from dingo_waveform.svd import SVDBasis

            loaded_h1 = SVDBasis.from_file(h1_file)
            loaded_l1 = SVDBasis.from_file(l1_file)

            # Verify
            assert loaded_h1.n_components == 10
            assert loaded_l1.n_components == 10
            assert loaded_h1.mismatches is not None
            assert loaded_l1.mismatches is not None

            # Test compression with loaded bases
            test_data = np.random.randn(5, 50) + 1j * np.random.randn(5, 50)
            compressed_h1 = loaded_h1.compress(test_data)
            compressed_l1 = loaded_l1.compress(test_data)

            assert compressed_h1.shape == (5, 10)
            assert compressed_l1.shape == (5, 10)
