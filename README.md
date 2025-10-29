## About

Temporary repository. For proposing a refactor of the dingo waveform generator.

## Examples

Interactive examples are available in the `examples/` directory:

- **`visualize_waveforms.py`** - Interactive marimo notebook for visualizing gravitational waveforms with plotly
  - Supports both YAML configuration files and interactive Python parameter controls
  - Run with: `marimo edit examples/visualize_waveforms.py`
  - See `examples/README.md` for full documentation

- **`example_waveform_config.yaml`** - Example YAML configuration file for waveform generation

To use the examples, install with the optional examples dependencies:

```bash
pip install -e ".[examples]"
```