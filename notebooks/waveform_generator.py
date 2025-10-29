import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import logging
    import numpy as np
    import plotly.graph_objects as go
    from dataclasses import asdict
    from dingo_waveform.dataset.dataset_settings import DatasetSettings
    from dingo_waveform.domains import build_domain
    from dingo_waveform.waveform_generator import build_waveform_generator
    return DatasetSettings, asdict, build_domain, build_waveform_generator, go, json, logging, mo, np


@app.cell
def _():
    # Default configuration for waveform generation
    default_config = {
        "domain": {
            "type": "FrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,
        },
        "waveform_generator": {
            "approximant": "IMRPhenomD",
            "f_ref": 20.0,
        },
        "intrinsic_prior": {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": "bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary='periodic')",
            "chi_1": "bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=Uniform(minimum=0, maximum=0.99))",
            "chi_2": "bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=Uniform(minimum=0, maximum=0.99))",
            "geocent_time": 0.0,
        },
        "num_samples": 1,
    }
    return (default_config,)


@app.cell
def _(mo):
    # Display configuration as editable JSON
    mo.md("## Waveform Generation Configuration")
    return


@app.cell
def _(default_config, json, mo):
    config_editor = mo.ui.text_area(
        value=json.dumps(default_config, indent=2),
        label="Edit configuration (JSON format):",
        rows=30,
        full_width=True,
    )
    config_editor
    return (config_editor,)


@app.cell
def _(config_editor, json, mo):
    # Parse the edited configuration
    try:
        edited_config = json.loads(config_editor.value)
        mo.md(f"✅ Configuration is valid JSON")
    except json.JSONDecodeError as e:
        edited_config = None
        mo.md(f"❌ Invalid JSON: {e}")
    return (edited_config,)


@app.cell
def _(edited_config, mo):
    # Display the parsed configuration
    if edited_config is not None:
        mo.md("### Current Configuration:")
        mo.tree(edited_config)
    return


@app.cell
def _(edited_config, mo):
    # Button to generate waveform
    generate_button = mo.ui.run_button(label="Generate Waveform")
    generate_button
    return (generate_button,)


@app.cell
def _(DatasetSettings, build_domain, build_waveform_generator, edited_config, generate_button, logging, mo):
    import sys

    # Configure root logger to capture all logs including from dingo_waveform package
    _root_logger = logging.getLogger()
    _root_logger.setLevel(logging.INFO)

    # Add handler to root logger if it doesn't have one already
    if not _root_logger.handlers:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setLevel(logging.INFO)
        _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(_formatter)
        _root_logger.addHandler(_handler)

    # Also configure dingo_waveform logger specifically
    _dingo_logger = logging.getLogger('dingo_waveform')
    _dingo_logger.setLevel(logging.INFO)

    _logger = logging.getLogger(__name__)

    # Generate waveform when button is clicked
    if generate_button.value and edited_config is not None:
        try:
            _logger.info("Starting waveform generation...")

            # Build settings from config
            _logger.info("Building settings from config")
            settings = DatasetSettings.from_dict(edited_config)

            # Build domain and waveform generator
            _logger.info("Building domain")
            domain = build_domain(edited_config["domain"])
            wfg_params = {
                "approximant": edited_config["waveform_generator"]["approximant"],
                "f_ref": edited_config["waveform_generator"]["f_ref"],
                "domain": domain,
            }
            _logger.info(f"Building waveform generator with approximant {wfg_params['approximant']}")
            wfg = build_waveform_generator(wfg_params, domain)

            # Sample parameters from prior
            _logger.info("Sampling parameters from prior")
            wf_params = settings.intrinsic_prior.sample()

            # Generate waveform
            _logger.info("Generating waveform polarizations")
            polarization = wfg.generate_hplus_hcross(wf_params)

            waveform_data = {
                "h_plus": polarization.h_plus,
                "h_cross": polarization.h_cross,
                "domain": domain,
                "parameters": wf_params,
            }

            _logger.info(f"Waveform generated successfully with shape {polarization.h_plus.shape}")
            status_msg = mo.md("✅ Waveform generated successfully!")

        except Exception as e:
            _logger.error(f"Error generating waveform: {e}", exc_info=True)
            waveform_data = None
            status_msg = mo.md(f"❌ Error generating waveform: {e}")
    else:
        waveform_data = None
        status_msg = mo.md("")

    status_msg
    return (waveform_data,)


@app.cell
def _(asdict, mo, np, waveform_data):
    import sys
    print(f"DEBUG: waveform_data is None = {waveform_data is None}", file=sys.stderr)

    if waveform_data is not None:
        print("DEBUG: Displaying waveform info", file=sys.stderr)
        # Display waveform properties
        info = f"""
## Waveform Information

### Waveform Properties
- **Shape**: {waveform_data['h_plus'].shape}
- **Domain type**: {type(waveform_data['domain']).__name__}
- **h_plus amplitude range**: [{np.abs(waveform_data['h_plus']).min():.2e}, {np.abs(waveform_data['h_plus']).max():.2e}]
- **h_cross amplitude range**: [{np.abs(waveform_data['h_cross']).min():.2e}, {np.abs(waveform_data['h_cross']).max():.2e}]

### Sampled Parameters
"""

        mo.vstack([
            mo.md(info),
            mo.tree(asdict(waveform_data['parameters']))
        ])
    return


@app.cell
def _(go, mo, np, waveform_data):
    import sys
    print(f"DEBUG MAG: waveform_data is None = {waveform_data is None}", file=sys.stderr)

    if waveform_data is not None:
        print("DEBUG MAG: Creating magnitude plot", file=sys.stderr)
        # Get frequency array from domain
        _domain = waveform_data['domain']
        _frequencies = _domain.sample_frequencies()

        # Calculate magnitudes
        _h_plus_mag = np.abs(waveform_data['h_plus'])
        _h_cross_mag = np.abs(waveform_data['h_cross'])

        print(f"DEBUG MAG: frequencies shape = {_frequencies.shape}, h_plus_mag shape = {_h_plus_mag.shape}", file=sys.stderr)

        # Create magnitude plot
        _fig_mag = go.Figure()
        _fig_mag.add_trace(go.Scatter(
            x=_frequencies,
            y=_h_plus_mag,
            name='|h_plus(f)|',
            mode='lines',
            line=dict(color='blue')
        ))
        _fig_mag.add_trace(go.Scatter(
            x=_frequencies,
            y=_h_cross_mag,
            name='|h_cross(f)|',
            mode='lines',
            line=dict(color='red')
        ))
        _fig_mag.update_layout(
            title="Waveform Magnitude in Frequency Domain",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
        )

        print("DEBUG MAG: About to display plot", file=sys.stderr)
        mo.vstack([
            mo.md("## Frequency-Domain Visualizations"),
            mo.ui.plotly(_fig_mag)
        ])
    return


@app.cell
def _(go, mo, np, waveform_data):
    if waveform_data is not None:
        # Get frequency array from domain
        _domain_phase = waveform_data['domain']
        _frequencies_phase = _domain_phase.sample_frequencies()

        # Calculate phases with unwrapping
        _h_plus_phase = np.unwrap(np.angle(waveform_data['h_plus']))
        _h_cross_phase = np.unwrap(np.angle(waveform_data['h_cross']))

        # Create phase plot
        _fig_phase = go.Figure()
        _fig_phase.add_trace(go.Scatter(
            x=_frequencies_phase,
            y=_h_plus_phase,
            name='arg(h_plus)',
            mode='lines',
            line=dict(color='blue')
        ))
        _fig_phase.add_trace(go.Scatter(
            x=_frequencies_phase,
            y=_h_cross_phase,
            name='arg(h_cross)',
            mode='lines',
            line=dict(color='red')
        ))
        _fig_phase.update_layout(
            title="Waveform Phase in Frequency Domain (Unwrapped)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Phase (radians)",
            xaxis_type="log",
            height=500,
        )

        _output_phase = mo.ui.plotly(_fig_phase)
        _output_phase
    return


if __name__ == "__main__":
    app.run()
