import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")

"""

domain:
  type: MultibandedFrequencyDomain
  nodes: [20.0, 38.0, 50.0, 66.0, 82.0, 1810.0]
  delta_f_initial: 0.125
  base_delta_f: 0.125

waveform_generator:
  approximant: IMRPhenomXPHM
  f_ref: 20.0
  spin_conversion_phase: 0.0

intrinsic_prior:
  mass_1: bilby.core.prior.Constraint(minimum=10.0, maximum=120.0)
  mass_2: bilby.core.prior.Constraint(minimum=10.0, maximum=120.0)
  chirp_mass: bilby.gw.prior.UniformInComponentsChirpMass(minimum=15.0, maximum=150.0)
  mass_ratio: bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)
  a_1: bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)
  a_2: bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)
  tilt_1: default
  tilt_2: default
  phi_12: default
  phi_jl: default
  theta_jn: default
  phase: default
  luminosity_distance: 100.0
  geocent_time: 0.0

num_samples: 1_000 #17_000 # 25_000_000

compression:
  whitening: aLIGO_ZERO_DET_high_P_asd.txt
  svd:
    size: 200
    num_training_samples: 10000 #50000
    num_validation_samples: 1000 #10000



"""

@app.cell
def _init_imports():
    """Initialize all required imports for the notebook"""
    import sys
    import marimo as mo
    import json
    import logging
    import numpy as np
    import plotly.graph_objects as go
    from dataclasses import asdict
    from dingo_waveform.dataset.dataset_settings import DatasetSettings
    from dingo_waveform.domains import build_domain
    from dingo_waveform.waveform_generator import build_waveform_generator
    return DatasetSettings, asdict, build_domain, build_waveform_generator, go, json, logging, mo, np, sys


@app.cell
def _init_logging(logging, sys):
    """Initialize logging system - runs once on startup"""
    logger = logging.getLogger('gravitational_waveform_notebook')
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    dingo_logger = logging.getLogger('dingo_waveform')
    dingo_logger.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    logger.info("Logging system initialized for gravitational waveform notebook")

    return (logger,)


@app.cell
def _init_default_configuration():
    """Initialize default waveform configuration - runs once on startup"""
    default_config = {
        "domain": {
            "type": "UniformFrequencyDomain",
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
def _ui_display_header(mo):
    """Display configuration section header"""
    mo.md("## Waveform Generation Configuration")
    return


@app.cell
def _ui_create_config_editor(default_config, json, mo):
    """Create interactive JSON configuration editor"""
    config_editor = mo.ui.text_area(
        value=json.dumps(default_config, indent=2),
        label="Edit configuration (JSON format):",
        rows=30,
        full_width=True,
    )
    config_editor
    return (config_editor,)


@app.cell
def _react_validate_config(config_editor, json, mo, logger):
    """Validate JSON configuration"""
    try:
        edited_config = json.loads(config_editor.value)
        logger.info("Configuration JSON validation successful")
        validation_status = mo.md(f"✅ Configuration is valid JSON")
    except json.JSONDecodeError as e:
        edited_config = None
        logger.error(f"Configuration JSON validation failed: {e}")
        validation_status = mo.md(f"❌ Invalid JSON: {e}")

    validation_status
    return (edited_config,)


@app.cell
def _ui_create_generate_button(edited_config, mo):
    """Create waveform generation button - only when config is valid"""
    if edited_config is None:
        mo.stop("Configuration is invalid - cannot create generate button")

    generate_button = mo.ui.run_button(label="Generate Waveform")
    generate_button
    return (generate_button,)


@app.cell
def _compute_generate_waveform(generate_button, edited_config, DatasetSettings, build_domain, build_waveform_generator,
                               mo, logger):
    """Generate waveform - only runs when user clicks generate button"""

    # Stop execution if button hasn't been clicked - this is the key fix!
    mo.stop(not generate_button.value, "Click 'Generate Waveform' to proceed")

    # Also ensure config is valid
    if edited_config is None:
        mo.stop("Configuration is invalid")

    logger.info("Starting waveform generation process")

    try:
        # Build settings from config
        logger.info("Building dataset settings from configuration")
        settings = DatasetSettings.from_dict(edited_config)

        # Build domain and waveform generator
        logger.info("Building frequency domain")
        domain = build_domain(edited_config["domain"])

        wfg_params = {
            "approximant": edited_config["waveform_generator"]["approximant"],
            "f_ref": edited_config["waveform_generator"]["f_ref"],
            "domain": domain,
        }
        logger.info(f"Building waveform generator with approximant: {wfg_params['approximant']}")
        wfg = build_waveform_generator(wfg_params, domain)

        # Sample parameters from prior
        logger.info("Sampling physical parameters from prior distributions")
        wf_params = settings.intrinsic_prior.sample()
        logger.debug(f"Sampled parameters: {wf_params}")

        # Generate waveform
        logger.info("Generating gravitational waveform polarizations")
        polarization = wfg.generate_hplus_hcross(wf_params)

        waveform_data = {
            "h_plus": polarization.h_plus,
            "h_cross": polarization.h_cross,
            "domain": domain,
            "parameters": wf_params,
        }

        logger.info(f"Waveform generated successfully with shape: {polarization.h_plus.shape}")
        status_msg = mo.md("✅ Waveform generated successfully!")
        status_msg

    except Exception as e:
        logger.error(f"Error during waveform generation: {e}", exc_info=True)
        mo.md(f"❌ Error generating waveform: {e}")
        mo.stop("Waveform generation failed")

    return (waveform_data,)


@app.cell
def _viz_display_waveform_info(waveform_data, asdict, mo, np, logger):
    """Display waveform information - only runs when waveform_data exists and is valid"""
    logger.info("Displaying waveform information and properties")

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
def _viz_plot_components(waveform_data, go, mo, np, logger):
    """Create real/imag component visualization - only runs when waveform_data is valid"""
    logger.info("Creating waveform real/imag component visualization")

    frequencies = waveform_data['domain'].sample_frequencies()
    h_plus = waveform_data['h_plus']
    h_cross = waveform_data['h_cross']

    logger.debug(f"Frequency array shape: {frequencies.shape}, h_plus shape: {h_plus.shape}, h_cross shape: {h_cross.shape}")

    # Create plot of real and imaginary parts for both polarizations
    fig_components = go.Figure()

    # h_plus: real and imaginary
    fig_components.add_trace(go.Scatter(
        x=frequencies,
        y=np.real(h_plus),
        name='Re(h_plus)',
        mode='lines',
        line=dict(color='blue', width=2, dash='solid'),
        legendgroup='h_plus'
    ))
    fig_components.add_trace(go.Scatter(
        x=frequencies,
        y=np.imag(h_plus),
        name='Im(h_plus)',
        mode='lines',
        line=dict(color='blue', width=2, dash='dash'),
        legendgroup='h_plus'
    ))

    # h_cross: real and imaginary
    fig_components.add_trace(go.Scatter(
        x=frequencies,
        y=np.real(h_cross),
        name='Re(h_cross)',
        mode='lines',
        line=dict(color='red', width=2, dash='solid'),
        legendgroup='h_cross'
    ))
    fig_components.add_trace(go.Scatter(
        x=frequencies,
        y=np.imag(h_cross),
        name='Im(h_cross)',
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        legendgroup='h_cross'
    ))

    fig_components.update_layout(
        title="Waveform Components in Frequency Domain",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Strain components",
        xaxis_type="log",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    logger.debug("Real/imag component plot created successfully")
    mo.vstack([
        mo.md("## Frequency-Domain Visualizations"),
        mo.ui.plotly(fig_components)
    ])
    return


@app.cell
def _viz_plot_phase(waveform_data, go, mo, np, logger):
    """Create phase visualization - only runs when waveform_data is valid"""
    logger.info("Creating waveform phase visualization")

    domain_phase = waveform_data['domain']
    frequencies_phase = domain_phase.sample_frequencies()

    # Calculate phases with unwrapping
    h_plus_phase = np.unwrap(np.angle(waveform_data['h_plus']))
    h_cross_phase = np.unwrap(np.angle(waveform_data['h_cross']))

    logger.debug(f"Phase calculation completed for {len(frequencies_phase)} frequency points")

    # Create phase plot
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(
        x=frequencies_phase,
        y=h_plus_phase,
        name='arg(h_plus)',
        mode='lines',
        line=dict(color='blue')
    ))
    fig_phase.add_trace(go.Scatter(
        x=frequencies_phase,
        y=h_cross_phase,
        name='arg(h_cross)',
        mode='lines',
        line=dict(color='red')
    ))
    fig_phase.update_layout(
        title="Waveform Phase in Frequency Domain (Unwrapped)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Phase (radians)",
        xaxis_type="log",
        height=500,
    )

    logger.debug("Phase plot created successfully")
    output_phase = mo.ui.plotly(fig_phase)
    output_phase
    return

@app.cell
def _viz_gwpy_frequency_diagnostics(waveform_data, mo, go, np, logger):
    """GWpy-based frequency-domain diagnostics rendered with Plotly.

    This cell:
      - Wraps |h_plus(f)| and |h_cross(f)| in GWpy FrequencySeries with f0, df metadata
      - Plots |h(f)| and characteristic strain h_c(f) = 2 f |h(f)|
      - Uses Plotly for interactive rendering inside marimo
    """
    logger.info("Creating GWpy-based frequency-domain diagnostics")

    # Try to import GWpy; if unavailable, guide the user
    try:
        from gwpy.frequencyseries import FrequencySeries
    except Exception as e:
        logger.warning(f"GWpy import failed: {e}")
        mo.vstack([
            mo.md("## GWpy Diagnostics"),
            mo.md(
                "❌ GWpy is not available. Install with `pip install gwpy` to enable this cell."
            ),
        ])
        mo.stop("GWpy not installed")

    # Extract data
    freqs = waveform_data["domain"].sample_frequencies()
    h_plus_ = waveform_data["h_plus"]
    h_cross_ = waveform_data["h_cross"]

    # Compute basic derived quantities
    abs_hp = np.abs(h_plus_)
    abs_hx = np.abs(h_cross_)

    # Characteristic strain: h_c(f) = 2 f |h(f)|
    hc_hp = 2.0 * freqs * abs_hp
    hc_hx = 2.0 * freqs * abs_hx

    # Build GWpy FrequencySeries with metadata (f0, df)
    # Use the first frequency as f0 and mean spacing as df, assuming uniform grid
    if freqs.size < 2:
        mo.stop("Insufficient frequency samples for GWpy FrequencySeries")
    f0 = float(freqs[0])
    df = float(np.mean(np.diff(freqs)))

    hp_abs_fs = FrequencySeries(abs_hp, f0=f0, df=df)
    hx_abs_fs = FrequencySeries(abs_hx, f0=f0, df=df)
    hp_hc_fs = FrequencySeries(hc_hp, f0=f0, df=df)
    hx_hc_fs = FrequencySeries(hc_hx, f0=f0, df=df)

    # Prepare for log plotting: avoid zeros by flooring to a tiny positive value
    # This stabilizes Plotly's log-scale rendering
    tiny = 1e-300
    y_abs_hp = np.maximum(hp_abs_fs.value, tiny)
    y_abs_hx = np.maximum(hx_abs_fs.value, tiny)
    y_hc_hp = np.maximum(hp_hc_fs.value, tiny)
    y_hc_hx = np.maximum(hx_hc_fs.value, tiny)

    # Plot 1: |h(f)| for both polarizations
    fig_abs = go.Figure()
    fig_abs.add_trace(go.Scatter(
        x=freqs, y=y_abs_hp, mode="lines", name="|h_plus(f)|", line=dict(color="blue")
    ))
    fig_abs.add_trace(go.Scatter(
        x=freqs, y=y_abs_hx, mode="lines", name="|h_cross(f)|", line=dict(color="red")
    ))
    fig_abs.update_layout(
        title="GWpy FrequencySeries: |h(f)|",
        xaxis_title="Frequency (Hz)",
        yaxis_title="|h(f)| (dimensionless strain)",
        xaxis_type="log",
        yaxis_type="log",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    # Plot 2: Characteristic strain h_c(f) = 2 f |h(f)|
    fig_hc = go.Figure()
    fig_hc.add_trace(go.Scatter(
        x=freqs, y=y_hc_hp, mode="lines", name="h_c, plus", line=dict(color="blue", dash="solid")
    ))
    fig_hc.add_trace(go.Scatter(
        x=freqs, y=y_hc_hx, mode="lines", name="h_c, cross", line=dict(color="red", dash="solid")
    ))
    fig_hc.update_layout(
        title="Characteristic Strain: h_c(f) = 2 f |h(f)|",
        xaxis_title="Frequency (Hz)",
        yaxis_title="h_c(f) (dimensionless)",
        xaxis_type="log",
        yaxis_type="log",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    logger.info("GWpy diagnostics created successfully")

    mo.vstack([
        mo.md("## GWpy Frequency-Domain Diagnostics"),
        mo.md(
            "- This view wraps arrays into GWpy FrequencySeries with f0, df metadata for domain-aware handling. "
            "Plots show |h(f)| and characteristic strain h_c(f) for quick validation before export."
        ),
        mo.ui.plotly(fig_abs),
        mo.ui.plotly(fig_hc),
    ])
    return



if __name__ == "__main__":
    app.run()