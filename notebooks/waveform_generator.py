import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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
def _(default_config, mo):
    import json

    config_editor = mo.ui.text_area(
        value=json.dumps(default_config, indent=2),
        label="Edit configuration (JSON format):",
        rows=30,
        full_width=True,
    )
    config_editor
    return config_editor, json


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
def _(edited_config, generate_button, mo):
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print(f"DEBUG: Button value = {generate_button.value}")
    print(f"DEBUG: edited_config is None = {edited_config is None}")

    # Generate waveform when button is clicked
    # Button value is None initially, then becomes a counter
    if generate_button.value is not None and generate_button.value > 0 and edited_config is not None:
        print("DEBUG: Inside generation block")
        from dingo_waveform.dataset.dataset_settings import DatasetSettings
        from dingo_waveform.domains import build_domain
        from dingo_waveform.waveform_generator import build_waveform_generator

        try:
            print("Starting waveform generation...")
            logger.info("Starting waveform generation...")

            # Build settings from config
            print("Building settings from config")
            settings = DatasetSettings.from_dict(edited_config)

            # Build domain and waveform generator
            print("Building domain")
            domain = build_domain(edited_config["domain"])
            wfg_params = {
                "approximant": edited_config["waveform_generator"]["approximant"],
                "f_ref": edited_config["waveform_generator"]["f_ref"],
                "domain": domain,
            }
            print(f"Building waveform generator with approximant {wfg_params['approximant']}")
            wfg = build_waveform_generator(wfg_params, domain)

            # Sample parameters from prior
            print("Sampling parameters from prior")
            wf_params = settings.intrinsic_prior.sample()

            # Generate waveform
            print("Generating waveform polarizations")
            polarization = wfg.generate_hplus_hcross(wf_params)

            waveform_data = {
                "h_plus": polarization.h_plus,
                "h_cross": polarization.h_cross,
                "domain": domain,
                "parameters": wf_params,
            }

            print(f"Waveform generated successfully with shape {polarization.h_plus.shape}")
            status_msg = mo.md("✅ Waveform generated successfully!")

        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Error generating waveform: {e}", exc_info=True)
            waveform_data = None
            status_msg = mo.md(f"❌ Error generating waveform: {e}")
    else:
        print("DEBUG: Button not clicked or config is None")
        waveform_data = None
        status_msg = mo.md("")

    status_msg
    return (waveform_data,)


@app.cell
def _(mo, waveform_data):
    if waveform_data is not None:
        import numpy as np
        from dataclasses import asdict

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

        output = [
            mo.md(info),
            mo.tree(asdict(waveform_data['parameters']))
        ]
        mo.vstack(output)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
