# Documentation

This directory contains the documentation for dingo-waveform, built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Building the Documentation

### Install Dependencies

```bash
pip install -e ".[docs]"
```

### Build Documentation

To build the documentation to the `site/` directory:

```bash
mkdocs build
```

### Serve Documentation Locally

To preview the documentation with live reloading:

```bash
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

### Deploy to GitHub Pages

To deploy documentation to GitHub Pages:

```bash
mkdocs gh-deploy
```

## Documentation Structure

```
docs/
├── index.md                 # Home page
├── getting-started/         # Installation and quick start
│   ├── installation.md
│   └── quickstart.md
├── concepts/                # Core concepts
│   ├── overview.md
│   ├── domains.md
│   ├── approximants.md
│   ├── polarizations.md
│   └── modes.md
├── cli/                     # CLI tool documentation
│   ├── dingo-verify.md
│   ├── dingo-plot.md
│   └── dingo-generate-dataset.md
├── examples/                # Tutorials and examples
│   ├── basic-waveform.md
│   ├── mode-separated.md
│   ├── plotting.md
│   └── dataset-generation.md
└── api/                     # API reference (auto-generated)
    ├── waveform-generator.md
    ├── domains.md
    ├── approximants.md
    ├── polarizations.md
    ├── plotting.md
    └── dataset.md
```

## Writing Documentation

### Markdown Features

The documentation supports:

- **Math equations**: Use LaTeX math with `\( inline \)` or `\[ display \]`
- **Code blocks**: Syntax highlighted code with ` ```python `
- **Admonitions**: Notes, warnings, etc. with `!!! note`
- **Tables**: Standard markdown tables
- **Tabs**: Content tabs with `=== "Tab Title"`

### API Documentation

API reference pages use [mkdocstrings](https://mkdocstrings.github.io/) to auto-generate documentation from docstrings:

```markdown
::: dingo_waveform.WaveformGenerator
    options:
      show_root_heading: true
      show_source: true
```

### Docstring Format

Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html):

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Short description.

    Longer description with more details.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> my_function("test", 42)
    True
    """
    return True
```

## Configuration

The documentation configuration is in `mkdocs.yml` at the repository root.

Key settings:

- **Theme**: Material for MkDocs with dark/light mode
- **Plugins**: mkdocstrings for API docs, search
- **Extensions**: Math support, code highlighting, admonitions
- **Navigation**: Defined in the `nav` section

## Contributing

When adding new features:

1. Update or create relevant documentation pages
2. Ensure docstrings are complete and follow NumPy style
3. Build and verify documentation locally
4. Add examples where appropriate

## Troubleshooting

### Missing Dependencies

If mkdocs commands fail, reinstall documentation dependencies:

```bash
pip install -e ".[docs]" --force-reinstall
```

### Broken Links

Check for broken links:

```bash
mkdocs build --strict
```

### Docstring Warnings

Fix docstring format issues reported during build. Common issues:

- Missing parameter descriptions
- Mismatched parameter names
- Incorrect section headers (should be underlined with dashes)
