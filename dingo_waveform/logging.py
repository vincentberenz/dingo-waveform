import logging
from dataclasses import Field, fields, is_dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Optional, Protocol, TypeVar, Union

import numpy as np
from rich.console import Console
from rich.style import Style
from rich.table import Table

_TYPE_STYLES = {
    str: Style(color="cyan"),
    int: Style(color="green"),
    float: Style(color="yellow"),
    bool: Style(color="blue"),
    datetime: Style(color="magenta"),
    np.ndarray: Style(color="purple"),  # New style for numpy arrays
}

NONE_STYLE = Style(color="grey50")


class DataclassType(Protocol):
    __dataclass_fields__: dict[str, Field]
    __dataclass_params__: dict[str, Any]
    __post_init__: Optional[Callable[[], None]]


def _get_value_style(value: Any) -> Style:
    """Get the appropriate style for a value based on its type."""
    if value is None:
        return NONE_STYLE

    # Get the base type (strip Optional/Union wrappers)
    value_type = type(value)

    return _TYPE_STYLES.get(value_type, Style())


def _get_type_representation(value: Any) -> str:
    """Generate detailed type information, including numpy array specifics."""
    if isinstance(value, np.ndarray):
        return f"numpy.ndarray(shape={value.shape}, dtype='{value.dtype}')"

    # Handle other types as before
    value_type = type(value)
    repr_ = str(value_type).replace("typing.", "")
    return repr_.replace("class ", "").replace("<'", "").replace("'>", "")


def to_table(
    instance: Union[DataclassType, Dict[str, Any]], console: Optional[Console] = None
) -> str:
    """
    Convert a dataclass instance or a dictionary to a formatted table string.

    Args:
        instance: An instance of a dataclass or a dictionary

    Returns:
        A string representing a formatted table with columns for field name, type, and value
    """

    if console is None:
        console = Console()

    # If instance is a dataclass, convert it to a dictionary
    if is_dataclass(instance):
        instance = {
            field.name: getattr(instance, field.name) for field in fields(instance)
        }

    # Ensure instance is a dictionary
    if not isinstance(instance, dict):
        raise ValueError("Instance must be a dataclass or a dictionary.")

    table = Table()

    # Add columns
    table.add_column("Field", style="bold")
    table.add_column("Type")
    table.add_column("Value")

    # Process each key-value pair
    for key, value in instance.items():
        value_style = _get_value_style(value)

        # Get enhanced type representation
        type_repr = _get_type_representation(value)

        # Format value representation for numpy arrays
        if isinstance(value, np.ndarray):
            value_repr = f"Array({value.shape})"
        else:
            value_repr = str(value)

        # Add row with styled value
        table.add_row(key, type_repr, value_repr, style=value_style)

    # Capture the output as a string
    with console.capture() as capture:
        console.print(table)

    return capture.get()


class TableStr:

    _console = Console()

    def to_table(self, header: Optional[str] = None) -> str:
        if header is None:
            return to_table(self._console, self)  # type: ignore
        return header + "\n" + to_table(self, console=self._console)  # type: ignore
