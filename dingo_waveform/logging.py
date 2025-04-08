from dataclasses import Field, fields, is_dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Optional, Protocol, TypeVar, Union

import numpy as np
from rich.console import Console
from rich.style import Style
from rich.table import Table

# This will be used by the 'to_table' method in this module
# to 'cast' type to a color (e.g. str value will be printed in
# cyan in the table)
_TYPE_STYLES = {
    str: Style(color="cyan"),
    int: Style(color="yellow"),
    float: Style(color="green"),
    bool: Style(color="blue"),
    datetime: Style(color="magenta"),
    np.ndarray: Style(color="purple"),
}

# to complete _TYPE_STYLES. None values will be grey.
_NONE_STYLE = Style(color="grey50")


class DataclassType(Protocol):
    # typing hints protocol for dataclasses.
    # For reason I do not fully understand, dataclass is not a type.
    # Therefore typing hints like 'f(a: dataclass)->None' is not valid.
    # This is a protocol to implement a similar feature.
    # e.g. 'f(a: DataclassType)->None'.
    # (I am not sure it works that well).
    __dataclass_fields__: dict[str, Field]
    __dataclass_params__: dict[str, Any]
    __post_init__: Optional[Callable[[], None]]


def _get_value_style(value: Any) -> Style:
    # returns the Style (i.e. the color) associated with the value,
    # based on _TYPE_STYLES (defined in this module) or _NONE_STYLE
    # (if value is None).
    if value is None:
        return _NONE_STYLE
    value_type = type(value)
    return _TYPE_STYLES.get(value_type, Style())


def _get_type_representation(value: Any) -> str:
    # return a str representation of value based on its type.
    # e.g.if value is an int: simply 'int'.
    # For numpy array, the shape and dtype are also indicated.
    # If you want a type to be associated with more info in
    # to_table (see below), this is the place to edit.
    if isinstance(value, np.ndarray):
        return f"numpy.ndarray(shape={value.shape}, dtype='{value.dtype}')"

    value_type = type(value)
    repr_ = str(value_type).replace("typing.", "")
    return repr_.replace("class ", "").replace("<'", "").replace("'>", "")


def to_table(
    instance: Union[DataclassType, Dict[str, Any]], console: Optional[Console] = None
) -> str:
    """
    Convert a dataclass instance or a dictionary to a formatted table string.
    Useful for informative logging. Based on the "rich" package.

    Parameters
    ----------
    instance:
      The dataclass or dictionary to represent as a table
    console:
      The console used for printing to stdout. If None, one will be created.

    Returns
    -------
    A string representing the instance as a formatted table with columns
    for field name, type, and value
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
    """
    Mixin for dataclasses, adding the 'to_table' method. This method returns
    a table string representation of the dataclass, one line per field. Each
    line contains the field name, its type and its value.

    Usage:

    ```
    @dataclass
    class A(TableStr):
      a: int
      b: float
      c: str

    a = A(a=1, b=2.0, c="hello world")
    table_title = 'values of variable a'
    table = a.to_table(table_title)
    logger.info(table)

    # prints something like:

    # values of variable a
    # a    int    1
    # b    float  2.0
    # c    str    hello_world
    ```
    """

    # console (tool for printing to stdout) to be reused over
    _console = Console()

    def to_table(self, header: Optional[str] = None) -> str:
        """
        Returns a table representation of 'self', 'self' expected to be
        an instance of dataclass.

        Parameters
        ----------
        header
          title that will be including in the table representation

        Returns
        -------
        The table.
        """
        if header is None:
            return to_table(self._console, self)  # type: ignore
        return header + "\n" + to_table(self, console=self._console)  # type: ignore
