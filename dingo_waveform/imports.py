import importlib
import inspect
from typing import Any, Callable, List, Optional, Tuple, Type, Union, get_type_hints


def check_function_signature(
    fn: Callable, expected_param_types: List[Type], expected_return_type: Type
) -> bool:
    """
    Check if the function signature matches the expected signature.

    Parameters
    ----------
    fn : Callable
        The function to check.
    expected_param_types : List[type]
        The expected types of the function's parameters.
    expected_return_type : type
        The expected return type of the function.

    Returns
    -------
    bool
        True if the function signature matches, False otherwise.
    """

    sig = inspect.signature(fn)
    params = sig.parameters
    return_annotation = sig.return_annotation

    # Get type hints for the function
    type_hints = get_type_hints(fn)

    # Check return type
    if (
        return_annotation is not inspect.Signature.empty
        and return_annotation != expected_return_type
    ):
        return False

    # Check parameter types
    param_types = [type_hints.get(name, None) for name in params]

    # Check if all expected_param_types are in param_types
    for expected_type in expected_param_types:
        if expected_type not in param_types:
            return False

    # Check if all required parameters are covered
    for name, param in params.items():
        if (
            param.default is inspect.Parameter.empty
            and type_hints.get(name, None) not in expected_param_types
        ):
            return False

    return True


def import_entity(import_path: str) -> Tuple[Any, str, str]:
    """
    Given an import path as a string, returns a tuple containing the imported entity,
    the related module (as a string), and the entity name (as a string).

    Parameters
    ----------
    import_path : The import path as a string.

    Returns
    -------
    A tuple containing the imported entity, module path, and entity name.

    Raises
    ------
    ImportError
        If the import failed.
    """
    # Split the import path to get the module path and entity name
    *module_parts, entity_name = import_path.split(".")
    module_path = ".".join(module_parts)

    try:
        # Import the module dynamically
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")

    try:
        # Retrieve the entity from the module
        entity = getattr(module, entity_name)
    except AttributeError as e:
        raise ImportError(
            f"Could import module '{module_path}', but not '{entity_name}'. Error: {e}"
        )

    return entity, module_path, entity_name


def import_function(
    import_path_or_fn: Optional[Union[str, Callable]],
    expected_param_types: List[type],
    expected_return_type: type,
) -> Optional[Callable]:
    if import_path_or_fn is None:
        return None
    if not isinstance(import_path_or_fn, str):
        return import_path_or_fn
    fn, _, entity_name = import_entity(import_path_or_fn)
    proper_signature: bool = check_function_signature(
        fn, expected_param_types, expected_return_type
    )
    if not proper_signature:
        raise ValueError(
            f"imported function {entity_name} from import path {import_path_or_fn}, "
            f"but this function does not have the expected signature "
            f"(args: {expected_return_type}, return: {expected_return_type}"
        )
    return fn
