from __future__ import annotations

import copy
import importlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Any

from graph_pes.utils.logger import logger

ALLOWED_PACKAGES = {"torch", "graph_pes"}


def is_allowed_for_import(nested_module_name: str) -> bool:
    """
    Check if a nested module is allowed for import.

    Parameters
    ----------
    nested_module_name
        The name of the nested module.

    Returns
    -------
    bool
        Whether the nested module is allowed for import.
    """

    if any(
        nested_module_name.startswith(package) for package in ALLOWED_PACKAGES
    ):
        return True

    user_string = os.environ.get("GRAPH_PES_ALLOW_IMPORT", "")
    if not user_string:
        return False
    user_specified = set(user_string.split(","))
    logger.debug(f"User specified allowed packages: {user_specified}")
    return any(
        nested_module_name.startswith(package) for package in user_specified
    )


def create_from_data(thing: Any, type_msg: str | None = None) -> Any:
    """
    Create python objects from a data specification.

    Parameters
    ----------
    thing
        A string or a dictionary.
    type_msg
        A message to display if the type of ``thing`` is not recognized.

    Returns
    -------
    Any
        The created object.

    Example
    -------
    >>> create_from_data("torch.nn.ReLU")
    ReLU()
    """

    if isinstance(thing, str):
        return create_from_string(thing)
    elif isinstance(thing, dict):
        return create_from_dict(thing)
    else:
        if type_msg is None:
            type_msg = f"Expected {thing} to be a string or a dictionary."
        raise TypeError(type_msg)


def _import(thing: str) -> Any:
    """
    Import a module or object from a fully qualified name.

    Example
    -------
    >>> _import("torch.nn.Tanh")
    <class 'torch.nn.modules.activation.Tanh'>
    """

    module_name, obj_name = thing.rsplit(".", 1)
    if not is_allowed_for_import(module_name):
        raise ImportError(
            f"Attempted to import {obj_name} from {module_name}, "
            "which is not allowed for safety reasons. Do you want to "
            "allow this import? If so, set the GRAPH_PES_ALLOW_IMPORT "
            "environment variable to include the package you want to "
            "import from: e.g. `export GRAPH_PES_ALLOW_IMPORT=my_package_1,"
            "my_package_2`."
        )

    dir_command_was_run_from = str(Path(os.getcwd()).resolve())
    if dir_command_was_run_from not in sys.path:
        sys.path.append(dir_command_was_run_from)

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        assumed_file = Path(module_name.replace(".", "/")).with_suffix(".py")
        if not assumed_file.exists():
            raise ImportError(
                f"While attempting to import {obj_name} from {module_name}, "
                "we could not find the module or the file "
                f"belonging to {module_name}."
            ) from None

        logger.debug(
            f"Could not directly import '{module_name}' - "
            f"trying to load it from {assumed_file}"
        )
        spec = importlib.util.spec_from_file_location(
            name=module_name, location=assumed_file
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return getattr(module, obj_name)


def warn_about_import_error(s: str):
    if not Path(s).exists():
        warnings.warn(
            f"Encountered a string ({s}) that looks like it "
            "could be meant to be imported - we couldn't do "
            "this. This may cause issues later.",
            stacklevel=2,
        )


def create_from_dict(d: dict[str, Any]) -> dict[str, Any] | Any:
    """
    Create objects from a dictionary.

    Two cases:
    1. ``d`` has a single key, and the value is a dictionary. In this case, we
       try to import the key (assuming it points to some callable), parse
       the value dictionary as keyword arguments and call the imported object
       with these arguments.
    2. leave the keys of ``d`` as they are, recursively create objects from
       the values and return the resulting dictionary.

    Parameters
    ----------
    d
        A dictionary.

    Returns
    -------
    dict[str, Any] | Any
        The resulting dictionary or object.

    Example
    -------
    A single-key dictionary with a dictionary value gets mapped to an object:

    >>> d = {"torch.nn.Linear": {"in_features": 10, "out_features": 20}}
    >>> create_from_dict(d)
    Linear(in_features=10, out_features=20, bias=True)

    A dictionary with multiple keys gets mapped to a dictionary:
    >>> d = {
    ...     "a": {"b": 1},
    ...     "c": "torch.nn.ReLU()",
    ... }
    >>> create_from_dict(d)
    {'a': {'b': 1}, 'c': ReLU()}

    Note that the string "torch.nn.ReLU()" is imported and called using this
    syntax.
    """
    return _create_from_dict(d, depth=0)


def _create_from_dict(d: dict[str, Any], depth: int) -> dict[str, Any] | Any:
    new_d = copy.deepcopy(d)

    def log(msg):
        logger.debug(f"create_from_dict: {'    ' * depth}{msg}")

    log(f"creating from {d}")

    # 1. recursively create objects from values
    for k, v in new_d.items():
        log(f"{k=}, {v=}")
        if isinstance(v, dict):
            log(f"recursing into {v=}")
            new_d[k] = _create_from_dict(v, depth + 1)
            log(f"received {new_d[k]}")

        elif isinstance(v, str) and looks_importable(v):
            try:
                log(f"trying to import '{v}'")
                obj = create_from_string(v)
                log(f"successfully imported {obj}")
                new_d[k] = obj

            except ImportError:
                log(f"failed to import '{v}'")
                warn_about_import_error(v)

    # 2. if dict has only one key, and that key maps to a dictionary of values
    #    try to import the key, call it with the values as kwargs and
    #    return the result. If any of this fails, return the dict as is.
    if len(new_d) == 1:
        k, kwargs = next(iter(new_d.items()))
        if isinstance(kwargs, dict) and looks_importable(k):
            log(
                "found a single-item dict with importable "
                f"key - trying to import '{k}' and call it with {kwargs}"
            )
            try:
                callable = _import(k)
            except ImportError:
                warn_about_import_error(k)
                callable = None

            if callable is not None:
                try:
                    result = callable(**kwargs)
                except Exception as e:
                    log(f"failed to call {callable} with {kwargs}: {e}")
                    raise e

                log(f"successfully created {result}")
                log(f"final result: {result}")
                return result

    # 3. if dict has more than one key, return it as is
    log(f"final result: {new_d}")
    return new_d


def looks_importable(s: str):
    return "." in s


def create_from_string(s: str):
    if s.endswith("()"):
        return _import(s[:-2])()
    else:
        return _import(s)
