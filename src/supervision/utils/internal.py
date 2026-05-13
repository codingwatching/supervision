from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import Callable
from typing import Any, Generic, TypeVar


class SupervisionWarnings(Warning):
    """Supervision warning category.
    Set the deprecation warnings visibility for Supervision library.
    You can set the environment variable SUPERVISON_DEPRECATION_WARNING to '0' to
    disable the deprecation warnings.
    """

    pass


def format_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """
    Format a warning the same way as the default formatter, but also include the
    category name in the output.
    """
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = format_warning

if os.getenv("SUPERVISON_DEPRECATION_WARNING") == "0":
    warnings.simplefilter("ignore", SupervisionWarnings)
else:
    warnings.simplefilter("always", SupervisionWarnings)


T = TypeVar("T")


class classproperty(Generic[T]):
    """
    A decorator that combines @classmethod and @property.
    It allows a method to be accessed as a property of the class,
    rather than an instance, similar to a classmethod.

    Usage:
        @classproperty
        def my_method(cls):
            ...
    """

    def __init__(self, fget: Callable[..., T]):
        """
        Args:
            The function that is called when the property is accessed.
        """
        self.fget = fget

    def __get__(self, owner_self: Any, owner_cls: type | None = None) -> T:
        """
        Override the __get__ method to return the result of the function call.

        Args:
            owner_self: The instance through which the attribute was accessed, or None.
                Irrelevant for class properties.
            owner_cls: The class through which the attribute was accessed.

        Returns:
            The result of calling the function stored in 'fget' with 'owner_cls'.
        """
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(owner_cls)


def get_instance_variables(instance: Any, include_properties: bool = False) -> set[str]:
    """
    Get the public variables of a class instance.

    Args:
        instance: The instance of a class
        include_properties: Whether to include properties in the result

    Usage:
        ```pycon
        >>> from supervision.utils.internal import get_instance_variables
        >>> import numpy as np
        >>> from supervision import Detections
        >>> detections = Detections(xyxy=np.array([[1, 2, 3, 4]]))
        >>> variables = get_instance_variables(detections)
        >>> 'xyxy' in variables
        True
        >>> 'data' in variables
        True

        ```
    """
    if isinstance(instance, type):
        raise ValueError("Only class instances are supported, not classes.")

    fields = {
        name
        for name, val in inspect.getmembers(instance)
        if not callable(val) and not name.startswith("_")
    }

    if not include_properties:
        properties = {
            name
            for name, val in inspect.getmembers(instance.__class__)
            if isinstance(val, property)
        }
        fields -= properties

    return fields
