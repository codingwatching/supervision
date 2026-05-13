from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def deprecated(
    *,
    target: Any = ...,
    deprecated_in: str | None = ...,
    remove_in: str | None = ...,
    **kwargs: Any,
) -> Callable[[F], F]: ...
def deprecated_class(
    *,
    target: Any = ...,
    deprecated_in: str | None = ...,
    remove_in: str | None = ...,
    **kwargs: Any,
) -> Callable[[type[Any]], type[Any]]: ...
def void(*args: Any, **kwargs: Any) -> Any: ...
