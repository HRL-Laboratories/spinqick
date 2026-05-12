from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spinqick.core.spinqick_data import CompositeSpinqickData, SpinqickData


class DataHandler(ABC):
    """Interface for persisting SpinqickData to a storage backend.

    Handles saving and loading of SpinqickData objects.
    """

    @abstractmethod
    def save(self, data: SpinqickData | CompositeSpinqickData) -> Any:
        """Save data. Dispatches internally for composite vs single.

        Returns a backend-specific handle.
        """
        ...

    @abstractmethod
    def load(self, identifier: str) -> SpinqickData | CompositeSpinqickData:
        """Load data by identifier. Auto-detects type from stored metadata."""
        ...

    @abstractmethod
    def save_plot(self, handle: Any, fignum: int | str | None = None) -> None:
        """Save a matplotlib figure to the dataset identified by *handle*."""
        ...

    @abstractmethod
    def close(self, handle: Any) -> None:
        """Close / finalise the dataset identified by *handle*."""
        ...

    # Shared helpers available to all backends:
    @staticmethod
    def get_sweep_vars(axis_dict: dict) -> dict:
        """Extract sweep variables from a flat axis dict."""
        return {
            k: v for k, v in axis_dict.items() if isinstance(v, dict) and "data" in v
        }
