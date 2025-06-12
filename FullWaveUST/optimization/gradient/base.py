import abc
import numpy as np
from typing import Callable, Optional, Any


class GradientEvaluator(abc.ABC):
    """
    Base class for model-space gradient computation.

    Users can inject a custom gradient function via `deriv_fn`, or subclass
    and override `evaluate(model, data)` for specialized logic.
    """

    def __init__(
        self,
        deriv_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    ):
        """
        Parameters
        ----------
        deriv_fn : callable, optional
            User-provided function `deriv_fn(model, data) -> gradient`.
        """
        self.deriv_fn = deriv_fn
        self.op: Optional[Any] = None

    def set_deriv_fn(
        self,
        fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> None:
        """
        Set or override the gradient function.

        Parameters
        ----------
        fn : callable
            Signature `fn(model, data) -> gradient`.
        """
        self.deriv_fn = fn

    def attach_operator(self, op: Any) -> None:
        """
        Optionally attach an operator for computing `data` if not provided.

        Parameters
        ----------
        op : any
            Must provide `residual(model)` to generate default `data`.
        """
        self.op = op

    @abc.abstractmethod
    def evaluate(
        self,
        model: np.ndarray,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient from provided `data`.

        Subclasses should override this if no `deriv_fn` is supplied.

        Parameters
        ----------
        model : ndarray
            Model parameters array.
        data : ndarray
            Input data (e.g., residuals) for gradient computation.

        Returns
        -------
        ndarray
            Gradient image, shape (ny, nx) or (ny, nx, N).
        """
        ...

    def gradient(
        self,
        model: np.ndarray,
        data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute gradient image. Uses `deriv_fn` if provided, otherwise `evaluate()`.

        If `data` is None and an operator is attached, data is set to `op.residual(model)`.
        """
        if data is None:
            if self.op is None:
                raise RuntimeError("No data provided and no operator attached.")
            data = self.op.residual(model)  # type: ignore[attr-defined]

        grad = self.deriv_fn(model, data) if self.deriv_fn else self.evaluate(model, data)
        # Sum over channel axis if present
        return grad.sum(axis=-1) if grad.ndim == 3 else grad

    __call__ = gradient  # allow instance to be called directly
