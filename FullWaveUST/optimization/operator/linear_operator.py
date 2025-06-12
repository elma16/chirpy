from .base import Operator


class LinearOperator(Operator):
    """Adds the adjoint mapping to :class:`Operator`."""

    @abstractmethod
    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint map ``x = A* (y)``."""
        ...

    # ---------- spectral-norm & unit-test helpers -------------------- #
    def is_linear(self) -> bool:
        return True

    def dot_test(self, tol: float = 1e-6) -> bool:
        """
        Dot-product test :math:`⟨Ax,\,y⟩ = ⟨x,\,A^*y⟩`.

        Returns *True* if the relative mismatch < *tol*.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal(self.domain_shape)
        y = rng.standard_normal(self.range_shape)
        lhs = np.vdot(self.apply(x), y)
        rhs = np.vdot(x, self.adjoint(y))
        return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12) < tol
