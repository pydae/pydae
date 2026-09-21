from pydae.core.builder.casadi_builder import CasadiBuilder, MathBackend
from pydae.core.builder.sympy_builder import Builder
from pydae.core.model.casadi_model import CasadiModel

__all__ = ["Builder", "CasadiBuilder", "CasadiModel", "MathBackend",
           "MlxBuilder", "MlxSweep", "MlxSmallSignal", "JacobianPattern"]


def __getattr__(name):
    """Expose the MLX classes lazily: importing them pulls in mlx (Apple Silicon only)."""
    if name == "MlxBuilder":
        from pydae.core.builder.mlx_builder import MlxBuilder
        return MlxBuilder
    if name == "MlxSweep":
        from pydae.core.builder.mlx_sweep import MlxSweep
        return MlxSweep
    if name == "MlxSmallSignal":
        from pydae.core.builder.mlx_ssa import MlxSmallSignal
        return MlxSmallSignal
    if name == "JacobianPattern":
        from pydae.core.builder.mlx_sparsity import JacobianPattern
        return JacobianPattern
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
