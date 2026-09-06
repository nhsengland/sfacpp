import atexit

try:
    import pysfacpp_internal
    _HAS_BACKEND = True
except ImportError:
    _HAS_BACKEND = False

from pysfacpp.monte_carlo import (
    DGPConfig,
    DGPData,
    ClassSpec,
    InefficiencyDist,
    EstimationResult,
    MCMetrics,
    MCConfig,
    generate_data,
    estimate_lcm,
    compute_metrics,
    run_monte_carlo,
)

if _HAS_BACKEND:
    from pysfacpp.model import (
        PySfaCpp,
        PySfaCppLcm,
        PySfaCppLcmCross,
        PySfaCppResult,
        PySfaCppLcmResult,
        PySfaCppLcmCrossResult,
    )

    def _shutdown_cleanup() -> None:
        try:
            pysfacpp_internal.teardown_logger()
        except Exception:
            pass
        try:
            pysfacpp_internal.flush_tls()
        except Exception:
            pass

    atexit.register(_shutdown_cleanup)