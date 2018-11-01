__all__ = ['priors', 'results', 'gpu_utils', 'utils', 'driver',
           'fit_model', 'galaxy', 'instrument', 'isochrones', 'plotting',
           'metalmodels', 'agemodels', 'dustmodels', 'distancemodels',
           '__version__']

from .simulation import gpu_utils
from . import instrument
from .simulation import driver
from .isochrones import isochrones
from .galaxy import (galaxy, metalmodels, dustmodels, agemodels,
                     distancemodels)
from .sampling import (fit_model, priors, results)
from .utils import (plotting, utils)

if gpu_utils._GPU_AVAIL:
    import warnings
    warnings.warn(
        ("GPU Acceleration is available. To activate, run:\n"
         "pcmdpy.gpu_utils.initialize_gpu()"),
        RuntimeWarning
    )

__version__ = "0.5.1"
