__all__ = ['priors', 'results', 'gpu_utils', 'utils', 'driver',
           'fit_model', 'galaxy', 'instrument', 'isochrones', 'plotting',
           'metalmodels', 'agemodels', 'dustmodels', 'distancemodels',
           'sfhmodels', '__version__']

from .simulation import gpu_utils
from . import instrument
from .simulation import driver
from .isochrones import isochrones
from .galaxy import (galaxy, metalmodels, dustmodels, sfhmodels,
                     distancemodels)
from .sampling import (fit_model, priors, results)
from .utils import (utils)
from .plotting import (plotting)

GPU_AVAIL = gpu_utils._GPU_AVAIL
GPU_ACTIVE = gpu_utils._GPU_ACTIVE

# rename for backwards compatability
gpu_utils._CUDAC_AVAIL = GPU_ACTIVE
agemodels = sfhmodels

__version__ = "0.8.0"
