__all__ = ['priors', 'gpu_utils', 'utils', 'driver',
           'fit_model', 'galaxy', 'instrument', 'isochrones', 'plotting',
           'metalmodels', 'agemodels', 'dustmodels', 'distancemodels',
           '__version__']

from .simulation import driver
from .simulation import gpu_utils
from .sampling import fit_model
from .sampling import priors
from .galaxy import metalmodels
from .galaxy import dustmodels
from .galaxy import agemodels
from .galaxy import distancemodels
from .galaxy import galaxy
from . import instrument
from .isochrones import isochrones, Isochrone_Model
from .utils import plotting
from .utils import utils

if gpu_utils._GPU_AVAIL:
    import warnings
    warnings.warn(
        ("GPU Acceleration is available. To activate, run:\n"
         "pcmdpy.gpu_utils.initialize_gpu()"),
        RuntimeWarning
    )

__version__ = "0.4.9"
