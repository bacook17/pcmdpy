__all__ = ['priors', 'gpu_utils', 'utils', 'driver',
           'fit_model', 'galaxy', 'instrument', 'isochrones', 'plotting',
           'metalmodels', 'agemodels', 'dustmodels', 'distancemodels',
           '__version__']

from . import gpu_utils
from . import utils
from . import driver
from . import fit_model
from . import metalmodels
from . import dustmodels
from . import agemodels
from . import distancemodels
from . import galaxy
from . import instrument
from . import isochrones
from . import priors
from . import plotting

if gpu_utils._GPU_AVAIL:
    import warnings
    warnings.warn(
        ("GPU Acceleration is available. To activate, run:\n"
         "pcmdpy.gpu_utils.initialize_gpu()"),
        RuntimeWarning
    )

__version__ = "0.4.7"
