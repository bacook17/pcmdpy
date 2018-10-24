__all__ = ['priors', 'gpu_utils', 'utils', 'driver',
           'fit_model', 'galaxy', 'instrument', 'isochrones', 'plotting', 
           'metalmodels', 'agemodels', 'dustmodels', 'distancemodels']

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
    try:
        gpu_utils.initialize_gpu()
    except Exception as e:
        print('No GPU Acceleration Available')

__version__ = "0.4.4"
