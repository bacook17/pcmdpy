__all__ = ['sfhmodels', 'distancemodels', 'dustmodels', 'metalmodels',
           'galaxy', 'imf']

from . import sfhmodels
from . import distancemodels
from . import dustmodels
from . import metalmodels
from . import galaxy
from . import imf

# rename for backwards compatability
agemodels = sfhmodels
