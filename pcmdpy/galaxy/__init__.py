__all__ = ['sfhmodels', 'distancemodels', 'dustmodels', 'metalmodels',
           'galaxy']

from . import sfhmodels
from . import distancemodels
from . import dustmodels
from . import metalmodels
from . import galaxy


# rename for backwards compatability
agemodels = sfhmodels
