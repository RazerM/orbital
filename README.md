## Orbital

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/RazerM/orbital?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status][bsi]][bsl] [![PyPI Version][ppi]][ppl] [![Python Version][pvi]][pvl] [![MIT License][mli]][mll]

  [bsi]: http://img.shields.io/travis/RazerM/orbital.svg?style=flat-square
  [bsl]: https://travis-ci.org/RazerM/orbital
  [ppi]: http://img.shields.io/pypi/v/orbitalpy.svg?style=flat-square
  [ppl]: https://pypi.python.org/pypi/orbitalpy/
  [pvi]: https://img.shields.io/badge/python-2.7%2C%203-brightgreen.svg?style=flat-square
  [pvl]: https://www.python.org/downloads/
  [mli]: http://img.shields.io/badge/license-MIT-blue.svg?style=flat-square
  [mll]: https://raw.githubusercontent.com/RazerM/orbital/master/LICENSE

Orbital is a high level orbital mechanics package for Python.

### Installation

```bash
$ pip install orbitalpy
```

### Example

```python
from orbital import earth, KeplerianElements, Maneuver, plot

from scipy.constants import kilo
import matplotlib.pyplot as plt

orbit = KeplerianElements.with_altitude(1000 * kilo, body=earth)
man = Maneuver.hohmann_transfer_to_altitude(10000 * kilo)
plot(orbit, title='Maneuver 1', maneuver=man)
plt.show()
```

![Example plot](http://i.fraz.eu/5b84e.png)

### Documentation

For more information, view the [documentation online][doc].

  [doc]: http://pythonhosted.org/OrbitalPy/