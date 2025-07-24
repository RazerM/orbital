# Orbital

<p>
  <a href="https://pypi.org/project/orbitalpy/"><img src="https://img.shields.io/pypi/v/orbitalpy.svg" alt="PyPI" /></a>
  <a href="https://raw.githubusercontent.com/RazerM/orbital/master/LICENSE"><img src="https://img.shields.io/pypi/l/orbitalpy.svg" alt="MIT License" /></a>
  <a href="https://raw.githubusercontent.com/RazerM/orbital/master/LICENSE"><img src="https://img.shields.io/pypi/pyversions/orbitalpy.svg" alt="Python Versions" /></a>
  <a href="https://github.com/RazerM/orbital/actions"><img src="https://github.com/RazerM/orbital/actions/workflows/ci.yml/badge.svg" alt="GitHub Actions Status" /></a>
  <a href="http://orbitalpy.readthedocs.org/en/latest/"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation" /></a>
</p>

Orbital is a high level orbital mechanics package for Python.

### Installation

```bash
pip install orbitalpy
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

[doc]: http://orbitalpy.readthedocs.org/en/latest/
