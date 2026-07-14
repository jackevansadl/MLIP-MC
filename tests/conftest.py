"""
Pytest configuration and shared fixtures.
"""
import os
import sys

import numpy as np
import pytest

# Tests use the mlip_mc package

# Block gpaw from being imported during tests. When GPAW is installed,
# merely importing ``ase.io`` (which every test module does through
# mlip_mc) imports gpaw via ASE's external IO-format entry points, and
# gpaw's ``__init__`` detects pytest and switches on its debug mode:
# ``np.seterr(over='raise', divide='raise', invalid='raise')`` plus
# NaN-scribbling wrappers around ``np.empty``/``np.empty_like``. Those
# global side effects change this suite's numerics (overflow in np.exp
# raises instead of giving inf) and slow it down enormously. Nothing in
# this suite uses gpaw; ASE catches the failed entry-point import and
# just skips registering the gpaw-yaml format.
sys.modules.setdefault('gpaw', None)


@pytest.fixture(autouse=True, scope='session')
def _numpy_default_errstate():
    """Belt-and-braces: restore numpy's default floating-point error
    handling in case gpaw (or another package) was imported before the
    blocker above took effect and called ``np.seterr`` globally."""
    np.seterr(over='warn', divide='warn', invalid='warn', under='ignore')
    yield
