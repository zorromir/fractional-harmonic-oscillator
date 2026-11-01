import numpy as np
from numpy.testing import assert_allclose
from fractional_oscillator import mittag_leffler


def test_mittag_leffler_recovers_exponential():
    z=np.array([-2.,-.5,0.,1.])
    assert_allclose(mittag_leffler(z,1),np.exp(z),atol=2e-13,rtol=1e-12)


def test_mittag_leffler_recovers_cosine():
    t=np.linspace(0,8,17)
    assert_allclose(mittag_leffler(-t*t,2),np.cos(t),atol=3e-12)


def test_scalar_shape_and_errors():
    assert isinstance(mittag_leffler(0.0,1.5),float)
    import pytest
    with pytest.raises(ValueError):mittag_leffler(0,0)
