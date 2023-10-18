import numpy as np
from acsefunctions.taylor import sin, cos, tan, exp


def test_sin():
    assert np.allclose(sin(0), np.array([0.0]))
    assert np.allclose(sin(np.pi / 2), np.array([1.0]))
    assert np.allclose(sin(np.pi), np.array([0.0]))


def test_cos():
    assert np.allclose(cos(0), np.array([1.0]))
    assert np.allclose(cos(np.pi / 2), np.array([0.0]))
    assert np.allclose(cos(np.pi), np.array([-1.0]))


def test_tan():
    assert np.allclose(tan(0), np.array([0.0]))
    assert np.allclose(tan(np.pi / 4), np.array([1.0]))


def test_exp():
    assert np.allclose(exp(1), np.array([np.exp(1)]))
    assert np.allclose(exp(0), np.array([np.exp(0)]))
    assert np.allclose(exp([-1, 0, 1]), np.exp(np.array([-1, 0, 1])))
