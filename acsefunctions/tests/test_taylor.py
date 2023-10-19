import numpy as np
from acsefunctions.taylor import sin, cos, tan, exp


class TestSin:
    def test_zero(self):
        assert np.allclose(sin(0), np.array([0.0]))

    def test_pi_over_two(self):
        assert np.allclose(sin(np.pi / 2), np.array([1.0]))

    def test_pi(self):
        assert np.allclose(sin(np.pi), np.array([0.0]))


class TestCos:
    def test_zero(self):
        assert np.allclose(cos(0), np.array([1.0]))

    def test_pi_over_two(self):
        assert np.allclose(cos(np.pi / 2), np.array([0.0]))

    def test_pi(self):
        assert np.allclose(cos(np.pi), np.array([-1.0]))


class TestTan:
    def test_zero(self):
        assert np.allclose(tan(0), np.array([0.0]))

    def test_pi_over_four(self):
        assert np.allclose(tan(np.pi / 4), np.array([1.0]))


class TestExp:
    def test_one(self):
        assert np.allclose(exp(1), np.array([np.exp(1)]))

    def test_zero(self):
        assert np.allclose(exp(0), np.array([np.exp(0)]))

    def test_array(self):
        assert np.allclose(exp([-1, 0, 1]), np.exp(np.array([-1, 0, 1])))
