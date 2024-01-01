import numpy as np
from acsefunctions.taylor import sin, cos, tan, exp


class TestSin:
    """
    Test cases for the `sin` function.

    This class contains unit tests that validate the behavior of the
    `sin` function for specific angle values and convergence behavior.
    """

    def test_zero(self):
        assert np.allclose(sin(0), np.array([0.0]))

    def test_pi_over_two(self):
        assert np.allclose(sin(np.pi / 2), np.array([1.0]))

    def test_pi(self):
        assert np.allclose(sin(np.pi), np.array([0.0]))

    def test_pi_over_four(self):
        assert np.allclose(sin(np.pi / 4), np.array([np.sqrt(2) / 2]))

    def test_negative_pi(self):
        assert np.allclose(sin(-np.pi), np.array([0.0]))

    def test_three_pi_over_two(self):
        assert np.allclose(sin(3 * np.pi / 2), np.array([-1.0]))

    def test_convergence_with_high_N(self):
        """
        Test to see if a high value of N converges by reducing tolerance.
        """
        for tol in [1e-5, 1e-8, 1e-10]:
            assert np.isclose(sin(np.pi / 6, 1000), np.array([0.5]), atol=tol)

    def test_sin_empty_array(self):
        assert sin(np.array([])).size == 0

    def test_sin_symmetry(self):
        x = np.random.rand(10) * 2 * np.pi  # Random values in [0, 2π]
        assert np.allclose(sin(x), -sin(-x))  # sin should be an odd function



class TestCos:
    """
    Test cases for the `cos` function.

    This class contains test cases for various inputs to the `cos` function.
    It uses NumPy's testing functions to check if the results are close to
    the expected values within a specified tolerance.
    """

    def test_zero(self):
        assert np.allclose(cos(0), np.array([1.0]))

    def test_pi_over_two(self):
        assert np.allclose(cos(np.pi / 2), np.array([0.0]))

    def test_pi(self):
        assert np.allclose(cos(np.pi), np.array([-1.0]))

    def test_pi_over_four(self):
        assert np.allclose(cos(np.pi / 4), np.array([np.sqrt(2) / 2]))

    def test_negative_pi(self):
        assert np.allclose(cos(-np.pi), np.array([-1.0]))

    def test_three_pi_over_two(self):
        assert np.allclose(cos(3 * np.pi / 2), np.array([0.0]))

    def test_convergence_with_high_N(self):
        """
        Test to see if a high value of N converges by reducing tolerance.
        """
        for tol in [1e-5, 1e-8, 1e-10]:
            assert np.isclose(
                cos(np.pi / 6, 1000), np.array([np.sqrt(3) / 2]), atol=tol
            )

    def test_cos_periodicity(self):
        period = 2 * np.pi
        x = np.random.rand(10) * 2 * np.pi  # Random values in [0, 2π]
        assert np.allclose(cos(x), cos(x + period))  # cos should be periodic with 2π


class TestTan:
    """
    Test cases for the `tan` function.

    This class contains test cases for various inputs to the `tan` function.
    It uses NumPy's testing functions to check if the results are close to
    the expected values within a specified tolerance.
    """

    def test_zero(self):
        assert np.allclose(tan(0), np.array([0.0]))

    def test_pi_over_four(self):
        assert np.allclose(tan(np.pi / 4), np.array([1.0]))

    def test_five_pi_over_four(self):
        assert np.allclose(tan(5 * np.pi / 4), np.array([1.0]))

    def test_three_pi(self):
        assert np.allclose(tan(3 * np.pi), np.array([0.0]))

    def test_convergence_with_high_N(self):
        """
        Test to see if a high value of N converges by reducing tolerance.
        """
        for tol in [1e-5, 1e-8, 1e-10]:
            assert np.isclose(
                tan(np.pi / 6, 1000), np.array([np.sqrt(3) / 3]), atol=tol
            )

    def test_tan_division_by_near_zero(self):
        # cos(π/2) is zero, so tan should be undefined or very large
        assert np.isnan(tan(np.pi / 2))


class TestExp:
    """
    Test cases for the `exp` function.

    This class contains test cases for the `exp` function using various inputs.
    It checks if the results are close to the expected values
    within a specified tolerance.
    """

    def test_one(self):
        assert np.allclose(exp(1), np.array([np.exp(1)]))

    def test_zero(self):
        assert np.allclose(exp(0), np.array([np.exp(0)]))

    def test_array(self):
        assert np.allclose(exp([-1, 0, 1]), np.exp(np.array([-1, 0, 1])))

    def test_negative_decimal(self):
        assert np.allclose(exp([-1, 0.5]), np.exp(np.array([-1, 0.5])))

    def test_convergence_with_high_N(self):
        """
        Test to see if a high value of N converges by reducing tolerance.
        """
        for tol in [1e-5, 1e-8, 1e-10]:
            assert np.isclose(exp([3.7]), np.array([np.exp(3.7)]), atol=tol)

    def test_exp_array_input(self):
        x = np.array([0, 1, 2])
        expected = np.exp(x)
        assert np.allclose(exp(x), expected)

    
    def test_exp_2d_array_input(self):
        # Testing with a 2D array input
        x = np.array([[0, 1], [2, 3]])
        expected = np.exp(x)
        assert np.allclose(exp(x), expected), "exp function should handle 2D arrays."

    def test_exp_higher_dim_array_input(self):
        # Testing with a 3D array input
        x = np.random.rand(2, 2, 2)  # 3D array with random numbers
        expected = np.exp(x)
        assert np.allclose(exp(x), expected), "exp function should handle 3D arrays."

    def test_exp_varying_shape_array_input(self):
        # Testing with varying shape arrays
        for shape in [(5,), (3, 3), (2, 2, 2), (1, 4, 2)]:
            x = np.random.rand(*shape)
            expected = np.exp(x)
            assert np.allclose(exp(x), expected), f"exp function should handle array of shape {shape}."