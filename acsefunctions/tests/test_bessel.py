import pytest
import numpy as np
from scipy.special import gamma as scipy_gamma
from scipy.special import jv as scipy_bessel
from acsefunctions.bessel import factorial, gamma_function_lanczos, bessel_function


class TestFactorial:
    """
    Test suite for the factorial function.
    """

    def test_single_value(self):
        """
        Test factorial function for common positive integers.
        """
        assert factorial(5) == 120
        assert factorial(0) == 1
        assert factorial(1) == 1

    def test_negative_value_error(self):
        """
        Ensure factorial function raises error for negative integers.
        """
        with pytest.raises(ValueError, match="Factorial not defined for negative numbers."):
            factorial(-1)


class TestGammaFunction:
    """
    Test suite for the gamma_function_lanczos.
    """

    def test_gamma(self):
        """
        Test gamma_function_lanczos against scipy's gamma function.

        This test covers:
        - Testing on individual values.
        - Testing on a numpy array of values.
        - Testing on negative integers.
        - Testing on zero (where the gamma function diverges).
        - Testing on a large value for numerical stability.
        """
        # Test on single values
        for z in [0.5, 1, 2, 3.5, 5]:
            assert np.isclose(gamma_function_lanczos(z), scipy_gamma(z), atol=1e-5)

        # Test on a numpy array
        arr = np.array([0.5, 1, 2, 3.5, 5])
        expected_result = scipy_gamma(arr)
        np.testing.assert_allclose(gamma_function_lanczos(arr), expected_result, atol=1e-5)

        negative_integers = np.array([-1, -2, -3, -4, -5])
        expected_results = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        results = gamma_function_lanczos(negative_integers)
        assert np.all(
            results == expected_results
        ), f"Expected {expected_results} but got {results}"

        # Test on zero
        assert np.isclose(gamma_function_lanczos(0), (scipy_gamma(0) * -1))

        # Test on a large value
        assert np.isclose(gamma_function_lanczos(20), scipy_gamma(20), atol=1e-5)


class TestBesselFunction:
    """
    Test suite for the bessel_function.
    """

    def test_bessel_function(self):
        """
        Test bessel_function against scipy's Bessel function for real numbers.
        """
        alphas = [0, 1, 0.5, 2, 10]
        xs = [1, 1, 0.5, 10, 2]

        for alpha, x in zip(alphas, xs):
            result = bessel_function(alpha, x)
            expected = scipy_bessel(alpha, x)
            np.testing.assert_allclose(result, expected)

    def test_bessel_function_complex_numbers(self):
        """
        Test bessel_function against scipy's Bessel function for complex numbers.
        """
        alphas = [1, 0.5, 10, 50]
        xs = [1 + 1j, 1 - 1j, 0.5, 10, 2 - 2j]

        for alpha, x in zip(alphas, xs):
            result = bessel_function(alpha, x)
            expected = scipy_bessel(alpha, x)
            np.testing.assert_allclose(result, expected)

    def test_bessel_function_special_values(self):
        """
        Test bessel_function for known special values.
        """
        assert np.isclose(bessel_function(0, 0), 1)
        assert np.isclose(bessel_function(1, 0), 0)

    def test_bessel_function_symmetry(self):
        """
        Test the symmetry property Jn(-x) = (-1)^n Jn(x) for the bessel_function.
        """
        alpha = np.random.randint(0, 11)
        x = np.random.rand()

        result = bessel_function(alpha, -x)
        expected = (-1) ** alpha * bessel_function(alpha, x)
        np.testing.assert_allclose(result, expected)