import pytest
import numpy as np
from scipy.special import gamma as scipy_gamma
from acsefunctions.bessel import factorial, gamma_function_lanczos, bessel_function

def test_single_value():
    assert factorial(5) == 120
    assert factorial(0) == 1
    assert factorial(1) == 1


def test_negative_value_error():
    with pytest.raises(ValueError, match="Factorial not defined for negative numbers."):
        factorial(-1)




def test_gamma():
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
    assert np.all(results == expected_results), f"Expected {expected_results} but got {results}"
    
    # Test on a non-negative integer value (where gamma function diverges)
    # Scipys gamma function should produce a value of inf whereas the lanczos will produce -inf
    assert np.isclose(gamma_function_lanczos(0), (scipy_gamma(0) * -1))

    # Test on a large value to ensure numerical stability
    assert np.isclose(gamma_function_lanczos(20), scipy_gamma(20), atol=1e-5)
