import numpy as np
import functools


@functools.lru_cache(maxsize=None)  # Unbounded cache
def factorial(n):
    """
    Compute the factorial of n.

    Parameters
    ----------
    n : int or np.ndarray of int
        An integer or an array of integers for which the factorial is to be computed.

    Returns
    -------
    int or np.ndarray
        Factorial of n or an array of factorials.

    Examples
    --------
    >>> factorial(5)
    120

    >>> factorial(np.array([3, 4, 5]))
    array([  6,  24, 120])
    """
    if isinstance(n, np.ndarray):
        vectorized_factorial = np.vectorize(_single_factorial)
        return vectorized_factorial(n)
    else:
        return _single_factorial(n)


@functools.lru_cache(maxsize=None)  # Unbounded cache
def _single_factorial(value):
    """
    Helper function to compute factorial for a single integer.
    """
    if value < 0:
        raise ValueError("Factorial not defined for negative numbers.")
    if value == 0:
        return 1
    return value * _single_factorial(value - 1)


def gamma_function_lanczos(z):
    """
    Compute the gamma function using the Lanczos approximation.

    Parameters
    ----------
    z : float, complex, or np.ndarray
        The value or array of values at which to evaluate the gamma function.

    Returns
    -------
    float, complex or np.ndarray
        Approximated value(s) of the gamma function at z.

    Examples
    --------
    >>> gamma_function_lanczos(4.5)
    11.631728396567448

    >>> gamma_function_lanczos(np.array([4, 5, 6]))
    array([ 6., 24., 120.])

    >>> gamma_function_lanczos(0.5 + 1j)
    array([0.52848222+0.88001155j])

    Notes
    -----
    The function employs the Lanczos approximation with g=5 coefficients.
    For negative real values that are whole numbers, the function returns -inf.
    """
    # Lanczos coefficients for g=5
    p = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]

    z = np.atleast_1d(z)
    results = np.empty_like(z, dtype=np.complex128)

    for i, z_value in enumerate(z):
        if z_value.real <= 0 and z_value.imag == 0 and z_value.real % 1 == 0:
            results[i] = -np.inf
            continue

        y = z_value
        x = z_value
        tmp = x + 5.5
        tmp -= (x + 0.5) * np.log(tmp)
        ser = 1.000000000190015
        for j in range(6):
            y += 1
            ser += p[j] / y
        results[i] = np.exp(-tmp) * np.sqrt(2 * np.pi) * ser / x

    # Check if all values are real and if so, return a real array
    if np.all(np.isreal(results)):
        return results.real
    else:
        return results


def bessel_function(alpha, x, terms=100):
    """
    Compute the Bessel function of the first kind using its series representation.

    Parameters
    ----------
    alpha : float
        The order of the Bessel function.

    x : float or np.ndarray
        The value or array of values at which to evaluate the Bessel function.

    terms : int, optional
        The number of terms to use in the series expansion. Default is 100.

    Returns
    -------
    float or np.complex128
        The approximated value of the Bessel function at x.

    Examples
    --------
    >>> bessel_function(1, 2)
    0.5767248077568734

    >>> bessel_function(0, 0.5)
    0.9384698072408128

    Notes
    -----
    The function employs the series representation of the Bessel function of the first kind.
    If the computation involves complex numbers, the results are returned in `np.complex128` format.
    If all values are real, the result is returned as a float.
    """
    result = np.complex128(0.0)  # initialize as complex128
    for m in range(terms):
        term = (
            ((-1) ** m)
            / (np.complex128(factorial(m)) * gamma_function_lanczos(m + alpha + 1))
        ) * ((x / 2) ** (2 * m + alpha))
        result += term
        # Check if all values are real and if so, return a real array
    if np.all(np.isreal(result)):
        return result.real
    else:
        return result
