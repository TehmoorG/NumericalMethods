"""
Special Mathematical Functions (acsefunctions.bessel)

This module provides functions for computing various special mathematical functions
including factorials, gamma functions, and Bessel functions of the first kind. These
functions are often used in scientific computing, physics, and engineering applications.

Functions:
- factorial(n): Compute the factorial of an integer or array of integers.
- gamma_function_lanczos(z): Compute the gamma function using the Lanczos approximation.
- bessel_function(alpha, x, terms=100): Compute the Bessel function of the first kind.

The module is designed to be used with numpy arrays for efficient computation, especially
for vectorized operations over arrays of numbers.

Notes
- The functions are implemented with numerical stability and efficiency in mind.
- The Bessel function implementation is based on its series representation and is
  most accurate for small orders and arguments.
"""
import numpy as np


def factorial(n):
    """
    Compute the factorial of n.

    Parameters
    ----------
    n : int or np.ndarray of int
        An integer or an array of integers for which
        the factorial is to be computed.

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
    if isinstance(n, list):
        n = np.array(n)

    if isinstance(n, np.ndarray):
        vectorized_factorial = np.vectorize(_single_factorial)
        return vectorized_factorial(n)
    else:
        return _single_factorial(n)


def _single_factorial(value):
    """
    Help to compute factorial for a single integer.

    Parameters
    ----------
    value : int
        An integer for which the factorial is to be computed.

    Returns
    -------
    int
        Factorial of 'value'.

    Raises
    ------
    ValueError
        If 'value' is negative.

    Examples
    --------
    >>> _single_factorial(5)
    120

    >>> _single_factorial(0)
    1

    >>> _single_factorial(1)
    1

    Handling negative input by raising ValueError:
    >>> _single_factorial(-1)
    Traceback (most recent call last):
    ...
    ValueError: Factorial not defined for negative numbers.
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
    array([11.6317284])

    >>> gamma_function_lanczos(np.array([4, 5, 6]))
    array([  6.,  24., 120.])

    >>> gamma_function_lanczos(0.5 + 1j)
    array([0.30069462-0.42496788j])

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
    Compute the Bessel function of the first kind.

    Parameters
    ----------
    alpha : float
        The order of the Bessel function.

    x : float or np.ndarray
        The value or array of values at which to
        evaluate the Bessel function.

    terms : int, optional
        The number of terms to use in the series expansion.
        Default is 100.

    Returns
    -------
    float or np.complex128
        The approximated value of the Bessel function at x.

    Examples
    --------
    >>> bessel_function(1, 2)
    array([0.57672481])

    >>> bessel_function(0, 0.5)
    array([0.93846981])

    Notes
    -----
    The function employs the series representation of
    the Bessel function of the first kind.
    If the computation involves complex numbers, the
    results are returned in `np.complex128` format.
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
