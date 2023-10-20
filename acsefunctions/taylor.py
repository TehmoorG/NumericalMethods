import numpy as np


def exp(x, N=200):
    """
    Approximate the exponential function e^x using Taylor series.

    Parameters
    ----------
    x : float or numpy.ndarray
        The value (or array of values) at which to evaluate
        the exponential function.
    N : int, optional
        The number of terms in the Taylor series expansion.
        Default is 200.

    Returns
    -------
    numpy.ndarray
        The approximated value (or array of values) of e^x.

    Examples
    --------
    >>> exp(1)
    ...

    >>> exp(2, N=10)
    ...

    >>> exp(np.array([0, 1]))
    ...
    """
    x = np.asarray(x)

    result = np.ones_like(x, dtype=float)
    factorial = 1.0
    power_of_x = np.ones_like(x, dtype=float)

    for n in range(1, N + 1):
        power_of_x *= x
        factorial *= n
        result += power_of_x / factorial

    return result


def sin(x, N=20):
    """
    Approximate the sine function sin(x) using Taylor series.

    Parameters
    ----------
    x : float or numpy.ndarray
        The value (or array of values) in radians at which to
        evaluate the sine function.
    N : int, optional
        The number of terms in the Taylor series expansion. Default is 20.

    Returns
    -------
    numpy.ndarray
        The approximated value (or array of values) of sin(x).

    Examples
    --------
    >>> sin(0)
    array(0.)

    >>> sin(np.array([0, np.pi/2, np.pi]))
    array([ 0.        ,  1.        ,  1.2246468e-16])
    """
    x = np.asarray(x, dtype=float)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    result = x.copy()
    factorial = 1.0
    power_of_x = x.copy()
    sign = -1.0

    for n in range(3, 2 * N + 1, 2):
        power_of_x *= x * x  # raises x to the next odd power
        factorial *= n * (n - 1)
        term = sign * power_of_x / factorial
        result += term
        sign *= -1
    return result


def cos(x, N=20):
    """
    Approximate the cosine function cos(x) using Taylor series.

    Parameters
    ----------
    x : float or numpy.ndarray
        The value (or array of values) in radians at which
        to evaluate the cosine function.
    N : int, optional
        The number of terms in the Taylor series expansion. Default is 20.

    Returns
    -------
    numpy.ndarray
        The approximated value (or array of values) of cos(x).

    Examples
    --------
    >>> cos(0)
    array(1.)

    >>> cos(np.array([0, np.pi/2, np.pi]))
    array([ 1.        ,  6.123234e-17, -1.        ])
    """
    x = np.asarray(x, dtype=float)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    result = np.ones_like(x, dtype=float)
    factorial = 1.0
    power_of_x = np.ones_like(x, dtype=float)
    sign = -1.0

    for n in range(2, 2 * N + 1, 2):
        power_of_x *= x * x
        factorial *= n * (n - 1)
        term = sign * power_of_x / factorial
        result += term
        sign *= -1

    return result


def tan(x, N=20):
    """
    Approximate the tangent function tan(x) using Taylor series.

    Parameters
    ----------
    x : float or numpy.ndarray
        The value (or array of values) in radians at which
        to evaluate the tangent function.
    N : int, optional
        The number of terms in the Taylor series expansion for
        sin(x) and cos(x). Default is 20.

    Returns
    -------
    numpy.ndarray
        The approximated value (or array of values) of tan(x).

    Examples
    --------
    >>> tan(0)
    array(0.)

    >>> tan(np.array([0, np.pi/4]))
    array([0., 1.])

    Notes
    -----
    The function computes tan(x) by dividing the Taylor series
    approximations of sin(x) and cos(x).
    This may lead to inaccuracies or errors when cos(x) is close to zero.
    """
    x = np.array(x, dtype=float)

    s = sin(x, N)
    c = cos(x, N)

    c = np.where(np.abs(c) < 1e-10, np.nan, c)

    return s / c
