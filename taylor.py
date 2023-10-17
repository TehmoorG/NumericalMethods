def exp(x, N=20):
    """
    Approximate the exponential function e^x using Taylor series.

    Parameters
    ----------
    x : float
        The value at which to evaluate the exponential function.
    N : int, optional
        The number of terms in the Taylor series expansion. Default is 20.

    Returns
    -------
    float
        The approximated value of e^x.

    Examples
    --------
    >>> exp(1)
    2.7182818284590455

    >>> exp(2, 10)
    7.3887125220458545
    """
    result = 1.0
    factorial = 1.0
    power_of_x = 1.0

    for n in range(1, N+1):
        power_of_x *= x
        factorial *= n
        result += power_of_x / factorial

    return result

def sin(x, N=20):
    """
    Approximate the sine function sin(x) using Taylor series.

    Parameters
    ----------
    x : float
        The value in radians at which to evaluate the sine function.
    N : int, optional
        The number of terms in the Taylor series expansion. Default is 20.

    Returns
    -------
    float
        The approximated value of sin(x).

    Examples
    --------
    >>> sin(0)
    0.0

    >>> sin(math.pi / 2)
    1.0

    >>> sin(math.pi, 10)
    1.2246467991473532e-16
    """
    result = x
    factorial = 1
    power_of_x = x
    sign = -1

    for n in range(3, 2*N+1, 2):
        power_of_x *= x*x
        factorial *= n*(n-1)
        result += sign * power_of_x / factorial
        sign *= -1

    return result

def cos(x, N=20):
    """
    Approximate the cosine function cos(x) using Taylor series.

    Parameters
    ----------
    x : float
        The value in radians at which to evaluate the cosine function.
    N : int, optional
        The number of terms in the Taylor series expansion. Default is 20.

    Returns
    -------
    float
        The approximated value of cos(x).

    Examples
    --------
    >>> cos(0)
    1.0

    >>> cos(math.pi / 2)
    6.123233995736766e-17

    >>> cos(math.pi, 10)
    -1.0
    """
    result = 1.0
    factorial = 1
    power_of_x = 1.0
    sign = -1

    for n in range(2, 2*N+1, 2):
        power_of_x *= x*x
        factorial *= n*(n-1)
        result += sign * power_of_x / factorial
        sign *= -1

    return result

def tan(x, N=20):
    """
    Approximate the tangent function tan(x) using Taylor series.

    Parameters
    ----------
    x : float
        The value in radians at which to evaluate the tangent function.
    N : int, optional
        The number of terms in the Taylor series expansion for sin(x) and cos(x). Default is 20.

    Returns
    -------
    float
        The approximated value of tan(x).

    Examples
    --------
    >>> tan(0)
    0.0

    >>> tan(math.pi / 4)
    1.0

    >>> tan(math.pi / 2)  # This might result in a very large value or error due to cos(pi/2) being close to 0
    ...

    Notes
    -----
    The function computes tan(x) by dividing the Taylor series approximations of sin(x) and cos(x). 
    This may lead to inaccuracies or errors when cos(x) is close to zero.
    """
    s = sin(x, N)
    c = cos(x, N)
    
    if abs(c) < 1e-10:
        raise ValueError("tan(x) is undefined or too large for x close to pi/2 + k*pi.")

    return s / c