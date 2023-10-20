import numpy as np
import functools

@functools.lru_cache(maxsize=None)  # Unbounded cache
def factorial(n):
    """
    Compute the factorial of n.
    
    Parameters:
    -----------
    n : int or np.ndarray
        An integer or an array of integers for which factorial is to be computed.
        
    Returns:
    --------
    int or np.ndarray
        Factorial of n or an array of factorials.
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





# def gamma_function(z, num_terms=1000000):
    
#     # Ensure z is a numpy array
#     z = np.atleast_1d(z)

#     results = np.empty_like(z, dtype=np.complex128)

#     for i, z_value in enumerate(z):
#         product = 1.0
#         for n in range(1, num_terms+1):
#             factor = (1 + 1/n)**z_value * (1 + z_value/n)**-1
#             product *= factor
#         results[i] = 1/z_value * product

#     # Check if all values are real and if so, return a real array
#     if np.all(np.isreal(results)):
#         return results.real
#     else:
#         return results

    
def bessel_function(alpha, x, terms=100):
    result = np.complex128(0.0)  # initialize as complex128
    for m in range(terms):
        term = (((-1)**m) / (np.complex128(factorial(m)) * gamma_function_euler(m+alpha+1))) * ((x / 2)**(2*m + alpha))
        result += term
    return result

def gamma_function_lanczos(z):
    # Lanczos coefficients for g=5
    p = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]

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

print(gamma_function_lanczos(-1))