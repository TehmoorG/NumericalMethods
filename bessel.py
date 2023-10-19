import numpy as np
from scipy.integrate import quad

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
        result = np.ones_like(n, dtype=np.int64)
        for idx, value in np.ndenumerate(n):
            if value < 0:
                raise ValueError("Factorial not defined for negative numbers.")
            for i in range(1, value+1):
                result[idx] *= i
        return result

    else:
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers.")
        result = 1
        for i in range(1, n+1):
            result *= i
        return result




def gamma_function_euler(z, num_terms=1000):
    
    # Ensure z is a numpy array
    z = np.atleast_1d(z)

    results = np.empty_like(z, dtype=np.complex128)

    for i, z_value in enumerate(z):
        product = 1.0
        for n in range(1, num_terms+1):
            factor = (1 + 1/n)**z_value * (1 + z_value/n)**-1
            product *= factor
        results[i] = 1/z_value * product

    return results

# Example usage:
print(gamma_function_euler(0.5))  # For a scalar input
print(gamma_function_euler(np.array([0.5, 1.5])))  # For an array input



    
def bessel_function(alpha, x, terms=100):
    result = 0.0
    for m in range(terms):
        term = (((-1)**m) / ((factorial(m)) * gamma_euler_trapezoidal(m+alpha+1))) * ((x / 2)**(2*m + alpha))
        result += term






