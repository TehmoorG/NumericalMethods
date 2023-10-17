import math
import numpy as np

def exp(x, N=20):
    result = 1.0
    factorial = 1.0
    power_of_x = 1.0

    for n in range(1, N+1):
        power_of_x *= x
        factorial *= n
        result += power_of_x / factorial

    return result

print(exp(np.array([1,2,3])))

def sin(x, N=20):
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
    return sin(x, N) / cos(x, N)