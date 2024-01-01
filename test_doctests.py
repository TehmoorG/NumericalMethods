import doctest
import numpy as np
from acsefunctions import taylor, bessel

def run_doctests():
    doctest.testmod(taylor, verbose=True)  # Replace 'taylor' with the actual module
    doctest.testmod(bessel, verbose=True)  # Replace 'bessel' with the actual module

if __name__ == "__main__":
    run_doctests()
