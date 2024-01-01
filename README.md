# ACSE Functions Library

This library provides a collection of mathematical functions, including Taylor series approximations for trigonometric and exponential functions.

## Installation

Ensure you have Python 3.x installed. Clone the repository and navigate to the project directory:

```bash
git clone [your-repository-link]
cd mpm-assessment-2-acse-tg1523
```
### Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Features

- **Taylor Series Approximations:** Provides approximations for functions like `sin`, `cos`, `tan`, and `exp`.
- **Bessel Functions:** Efficient computation of Bessel functions.
- [Add other features or modules here.]

## Usage

Here's a basic example of how to use the Taylor series approximation for the `exp` function:

```python
from acsefunctions.taylor import exp
result = exp(5, N=20)
print(result)
```
## Building Documentation
## Building Documentation

To build the Sphinx documentation locally, you'll need Sphinx installed. This is given in `requirements.txt`:
Then you can build the documentation using:
```bash
make html
```
After running the command, a build/ directory will be created with the compiled HTML files for the documentation. This directory is ignored in the repository to keep the build artifacts local to your machine. The built documentation will be available under build/html. Open the index.html file in a web browser to view it.

## Accuracy Analysis

We've conducted accuracy analysis for our Taylor series approximations, comparing them against established libraries like NumPy. Detailed analysis can be found in the [Accuracy Analysis](#accuracy-analysis) section.

## Testing

Tests are available for all the major functions. Execute the tests using:

```bash
pytest .
```