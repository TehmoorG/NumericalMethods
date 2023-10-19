try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="acsefunctions",
    # Automatically set package version using git metadata
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Package for trigonometric functions etc.",
    author="Tehmoor",
    author_email="tehmoor.gull@imperial.ac.uk",
    packages=["acsefunctions"],
)
