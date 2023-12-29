from setuptools import setup
from Cython.Build import cythonize

setup(
    setup_requires=["numpy", "Cython"],
    ext_modules = cythonize("cyfunc.pyx")
)
