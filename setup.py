import os
from setuptools import setup
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

pkgdir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="corrcal",
    version="0.0.2",
    license='BSD 2-Clause License',
    author="CorrCal Development Team",
    description="Correlation Calibration for Quasi-redundant radio interferometer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UKZN-Astronomy/corrcal",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 2-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'cffi>=1.0'
    ],
    setup_requires=['cffi>=1.0'],
    cffi_modules=["{pkgdir}/build_cffi.py:ffi".format(pkgdir=pkgdir)]
)
