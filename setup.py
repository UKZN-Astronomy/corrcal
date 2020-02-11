from setuptools import setup, find_packages, Extension


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="corrcal",
    author="CorrCal Development Team",
    description="Correlation Calibration for Quasi-redundant radio interferometer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UKZN-Astronomy/corrcal",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 2-Clause License",
        "Operating System :: OS Independent",
    ],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    python_requires='>=3.7',
    install_requires=['numpy'],
    # Set main package directory to `src`
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        # Include any *.dat files found in the 'data' subdirectory
        # of the main package directory:
        '': ['data/*.dat'],
    },
    ext_modules=[
        Extension(
            'corrcal.c_corrcal',
            sources=['src/corrcal/src/corrcal_c_funcs.c'],
            extra_compile_args=['-std=c99', '-fopenmp'],
            include_dirs=['src/corrcal/src'],
        )
    ]
)
