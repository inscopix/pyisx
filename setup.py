from setuptools import setup, find_namespace_packages

setup(
    name='isx',
    version='2.0.0',
    packages=find_namespace_packages(),
    python_requires='>=3.9,<3.13',
    install_requires=[
        'h5py>=2.8.0',
        'numpy>=1.14',
        'scipy>=1.0',
        'tifffile>=0.15.1',
        'pandas>=0.20.1',
        'pillow>=8.0.1',
        'openpyxl>=3.0.10', # Required for pandas Excel support
    ],
    extras_require={
        'test': [
            # Optional dependencies for testing
            'pytest',
        ],
        'docs': [
            # Optional dependencies for testing
            'sphinx',
            'sphinx_rtd_theme',
            'myst_parser'
        ]
    },
    include_package_data=True,
    description="A python package for interacting with Inscopix data.",
    url="https://github.com/inscopix/pyisx",
    author="Inscopix, Inc.",
    has_ext_modules=lambda: True
)
