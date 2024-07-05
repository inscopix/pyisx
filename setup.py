from setuptools import setup, find_packages, Extension
import sys

# sys.argv.extend(['--plat-name', 'x86_64'])
setup(
    name='isx',
    version='1.9.2',
    packages=find_packages(include=['isx','isx.*']),
    python_requires='>= 3.7,< 3.11',
    install_requires=[
        'h5py>=2.8.0',
        'numpy>=1.14',
        'scipy>=1.0',
        'tifffile>=0.15.1',
        'pandas>=0.20.1',
        'pillow>=8.0.1',
        'openpyxl>=3.0.10', # Required for pandas Excel support
    ],
    include_package_data=True,
    description="Inscopix Data Processing Software Python API",
    url="https://www.inscopix.com/support",
    author="Inscopix, Inc.",
    author_email="support@inscopix.com",
    ext_modules=[
        Extension(
            name='isxcore',
            sources=[]
        )
    ]
)
