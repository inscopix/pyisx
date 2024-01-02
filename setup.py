from setuptools import setup

# Only for use with github actions
setup(
    name="isx",
    version="2023.11.21",
    description="Python-based ISXD file reader",
    author="Srinivas Gorur-Shandilya",
    author_email="code@srinivas.gs",
    url="",
    packages=[""],
    install_requires=[
        "beartype>=0.15.0",
        "numpy>=1.26.2",
    ],
    extras_require={
        "test": ["pytest>=7.2.0", "coverage>=7.3.2"],
        "ideas": ["ideas-data @ git+https://github.com/inscopix/ideas-data@2.0.0"],
        "dev": ["ipykernel>=6.20.1", "debugpy==1.6"],
    },
    python_requires=">=3.9,<4.0",
)
