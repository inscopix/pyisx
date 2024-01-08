
# Installation

## For Users

### Install latest version of code 

If you plan to use this in your own project, use your 
favorite package manager to install this in your project.

#### using [pip](https://pypi.org/project/pip/)

```bash
pip install git+https://github.com/inscopix/py_isx.git@main
```

#### using [poetry](https://python-poetry.org/):


```bash
poetry add git+ssh://git@github.com/inscopix/py_isx.git
```

Please refer to [Poetry's documentation](https://python-poetry.org/docs/cli/#add) for installing from
git repositories for further details. 

### Installing from a wheel

We also provide wheel files for installation locally. 

## For Developers

### Get the code

```bash
git clone git@github.com:inscopix/py_isx.git
```

### Prerequisites 

Make sure you have [poetry](https://python-poetry.org/) and
[make](https://www.gnu.org/software/make//) installed. 
Verify that both are installed:

```bash
make --version
GNU Make 3.81
Copyright (C) 2006  Free Software Foundation, Inc.
This is free software; see the source for copying conditions.
There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

This program built for i386-apple-darwin11.3.0
```

```bash
poetry --version
Poetry (version 1.7.1)
```

### Install locally

You can then install the API locally by navigating to the directory you downloaded the code in, and running:

```bash
poetry lock
poetry install --all-extras
```

Poetry installs using a "editable" install, so changes you make will be reflected in the code executed. 



