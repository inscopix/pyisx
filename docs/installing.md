
# Installation

## For Users

### Install latest version of code 

If you plan to use this in your own project, use your 
favorite package manager to install this in your project.



=== "pip"

    ```bash
    # you are strongly encouraged to install in a virtual envrionment
    pip install isx
    ```

=== "poetry"


    ```bash
    poetry add isx
    ```



## For Developers

!!! danger "Developers only"
    The instructions below are only if you intend to develop `py_isx`. If you merely want to use this API to read data, you do not have to do this.

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
# GNU Make 3.81

poetry --version
# Poetry (version 1.7.1)
```

### Install locally

You can then install the API locally by navigating to the directory you downloaded the code in, and running:

```bash
poetry lock
poetry install --all-extras
```

Poetry installs using a "editable" install, so changes you make will be reflected in the code executed. 



