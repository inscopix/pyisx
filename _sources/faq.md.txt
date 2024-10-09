# FAQ

* What's the difference between the [IDPS Python API](https://inscopix.com/software-analysis-miniscope-imaging/) and `pyisx`?
    * The IDPS (Inscopix Data Processing Software) Python API is a python package included with the IDPS GUI application. `pyisx` is an open-source version of this package, which contains the exact same functionality (excluding any algorithms), and can be installed without installing the entire IDPS application.
* Does this package include algorithms from the IDPS Python API?
    * No, currently the algorithms from the IDPS Python API are not included in `pyisx`
* What's the difference between `pyisx` and `isx`?
    * [pyisx](https://github.com/inscopix/pyisx) project is a python binding for [isxcore](https://github.com/inscopix/isxcore), a C++ API for interacting with Inscopix data.
    * `isx` is the python package for this project, with pre-built binaries available on [pypi](https://pypi.org/project/isx/).
* How do I file a bug report or a feature request?
    * If you have found a bug, we recommend searching the [issues page](https://github.com/inscopix/pyisx/issues) to see if it has already been reported. If not, please open a new issue.
    * If you have a feature request, please open a new issue with the label `enhancement`.
* Can I use this in my own projects?
    * Absolutely! `pyisx` is an open-source project and free to use. Please refer to the (license)[#license] for more details on usage guidelines.
