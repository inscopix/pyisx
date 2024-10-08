
# pyisx documentation

`isx` is a python package for interacting with Inscopix data.
This package encapsulates the following I/O functionality:

* Reading Inscopix files (`.isxd`, `.isxb`, `.gpio`, `.imu`)
* Writing Inscopix files (`.isxd`)
* Exporting Inscopix files to third-party formats (`.mp4`, `.tiff`, `.csv`)

The `isx` package is built from the `pyisx` project, a python binding for [isxcore](https://github.com/inscopix/isxcore), a C++ API for interacting with Inscopix data.

::: {toctree}
:maxdepth: 1
:caption: Contents:

overview
installation
reference
examples
contributing
faq
:::

## Links

* [Github](https://github.com/inscopix/pyisx)
* [isxcore Github](https://github.com/inscopix/isxcore)
* [Inscopix](https://inscopix.com/)

## Quick Start

To install `isx`, run the following command in a python environment:

```python
pip install isx
```

## License

This project has been released under a [CC BY-NC license](https://creativecommons.org/licenses/by-nc/4.0). This means that you are free to:

* Share — copy and redistribute the material in any medium or format
* Adapt — remix, transform, and build upon the material

   * The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:

* Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* NonCommercial — You may not use the material for commercial purposes .
* No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
