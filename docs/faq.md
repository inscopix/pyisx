# Frequently Asked Questions

## Can I install/use the IDPS API and this API at the same time?


It is not recommended to install or use the IDPS API and this API in the same python environment. 

This package is meant to be a drop-in replacement for the IDPS Python API, if the only use case of the IDPS Python API is to read Inscopix data files. 

## Can I write ISXD files using this API?

No. 

## Can I read `file_x` using this API?

This API only support a subset of Inscopix data types at
the moment of writing. See [this :material-arrow-top-right:](https://inscopix.github.io/py_isx/index.html#data-support) for a list of supported types.

