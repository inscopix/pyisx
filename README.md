# isx: pure-python API to read Inscopix data

![](https://github.com/inscopix/py_isx/actions/workflows/main.yml/badge.svg) 
![](https://img.shields.io/pypi/v/isx)

This is a pure-python API to read Inscopix ISXD files. 


## Documentation

[Read the documentation](https://inscopix.github.io/py_isx/)

## Support

|  File type | Support |
|  --------- | ------- |
| ISXD CellSet   | ✅ |
| ISXD Movie   | ✅ |
| ISXD Movie (multi-plane)   | ❌ |
| ISXD Movie (dual-color)   | ❌ |
| GPIO data   | ❌ |
| ISXD Events   | ❌ |
| ISXD VesselSet   | ❌ |


## Install

### Poetry

```bash
poetry add isx
```

### pip


```bash
pip install isx
```

## Caution

This is a work in progress, and all reading functions in the IDPS Python API are not supported yet. 


## Testing

This code is tested using GitHub Actions on the following python
versions:

- 3.9
- 3.10
- 3.11
- 3.12
