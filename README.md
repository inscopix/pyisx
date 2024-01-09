# py_isx

![](https://github.com/inscopix/py_isx/actions/workflows/main.yml/badge.svg)

Experimental pure-python API to read ISXD files. Please note 
that this is a work in progress and not feature complete. 
Use at your own risk. 



## Support

|  File type | Support |
|  --------- | ------- |
| CellSet   | ✅ |
| Movie   | ✅ |
| Events   | ❌ |
| VesselSet   | 🚧 |
| GPIO files   | ❌ |
| IMU files   | ❌ |

## Installation

### Poetry

```bash
poetry add git+ssh://git@github.com/inscopix/py_isx.git
```

### pip


```bash
pip install git+https://github.com/inscopix/py_isx.git@main
```

## Testing

This code is tested using GitHub Actions on the following python
versions:

- 3.9
- 3.10
- 3.11
- 3.12
