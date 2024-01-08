# py_isx

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
poetry add https://github.com/inscopix/py_isx
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
