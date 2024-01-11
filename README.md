# isx: pure-python API to read Inscopix data

![](https://github.com/inscopix/py_isx/actions/workflows/main.yml/badge.svg) 
![](https://img.shields.io/pypi/v/isx)

Experimental pure-python API to read Inscopix ISXD files. 
Please note that this is a work in progress and not feature complete. 
Use at your own risk. 



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

## Testing

This code is tested using GitHub Actions on the following python
versions:

- 3.9
- 3.10
- 3.11
- 3.12
