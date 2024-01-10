# isx: a pure-python API to read Inscopix data

`isx` is a lightweight pure-python API for reading 
Inscopix data files. 


<div class="grid cards" markdown>

- :fontawesome-brands-python: **Pure python**       
Lightweight implementation, written in pure Python with no C++ code. Works on any system that can run Python.
- :fontawesome-solid-handshake-simple: **Mirrors IDPS API**  
Acts as a drop-in replacement to the IDPS API, meaning that you don't have to rewrite existing code that reads ISXD data.
- :octicons-unlock-24: **Freely available**         
Freely available for non-commerical use.
- :material-download: **Easy install**     
Just `pip install isx`.

</div>



!!! tip "`isx` or `py_isx`?"
    This repository is called `py_isx`, but defines a package called `isx`. Therefore, you would import this as follows:

    ```python
    import isx
    ```

    The reason for this is so that it can be used as a back-end for code that was written for the IDPS Python API, without requiring any change in user code.

## User Support

!!! Warning "No user support"
    This repository is provided as-is, with no ongoing support from Inscopix. 

## Data Support

Currently, `isx` supports only a subset of Inscopix data types.


|  File type | Support |
|  --------- | ------- |
| ISXD CellSet   | ✅ |
| ISXD Movie   | ✅ |
| GPIO data   | ❌ |
| ISXD Events   | ❌ |
| ISXD VesselSet   | ❌ |

## License 

`isx` has been released under a [CC BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/).

This means that:

 You are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material
    The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:

- Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes .
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
