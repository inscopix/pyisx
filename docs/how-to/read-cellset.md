# How to read ISXD Cellset files


We first import `py_isx` using:


```python
import isx
```

## Read a CellSet

CellSets can be easily opened in Python using the `isx.CellSet.read` method.
```python
cell_set = isx.CellSet.read('cellset.isxd')
```
Now that you have a `CellSet` object, you can use it to access the data in the file.

### Get the number of cells in the CellSet
```python
num_cells = cell_set.num_cells
```
With this information, you can pull the data from specific cells, or iterate over all of the cells in the file.

### Get the trace from a cell

```python


