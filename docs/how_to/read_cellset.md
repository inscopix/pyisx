## Make sure `py_isx` is installed
```python
import py_isx
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
# Pick a number less than `num_cells`
cell_trace = cell_set.get_cell_trace(0)
```

### Get the name of a cell
```python
cell_name = cell_set.get_cell_name(0)
```

### Get the status of a cell
```python
cell_name = cell_set.get_cell_status(0)
```

#### Alternatively, loop over all cells in the file
```python
for cell_idx in range(num_cells):
    cell_trace = cell_set.get_cell_trace(cell_idx)
    cell_name = cell_set.get_cell_name(cell_idx)
    cell_status = cell_set.get_cell_status(cell_idx)
```